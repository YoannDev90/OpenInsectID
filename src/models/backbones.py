"""MobileNetV4-style backbones built from UIB (Universal Inverted Bottleneck) blocks.

Layer tables follow the MobileNetV4 paper (Qiu et al., 2024, arXiv:2404.10518):
UIB = [extra depthwise] -> expand 1x1 -> depthwise kxk -> squeeze 1x1, with an
identity shortcut when stride == 1 and channel count is preserved.

A ``timm:`` prefixed name transparently loads any backbone from the ``timm``
library instead (optional dependency, e.g. ``timm:mobilenetv5_base`` for the
MobileNetV5 encoder released in June 2025 as the Gemma 3n vision encoder).
Note: the only pretrained MobileNetV5 weights are published under the Gemma
license and the architecture is a ~300M-parameter encoder designed for
768x768 inputs — useful for distillation experiments, not for small edge
experts. MobileNetV4 Conv-S/M stays the default on-device backbone.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def make_divisible(v: float, divisor: int = 8, min_value: int | None = None) -> int:
    """Round channel count to the nearest multiple of ``divisor`` (MobileNet convention)."""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:  # ensure rounding never loses more than 10%
        new_v += divisor
    return new_v


class ConvBNAct(nn.Sequential):
    def __init__(
        self,
        inp: int,
        oup: int,
        kernel: int = 3,
        stride: int = 1,
        groups: int = 1,
        act: bool = True,
    ):
        layers: list[nn.Module] = [
            nn.Conv2d(
                inp, oup, kernel, stride, kernel // 2, groups=groups, bias=False
            ),
            nn.BatchNorm2d(oup),
        ]
        if act:
            layers.append(nn.SiLU(inplace=True))
        super().__init__(*layers)


class UIB(nn.Module):
    """Universal Inverted Bottleneck block (MobileNetV4).

    Optional extra depthwise conv -> 1x1 expand -> kxk depthwise (strided) ->
    1x1 squeeze, with residual shortcut when applicable.
    """

    def __init__(
        self,
        inp: int,
        oup: int,
        stride: int = 1,
        expansion: float = 4.0,
        kernel: int = 3,
        extra_dw: bool = False,
    ):
        super().__init__()
        hidden = int(round(inp * expansion))
        self.use_skip = stride == 1 and inp == oup

        layers: list[nn.Module] = []
        if extra_dw and stride == 1:
            layers.append(ConvBNAct(inp, inp, 3, 1, groups=inp))
        layers += [
            ConvBNAct(inp, hidden, 1, 1),  # expand
            ConvBNAct(hidden, hidden, kernel, stride, groups=hidden),  # depthwise
            nn.Conv2d(hidden, oup, 1, 1, 0, bias=False),  # squeeze (linear)
            nn.BatchNorm2d(oup),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        return x + out if self.use_skip else out


# Stage tables: (expansion, out_channels, stride, kernel, extra_dw, repeats).
# First UIB of each stage carries the stride; repeats are stride-1.
_MNV4_SMALL = dict(
    stem=24,
    head=1024,
    stages=[
        (2, 24, 1, 3, False, 1),
        (4, 48, 2, 3, True, 1),
        (4, 48, 1, 3, False, 1),
        (4, 96, 2, 5, True, 1),
        (4, 96, 1, 5, False, 1),
        (4, 192, 2, 5, True, 1),
        (4, 192, 1, 5, False, 4),
        (6, 192, 1, 5, False, 1),
        (6, 960, 2, 5, True, 1),
    ],
)

_MNV4_MEDIUM = dict(
    stem=32,
    head=2048,
    stages=[
        (2, 24, 1, 3, False, 1),
        (4, 48, 2, 3, True, 1),
        (4, 48, 1, 3, False, 1),
        (4, 96, 2, 5, True, 1),
        (4, 96, 1, 5, False, 2),
        (4, 192, 2, 5, True, 1),
        (4, 192, 1, 5, False, 5),
        (4, 384, 2, 5, True, 1),
        (4, 384, 1, 5, False, 10),
        (4, 576, 1, 5, False, 7),
        (6, 960, 2, 5, True, 1),
    ],
)

BACKBONE_TABLES = {
    "mnv4_small": _MNV4_SMALL,
    "mnv4_medium": _MNV4_MEDIUM,
}


class MobileNetV4(nn.Module):
    """MobileNetV4-style UIB stack returning pooled features of shape (B, head)."""

    def __init__(self, cfg: dict, width_mult: float = 1.0):
        super().__init__()
        stem = make_divisible(cfg["stem"] * width_mult)
        head = make_divisible(cfg["head"] * max(width_mult, 0.75))

        self.stem = ConvBNAct(3, stem, 3, 2)

        blocks: list[nn.Module] = []
        inp = stem
        for exp, ch, stride, kernel, extra_dw, repeats in cfg["stages"]:
            out = make_divisible(ch * width_mult)
            for i in range(repeats):
                blocks.append(
                    UIB(inp, out, stride if i == 0 else 1, exp, kernel, extra_dw)
                )
                inp = out
        self.blocks = nn.Sequential(*blocks)
        self.head_conv = ConvBNAct(inp, head, 1, 1)
        self.out_channels = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head_conv(x)
        # Global average pooling as a plain mean: mathematically identical to
        # AdaptiveAvgPool2d(1), but AdaptiveAvgPool2d's backward pass is one
        # of the CUDA ops PyTorch flags as nondeterministic — it raises under
        # torch.use_deterministic_algorithms(True), which the trainer enables.
        return x.mean((-2, -1))


class TimmBackbone(nn.Module):
    """Thin wrapper exposing a uniform ``out_channels`` API over a timm model."""

    def __init__(self, name: str, pretrained: bool = False, **kwargs):
        super().__init__()
        try:
            import timm
        except ImportError as e:  # pragma: no cover - optional dependency
            raise ImportError(
                "Loading a timm backbone requires the 'timm' package: "
                "pip install timm"
            ) from e
        self.model = timm.create_model(name, pretrained=pretrained, **kwargs)
        self.out_channels = int(getattr(self.model, "num_features", 0)) or int(
            self.model.head.in_features
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def build_backbone(name: str = "mnv4_small", width_mult: float = 1.0, **kwargs):
    """Build a backbone by name.

    - ``mnv4_small`` / ``mnv4_medium``: native MobileNetV4-style UIB stacks.
    - ``timm:<model>``: any timm model (requires timm), e.g. ``timm:mobilenetv5_base``.
    """
    if name.startswith("timm:"):
        return TimmBackbone(name[len("timm:") :], **kwargs)
    if name not in BACKBONE_TABLES:
        raise ValueError(
            f"Unknown backbone '{name}'. Available: {sorted(BACKBONE_TABLES)} "
            "or 'timm:<model_name>'."
        )
    return MobileNetV4(BACKBONE_TABLES[name], width_mult=width_mult)


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
