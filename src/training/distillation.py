"""Knowledge Distillation from a BioCLIP teacher into MobileNetV4 experts.

Section 2.2 of the plan: transfer BioCLIP's biologically-informed visual
features (pretrained on the iNaturalist tree of life) into the compact
edge-bound experts. The teacher is frozen; we match its softened logits.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    """KD = T^2 * KL(student_T || teacher_T), optionally blended with hard CE.

    Args:
        temperature: softmax temperature T (higher = softer targets).
        alpha: weight of the KD term vs. the hard-label CE term
            (1.0 = pure distillation, 0.0 = pure supervised).
    """

    def __init__(self, temperature: float = 4.0, alpha: float = 0.5):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        self.temperature = temperature
        self.alpha = alpha

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        target: torch.Tensor | None = None,
    ) -> torch.Tensor:
        t = self.temperature
        kd = F.kl_div(
            F.log_softmax(student_logits / t, dim=-1),
            F.log_softmax(teacher_logits / t, dim=-1),
            reduction="batchmean",
        ) * (t * t)
        if target is None or self.alpha >= 1.0:
            return self.alpha * kd
        ce = F.cross_entropy(student_logits, target, ignore_index=-100)
        return self.alpha * kd + (1.0 - self.alpha) * ce


def load_bioclip_teacher(pretrained: str = "imageomics/bioclip",
                         device: str | torch.device = "cpu") -> nn.Module:
    """Load a frozen BioCLIP image encoder as a distillation teacher.

    Requires the optional ``transformers`` + ``torch`` dependencies:
        pip install transformers

    The returned module maps images to a contrastive embedding space; use its
    visual projection output as the teacher signal. Adjust the attribute path
    if you swap in a different BioCLIP checkpoint.
    """
    try:
        from transformers import AutoModel
    except ImportError as e:  # pragma: no cover - optional dependency
        raise ImportError(
            "BioCLIP teacher requires 'transformers': pip install transformers"
        ) from e
    model = AutoModel.from_pretrained(pretrained)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return BioCLIPWrapper(model)


class BioCLIPWrapper(nn.Module):
    """Expose BioCLIP's pooled visual embedding as (B, D) teacher logits proxy.

    Adapts to common BioCLIP HF layouts (``vision_model`` / ``model.vision_model``
    / direct ``get_image_features``).
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def _features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        m = self.model
        if hasattr(m, "get_image_features"):
            return m.get_image_features(pixel_values=pixel_values)
        vision = getattr(m, "vision_model", None) or getattr(m, "model", m)
        out = vision(pixel_values=pixel_values)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            return out.pooler_output
        return out.last_hidden_state[:, 0]

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self._features(pixel_values)
