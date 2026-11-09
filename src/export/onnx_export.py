"""ONNX export for the shared trunk and the 4 level heads.

The 4-level cascade is data-dependent (the chosen child head depends on the
previous level's argmax), so it cannot be a single static graph. The shared
trunk collapses the payload to 5 modules:

- ``trunk.onnx``: image (1, 3, H, W) -> features (1, D)
- ``head_{order,family,genus,species}.onnx``: features (1, D) -> that level's
  global logits (1, n_level)

The child ranges needed to slice a head on-device (which rows belong to which
parent) are written into ``manifest.json``, so the app never recomputes the
taxonomy layout.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn


def export_module(
    module: nn.Module,
    out_path: str | Path,
    input_shape: tuple[int, ...],
    input_name: str = "image",
    opset: int = 18,
    batch_size: int = 1,
    dynamic_batch: bool = False,
) -> Path:
    """Export one module (trunk or head) to ONNX.

    Args:
        module: the module to export.
        out_path: destination ``.onnx`` file.
        input_shape: shape of the single input *excluding* the batch axis, e.g.
            ``(3, 224, 224)`` for the trunk, ``(D,)`` for a head.
        input_name: graph input name (``image`` for the trunk, ``features``
            for the heads).
        opset: ONNX opset version.
        dynamic_batch: make the batch axis dynamic (usually keep False for
            single-image on-device inference).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    module = module.eval().cpu()
    dummy = torch.randn(batch_size, *input_shape)
    dynamic_axes = (
        {input_name: {0: "batch"}, "logits": {0: "batch"}} if dynamic_batch else None
    )
    torch.onnx.export(
        module,
        dummy,
        str(out_path),
        input_names=[input_name],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
    )
    return out_path


def export_moe(
    model,
    out_dir: str | Path,
    image_size: int = 224,
    opset: int = 18,
) -> dict[str, Path]:
    """Export the trunk + the 4 heads of a ``HierarchicalMoE``.

    Writes 5 ``.onnx`` files plus a ``manifest.json`` describing the cascade
    (labels per level, child ranges per level, feature dim, image size).
    Returns {module_name: onnx_path}.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    feature_dim = int(model.heads.feature_dim)
    manifest = {
        "image_size": image_size,
        "feature_dim": feature_dim,
        "modules": {},
    }

    for name, module in model.export_modules():
        if name == "trunk":
            input_shape, input_name = (3, image_size, image_size), "image"
        else:  # head_*: consumes the trunk's features
            input_shape, input_name = (feature_dim,), "features"
        path = export_module(
            module, out_dir / f"{name}.onnx", input_shape, input_name, opset
        )
        paths[name] = path
        if name == "trunk":
            manifest["modules"][name] = {
                "input": input_name,
                "num_outputs": feature_dim,
            }
        else:
            manifest["modules"][name] = {
                "input": input_name,
                "num_outputs": model.heads.sizes[name.removeprefix("head_")],
            }

    # Label metadata for each level (global label order = index order).
    manifest["labels"] = {
        "order": model.order_names,
        "family": model.family_names,
        "genus": model.genus_names,
        "species": model.species_names,
    }
    # Child ranges: for level L with parents at level P, parent p owns the
    # head rows [ranges[p], ranges[p+1]) — the on-device slicing index.
    manifest["ranges"] = {
        level: getattr(model.heads, f"range_{level}").tolist()
        for level in ("family", "genus", "species")
    }
    manifest["parents"] = {
        "family": model.order_names,
        "genus": model.family_names,
        "species": model.genus_names,
    }

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return paths
