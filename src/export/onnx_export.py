"""ONNX export for the router and each expert module.

The 4-level cascade is data-dependent (the chosen expert depends on the
previous level's argmax), so it cannot be a single static graph. Each router
and expert is exported independently, keyed by its taxonomic node; the mobile
app loads them on demand (Option A in the plan).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn


def export_module(
    module: nn.Module,
    out_path: str | Path,
    image_size: int = 224,
    opset: int = 17,
    batch_size: int = 1,
    dynamic_batch: bool = False,
) -> Path:
    """Export one expert/router to ONNX.

    Args:
        module: the ``Expert`` (backbone + linear head) to export.
        out_path: destination ``.onnx`` file.
        image_size: square input resolution.
        opset: ONNX opset version.
        dynamic_batch: make the batch axis dynamic (usually keep False for
            single-image on-device inference).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    module = module.eval().cpu()
    dummy = torch.randn(batch_size, 3, image_size, image_size)
    dynamic_axes = (
        {"image": {0: "batch"}, "logits": {0: "batch"}} if dynamic_batch else None
    )
    torch.onnx.export(
        module,
        dummy,
        str(out_path),
        input_names=["image"],
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
    opset: int = 17,
) -> dict[str, Path]:
    """Export the router and every expert of a ``HierarchicalMoE``.

    Writes one ``.onnx`` per module plus a ``manifest.json`` describing the
    cascade (which expert belongs to which taxonomic node, class label lists).
    Returns {module_name: onnx_path}.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    manifest = {"image_size": image_size, "modules": {}}

    for name, module in model.export_modules():
        path = export_module(module, out_dir / f"{name}.onnx", image_size, opset)
        paths[name] = path

    # Label metadata for each level (global label order = index order).
    manifest["labels"] = {
        "order": model.order_names,
        "family": model.family_names,
        "genus": model.genus_names,
        "species": model.species_names,
    }
    # Map each expert module to the parent node + its local class list.
    for level, experts in (
        ("family", model.family_experts),
        ("genus", model.genus_experts),
        ("species", model.species_experts),
    ):
        for key in experts.keys():
            name = f"{level}_{key.split('_', 1)[1]}_{int(key.split('_', 1)[0])}"
            manifest["modules"][name] = {
                "level": level,
                "parent_key": key,
                "num_classes": int(experts[key].num_classes),
            }
    if model.router is not None:
        manifest["modules"]["router"] = {
            "level": "order",
            "num_classes": int(model.router.num_classes),
        }

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return paths
