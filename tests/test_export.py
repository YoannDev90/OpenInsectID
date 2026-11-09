"""Tests for ONNX export and dynamic INT8 quantization."""

import torch
import pytest

from src.models import HierarchicalMoE, TaxonomyTree
from src.export.onnx_export import export_module, export_moe
from src.export.quantize import quantize_dynamic_int8


RECORDS = [
    ("Coleoptera", "Carabidae", "Carabus", "Carabus auratus"),
    ("Coleoptera", "Carabidae", "Carabus", "Carabus insulicola"),
    ("Lepidoptera", "Nymphalidae", "Vanessa", "Vanessa atalanta"),
    ("Lepidoptera", "Nymphalidae", "Vanessa", "Vanessa cardui"),
]


@pytest.fixture(scope="module")
def model():
    tree = TaxonomyTree(RECORDS)
    return HierarchicalMoE(tree, backbone_name="mnv4_small", width_mult=0.25)


def test_export_moe_writes_five_modules(tmp_path, model):
    pytest.importorskip("onnx")
    paths = export_moe(model, tmp_path, image_size=224)
    assert set(paths) == {"trunk", "head_order", "head_family", "head_genus", "head_species"}
    for p in paths.values():
        assert p.exists() and p.stat().st_size > 1000
    manifest = (tmp_path / "manifest.json").read_text()
    assert '"ranges"' in manifest and '"labels"' in manifest


def test_onnx_graph_loads_and_runs(tmp_path, model):
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    paths = export_moe(model, tmp_path, image_size=224)

    trunk = ort.InferenceSession(str(paths["trunk"]), providers=["CPUExecutionProvider"])
    x = torch.randn(1, 3, 224, 224).numpy()
    feats = trunk.run(None, {"image": x})[0]
    assert feats.shape == (1, model.heads.feature_dim)

    head = ort.InferenceSession(str(paths["head_order"]), providers=["CPUExecutionProvider"])
    logits = head.run(None, {"features": feats})[0]
    assert logits.shape == (1, model.n_orders)


def test_dynamic_int8_reduces_size_and_runs(model):
    pytest.importorskip("onnx")
    trunk = model.trunk
    q = quantize_dynamic_int8(trunk)
    out = q(torch.randn(1, 3, 224, 224))
    assert out.shape == (1, model.heads.feature_dim)


def test_head_export_is_features_to_logits(tmp_path, model):
    """A head must take (1, D) features and emit its global label logits."""
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    paths = export_moe(model, tmp_path, image_size=224)
    head = ort.InferenceSession(
        str(paths["head_species"]), providers=["CPUExecutionProvider"]
    )
    assert head.get_inputs()[0].shape == [1, model.heads.feature_dim]
    logits = head.run(
        None, {"features": torch.randn(1, model.heads.feature_dim).numpy()}
    )[0]
    assert logits.shape == (1, model.n_species)
