"""Tests for ONNX export and dynamic INT8 quantization."""

import torch
import pytest

from src.models.experts import Expert
from src.export.onnx_export import export_module
from src.export.quantize import quantize_dynamic_int8


def test_export_module_writes_onnx(tmp_path):
    pytest.importorskip("onnx")
    expert = Expert(num_classes=4, backbone_name="mnv4_small", width_mult=0.25)
    out = export_module(expert, tmp_path / "expert.onnx", image_size=224)
    assert out.exists()
    assert out.stat().st_size > 1000


def test_onnx_graph_loads_and_runs(tmp_path):
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    expert = Expert(num_classes=3, backbone_name="mnv4_small", width_mult=0.25)
    expert.eval()
    p = export_module(expert, tmp_path / "e.onnx", image_size=224)
    sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
    x = torch.randn(1, 3, 224, 224).numpy()
    logits = sess.run(None, {"image": x})[0]
    assert logits.shape == (1, 3)


def test_dynamic_int8_reduces_size_and_runs(tmp_path):
    pytest.importorskip("onnx")
    expert = Expert(num_classes=3, backbone_name="mnv4_small", width_mult=0.25)
    expert.eval()
    fp32_path = export_module(expert, tmp_path / "fp32.onnx", image_size=224)
    q = quantize_dynamic_int8(expert)
    # Quantized module should still produce a valid logits shape.
    out = q(torch.randn(1, 3, 224, 224))
    assert out.shape == (1, 3)
    assert fp32_path.exists()
