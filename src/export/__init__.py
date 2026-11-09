"""Model export: PyTorch -> ONNX -> (INT8 quantized) -> TFLite.

Section 5 of the plan:
- The cascade is data-dependent (head selection follows the previous level's
  argmax), so it exports as 5 modules: one trunk (image -> features) and one
  head per level (features -> global logits), with child ranges in the
  manifest for on-device slicing.
- Post-training INT8 quantization with a calibration set, plus an optional
  quantization-aware-training hook.
- ONNX -> TFLite conversion.
"""

from .onnx_export import export_module, export_moe
from .quantize import (
    quantize_dynamic_int8,
    quantize_static_int8,
    TFLiteINT8Converter,
)

__all__ = [
    "export_module",
    "export_moe",
    "quantize_dynamic_int8",
    "quantize_static_int8",
    "TFLiteINT8Converter",
]
