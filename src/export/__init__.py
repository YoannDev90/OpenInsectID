"""Model export: PyTorch -> ONNX -> (INT8 quantized) -> TFLite.

Section 5 of the plan:
- Per-module export (router + each expert) since the cascade is data-dependent
  and cannot be one static graph.
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
