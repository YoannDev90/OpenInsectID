"""INT8 quantization and TFLite conversion (section 5 of the plan).

Two PTQ paths:
- PyTorch dynamic INT8 (fast, weight-only) — handy for a quick memory check.
- ONNX Runtime static INT8 with a calibration set (full PTQ, activation +
  weight), which is the path that maps onto NNAPI/Hexagon DSP on-device.

``TFLiteINT8Converter`` turns the quantized ONNX graph into a ``.tflite`` file
using onnx2tf when available, otherwise it falls back to a documented manual
step (the ONNX -> TFLite bridge moves fast; we avoid pinning a fragile dep).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import torch
import torch.nn as nn


def quantize_dynamic_int8(module: nn.Module) -> nn.Module:
    """PyTorch dynamic (weight-only) INT8 quantization of Linear layers.

    Returns a quantized copy. Conv layers stay FP32 — use static ONNX PTQ for
    full INT8 including activations.
    """
    module = module.eval().cpu()
    return torch.ao.quantization.quantize_dynamic(
        module, {nn.Linear}, dtype=torch.qint8
    )


class _CalibrationReader:
    """ONNX Runtime calibration data reader over a batch of images."""

    def __init__(self, calibration_images: torch.Tensor, input_name: str):
        self.images = calibration_images
        self.input_name = input_name
        self._i = 0

    def get_next(self):
        if self._i >= len(self.images):
            return None
        batch = self.images[self._i : self._i + 1].numpy()
        self._i += 1
        return {self.input_name: batch}


def quantize_static_int8(
    onnx_path: str | Path,
    out_path: str | Path,
    calibration_images: torch.Tensor,
) -> Path:
    """ONNX Runtime static INT8 PTQ with a calibration image tensor.

    Args:
        onnx_path: source FP32 ``.onnx``.
        out_path: destination ``.onnx``.
        calibration_images: (N, 3, H, W) tensor of representative images.
    """
    try:
        from onnxruntime.quantization import (
            CalibrationDataReader,
            QuantFormat,
            QuantType,
            quantize_static,
        )
        import numpy as np
    except ImportError as e:  # pragma: no cover - optional dependency
        raise ImportError(
            "Static INT8 PTQ requires 'onnxruntime' and 'onnx': "
            "pip install onnx onnxruntime"
        ) from e

    import onnx

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Discover the graph input name.
    model = onnx.load(str(onnx_path))
    input_name = model.graph.input[0].name

    class _Reader(CalibrationDataReader):
        def __init__(self):
            self._i = 0
            self._imgs = calibration_images

        def get_next(self):
            if self._i >= len(self._imgs):
                return None
            batch = self._imgs[self._i : self._i + 1].numpy().astype(np.float32)
            self._i += 1
            return {input_name: batch}

    quantize_static(
        model_input=str(onnx_path),
        model_output=str(out_path),
        calibration_data_reader=_Reader(),
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
    )
    return out_path


class TFLiteINT8Converter:
    """Convert a (quantized) ONNX graph to ``.tflite``.

    Prefers the ``onnx2tf`` CLI when installed. Otherwise raises with the exact
    install/run command so the notebook can print a actionable fallback rather
    than crashing on a missing native dep.
    """

    def __init__(self, onnx2tf_bin: str = "onnx2tf"):
        self.onnx2tf = onnx2tf_bin

    def available(self) -> bool:
        return shutil.which(self.onnx2tf) is not None

    def convert(
        self,
        onnx_path: str | Path,
        out_dir: str | Path,
        int8: bool = True,
        calibration_dir: str | Path | None = None,
    ) -> Path:
        onnx_path = Path(onnx_path)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if not self.available():
            raise RuntimeError(
                "onnx2tf not found. Install with:\n"
                "  pip install onnx2tf\n"
                "(it pins its own onnx/onnxruntime/onnxscript stack; no more\n"
                " tflite-support or tensorflow needed) or run the CLI:\n"
                f"  onnx2tf -i {onnx_path} -o {out_dir}"
                + (" -nuo" if int8 else "")
            )
        cmd = [self.onnx2tf, "-i", str(onnx_path), "-o", str(out_dir)]
        if not int8:
            cmd.append("-nuo")  # do not quantize, keep float
        subprocess.run(cmd, check=True)
        # onnx2tf names the output after the input stem.
        candidate = out_dir / f"{onnx_path.stem}.tflite"
        if candidate.exists():
            return candidate
        found = list(out_dir.glob("*.tflite"))
        if not found:
            raise FileNotFoundError(f"onnx2tf produced no .tflite in {out_dir}")
        return found[0]
