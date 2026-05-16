"""
EdgeForge -- INT8 Quantizer
Wraps onnxruntime.quantization with MCU-aware defaults.
Detects already-quantised models and skips re-quantisation.
"""
from __future__ import annotations

import logging
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto

log = logging.getLogger(__name__)


@dataclass
class QuantizeReport:
    mode:                str
    flash_before_kb:     float
    flash_after_kb:      float
    flash_reduction_pct: float
    nodes_quantized:     int
    accuracy_delta_est:  str
    already_quantized:   bool = False


def is_already_quantized(model_path: Path) -> bool:
    """
    Detect whether a model is already INT8 quantised.
    Checks for DequantizeLinear / QuantizeLinear nodes which indicate
    a QAT or post-training quantised model.
    """
    try:
        m = onnx.load(str(model_path))
        quant_ops = {"DequantizeLinear", "QuantizeLinear", "QLinearConv",
                     "QLinearMatMul", "QLinearAdd", "QLinearMul"}
        for node in m.graph.node:
            if node.op_type in quant_ops:
                return True
        return False
    except Exception:
        return False


def _preprocess(model_path: Path, output_path: Path) -> None:
    """Run onnxruntime shape inference pre-processing."""
    try:
        from onnxruntime.quantization.shape_inference import quant_pre_process
        quant_pre_process(str(model_path), str(output_path), skip_symbolic_shape=True)
    except Exception:
        shutil.copy2(model_path, output_path)


def quantize_dynamic(
    model_path:  str | Path,
    output_path: str | Path,
) -> QuantizeReport:
    """
    Apply dynamic INT8 quantisation -- weights only.
    Skips gracefully if model is already quantised.
    """
    try:
        from onnxruntime.quantization import quantize_dynamic as _qd, QuantType
    except ImportError:
        raise ImportError("onnxruntime not found. Install with: pip install onnxruntime")

    src = Path(model_path)
    dst = Path(output_path)
    dst.parent.mkdir(parents=True, exist_ok=True)

    flash_before_kb = src.stat().st_size / 1024

    # Already quantised -- just copy through, no re-quantisation
    if is_already_quantized(src):
        log.info("Model is already quantised -- skipping re-quantisation")
        shutil.copy2(src, dst)
        flash_after_kb = dst.stat().st_size / 1024
        return QuantizeReport(
            mode="dynamic",
            flash_before_kb=flash_before_kb,
            flash_after_kb=flash_after_kb,
            flash_reduction_pct=0.0,
            nodes_quantized=_count_quantized_nodes(dst),
            accuracy_delta_est="model already quantised -- no further reduction applied",
            already_quantized=True,
        )

    with tempfile.TemporaryDirectory(prefix="edgeforge_preproc_") as tmp:
        preprocessed = Path(tmp) / "preprocessed.onnx"
        _preprocess(src, preprocessed)

        _qd(
            model_input=str(preprocessed),
            model_output=str(dst),
            weight_type=QuantType.QInt8,
            op_types_to_quantize=["Conv", "MatMul", "Gemm", "ConvTranspose"],
            per_channel=False,
            reduce_range=False,
            extra_options={"DefaultTensorType": TensorProto.FLOAT},
        )

    flash_after_kb  = dst.stat().st_size / 1024
    reduction       = (1 - flash_after_kb / flash_before_kb) * 100
    nodes_quantized = _count_quantized_nodes(dst)

    return QuantizeReport(
        mode="dynamic",
        flash_before_kb=flash_before_kb,
        flash_after_kb=flash_after_kb,
        flash_reduction_pct=reduction,
        nodes_quantized=nodes_quantized,
        accuracy_delta_est="< 1% typical for CNN classification models",
    )


def quantize_static(
    model_path:       str | Path,
    output_path:      str | Path,
    calibration_data: list[np.ndarray] | None = None,
    calibration_dir:  str | Path | None = None,
) -> QuantizeReport:
    """Apply static INT8 quantisation -- weights + activations."""
    try:
        from onnxruntime.quantization import (
            quantize_static as _qs, QuantType, QuantFormat,
        )
    except ImportError:
        raise ImportError("onnxruntime not found. Install with: pip install onnxruntime")

    src = Path(model_path)
    dst = Path(output_path)
    dst.parent.mkdir(parents=True, exist_ok=True)

    if calibration_data is None and calibration_dir is None:
        raise ValueError(
            "Static quantisation requires calibration data. "
            "Provide calibration_data (list of arrays) or calibration_dir. "
            "Use quantize_dynamic() if you don't have calibration data."
        )

    # Already quantised -- just copy through
    if is_already_quantized(src):
        log.info("Model is already quantised -- skipping re-quantisation")
        shutil.copy2(src, dst)
        flash_kb = dst.stat().st_size / 1024
        return QuantizeReport(
            mode="static",
            flash_before_kb=flash_kb,
            flash_after_kb=flash_kb,
            flash_reduction_pct=0.0,
            nodes_quantized=_count_quantized_nodes(dst),
            accuracy_delta_est="model already quantised -- no further reduction applied",
            already_quantized=True,
        )

    samples = calibration_data or _load_calibration_dir(calibration_dir)
    if not samples:
        raise ValueError("No calibration samples found.")

    flash_before_kb = src.stat().st_size / 1024

    with tempfile.TemporaryDirectory(prefix="edgeforge_preproc_") as tmp:
        preprocessed = Path(tmp) / "preprocessed.onnx"
        _preprocess(src, preprocessed)

        model    = onnx.load(str(preprocessed))
        inp_name = model.graph.input[0].name
        reader   = _NumpyCalibrationReader(inp_name, samples)

        _qs(
            model_input=str(preprocessed),
            model_output=str(dst),
            calibration_data_reader=reader,
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            per_channel=False,
            reduce_range=False,
            op_types_to_quantize=["Conv", "MatMul", "Gemm", "ConvTranspose"],
            extra_options={"DefaultTensorType": TensorProto.FLOAT},
        )

    flash_after_kb  = dst.stat().st_size / 1024
    reduction       = (1 - flash_after_kb / flash_before_kb) * 100
    nodes_quantized = _count_quantized_nodes(dst)

    return QuantizeReport(
        mode="static",
        flash_before_kb=flash_before_kb,
        flash_after_kb=flash_after_kb,
        flash_reduction_pct=reduction,
        nodes_quantized=nodes_quantized,
        accuracy_delta_est="< 0.5% typical with good calibration data",
    )


def estimate_size_reduction(
    original_path:  str | Path,
    quantized_path: str | Path,
) -> dict:
    """Compare original and quantised model file sizes."""
    before_kb     = Path(original_path).stat().st_size  / 1024
    after_kb      = Path(quantized_path).stat().st_size / 1024
    reduction_kb  = before_kb - after_kb
    reduction_pct = (reduction_kb / before_kb * 100) if before_kb > 0 else 0.0
    return {
        "before_kb":     round(before_kb,     1),
        "after_kb":      round(after_kb,      1),
        "reduction_kb":  round(reduction_kb,  1),
        "reduction_pct": round(reduction_pct, 1),
    }


# ── Internal helpers ─────────────────────────────────────────────────────────

def _count_quantized_nodes(model_path: Path) -> int:
    try:
        m = onnx.load(str(model_path))
        return sum(
            1 for n in m.graph.node
            if "QLinear" in n.op_type or "QuantizeLinear" in n.op_type
        )
    except Exception:
        return 0


def _load_calibration_dir(cal_dir: str | Path) -> list[np.ndarray]:
    return [np.load(str(f)) for f in sorted(Path(cal_dir).glob("*.npy"))]


class _NumpyCalibrationReader:
    def __init__(self, input_name: str, samples: list[np.ndarray]):
        self._name    = input_name
        self._samples = iter(samples)

    def get_next(self) -> dict | None:
        try:
            arr = next(self._samples)
            if arr.ndim < 2:
                arr = arr[np.newaxis, ...]
            return {self._name: arr.astype(np.float32)}
        except StopIteration:
            return None

    def rewind(self) -> None:
        pass
