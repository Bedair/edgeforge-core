"""
EdgeForge — ONNX IR Converter
Converts any supported model format to ONNX as the unified internal IR.
Each converter is a thin, opinionated wrapper around existing tools.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from .detector import ModelFormat, detect


class ConversionError(Exception):
    """Raised when a model cannot be converted to ONNX."""
    pass


def to_onnx(
    src: str | Path,
    dst: str | Path | None = None,
    opset: int = 17,
) -> Path:
    """
    Convert any supported model to ONNX IR.

    Args:
        src:    Path to source model file or directory.
        dst:    Output .onnx path. Defaults to <src_stem>.onnx in a temp dir.
        opset:  ONNX opset version (default 17, widely supported).

    Returns:
        Path to the generated .onnx file.

    Raises:
        ConversionError: If the format is unsupported or conversion fails.
        FileNotFoundError: If src does not exist.
    """
    src = Path(src)
    fmt = detect(src)

    if dst is None:
        tmp = Path(tempfile.mkdtemp(prefix="edgeforge_"))
        dst = tmp / (src.stem + ".onnx")
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)

    converters = {
        ModelFormat.ONNX:        _copy_onnx,
        ModelFormat.TFLITE:      _tflite_to_onnx,
        ModelFormat.TORCHSCRIPT: _torch_to_onnx,
        ModelFormat.TF_FROZEN:   _tf_frozen_to_onnx,
        ModelFormat.TF_SAVED:    _tf_saved_to_onnx,
    }

    if fmt == ModelFormat.UNKNOWN:
        raise ConversionError(
            f"Cannot determine format of '{src}'. "
            "Supported: .tflite, .onnx, .pt/.pth, .pb, SavedModel directory."
        )

    converter = converters[fmt]
    converter(src, dst, opset)

    if not dst.exists():
        raise ConversionError(f"Conversion produced no output at '{dst}'.")

    return dst


# ── Individual converters ────────────────────────────────────────────────────

def _copy_onnx(src: Path, dst: Path, opset: int) -> None:
    """Already ONNX — just copy and optionally simplify."""
    shutil.copy2(src, dst)
    # Attempt simplification — non-fatal if onnx-simplifier not installed
    try:
        import onnxsim
        import onnx
        model = onnx.load(str(dst))
        model_simplified, ok = onnxsim.simplify(model)
        if ok:
            onnx.save(model_simplified, str(dst))
    except ImportError:
        pass  # onnx-simplifier optional


def _tflite_to_onnx(src: Path, dst: Path, opset: int) -> None:
    """
    TFLite → ONNX via tensorflow and tf2onnx.
    tf2onnx is the most reliable path for TFLite Micro models.
    """
    try:
        import subprocess
        result = subprocess.run(
            [
                "python", "-m", "tf2onnx.convert",
                "--tflite", str(src),
                "--output", str(dst),
                "--opset", str(opset),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            raise ConversionError(
                f"tf2onnx failed converting '{src}':\n{result.stderr}"
            )
    except FileNotFoundError:
        raise ConversionError(
            "tf2onnx not found. Install with: pip install tf2onnx"
        )


def _torch_to_onnx(src: Path, dst: Path, opset: int) -> None:
    """
    TorchScript (.pt/.pth) → ONNX via torch.onnx.export.
    Requires the model to accept a dummy input — we infer shape from
    the model's first input if possible, otherwise prompt the user.
    """
    try:
        import torch
    except ImportError:
        raise ConversionError(
            "PyTorch not found. Install with: pip install torch"
        )

    try:
        model = torch.jit.load(str(src), map_location="cpu")
        model.eval()
    except Exception as e:
        raise ConversionError(f"Could not load TorchScript model '{src}': {e}")

    # Attempt to infer input shape from model graph
    dummy = _infer_torch_dummy(model)
    if dummy is None:
        raise ConversionError(
            f"Could not infer input shape for '{src}'. "
            "Use --input-shape to specify manually."
        )

    try:
        torch.onnx.export(
            model,
            dummy,
            str(dst),
            opset_version=opset,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        )
    except Exception as e:
        raise ConversionError(f"torch.onnx.export failed: {e}")


def _tf_frozen_to_onnx(src: Path, dst: Path, opset: int) -> None:
    """TF frozen graph (.pb) → ONNX via tf2onnx."""
    try:
        result = subprocess.run(
            [
                "python", "-m", "tf2onnx.convert",
                "--graphdef", str(src),
                "--output", str(dst),
                "--opset", str(opset),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            raise ConversionError(
                f"tf2onnx failed converting frozen graph '{src}':\n{result.stderr}"
            )
    except FileNotFoundError:
        raise ConversionError(
            "tf2onnx not found. Install with: pip install tf2onnx"
        )


def _tf_saved_to_onnx(src: Path, dst: Path, opset: int) -> None:
    """TF SavedModel directory → ONNX via tf2onnx."""
    try:
        result = subprocess.run(
            [
                "python", "-m", "tf2onnx.convert",
                "--saved-model", str(src),
                "--output", str(dst),
                "--opset", str(opset),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            raise ConversionError(
                f"tf2onnx failed converting SavedModel '{src}':\n{result.stderr}"
            )
    except FileNotFoundError:
        raise ConversionError(
            "tf2onnx not found. Install with: pip install tf2onnx"
        )


# ── Helpers ──────────────────────────────────────────────────────────────────

def _infer_torch_dummy(model) -> "torch.Tensor | None":
    """
    Try to infer a suitable dummy input tensor from TorchScript graph.
    Returns None if inference fails.
    """
    try:
        import torch
        graph = model.graph
        inputs = list(graph.inputs())
        # Skip 'self' input
        data_inputs = [i for i in inputs if i.debugName() != "self"]
        if not data_inputs:
            return None
        t = data_inputs[0].type()
        sizes = t.sizes()
        # Replace batch dim (0 or -1) with 1
        sizes = [1 if s <= 0 else s for s in sizes]
        return torch.zeros(*sizes)
    except Exception:
        return None
