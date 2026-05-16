"""
EdgeForge - Model Format Detector
Identifies model format from magic bytes first, file extension as fallback.
Supports: TFLite, ONNX, TorchScript, TensorFlow SavedModel / frozen graph.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path


class ModelFormat(str, Enum):
    TFLITE       = "tflite"
    ONNX         = "onnx"
    TORCHSCRIPT  = "torchscript"
    TF_SAVED     = "tf_savedmodel"
    TF_FROZEN    = "tf_frozen"
    UNKNOWN      = "unknown"


# Magic byte signatures: (bytes_to_match, offset_in_file, format)
_MAGIC = [
    (b"\x18\x00\x00\x00", 0, ModelFormat.TFLITE),
    (b"ONNX",             0, ModelFormat.ONNX),
    (b"\x08\x01",         0, ModelFormat.ONNX),
    (b"PK\x03\x04",       0, ModelFormat.TORCHSCRIPT),
    (b"\x80\x02",         0, ModelFormat.TORCHSCRIPT),
]

_EXTENSION_MAP = {
    ".tflite": ModelFormat.TFLITE,
    ".onnx":   ModelFormat.ONNX,
    ".pt":     ModelFormat.TORCHSCRIPT,
    ".pth":    ModelFormat.TORCHSCRIPT,
    ".pb":     ModelFormat.TF_FROZEN,
}


def detect(path: "str | Path") -> ModelFormat:
    """
    Detect the model format of the given file.
    Strategy: magic bytes first, extension fallback, directory check for SavedModel.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Model path not found: {p}")

    if p.is_dir():
        if (p / "saved_model.pb").exists():
            return ModelFormat.TF_SAVED
        return ModelFormat.UNKNOWN

    try:
        with open(p, "rb") as f:
            header = f.read(16)
    except OSError:
        return ModelFormat.UNKNOWN

    if _is_tflite(header):
        return ModelFormat.TFLITE

    for sig, offset, fmt in _MAGIC:
        if header[offset: offset + len(sig)] == sig:
            if fmt == ModelFormat.ONNX and not _confirm_onnx(p):
                continue
            return fmt

    ext = p.suffix.lower()
    return _EXTENSION_MAP.get(ext, ModelFormat.UNKNOWN)


def _is_tflite(header: bytes) -> bool:
    """TFLite FlatBuffer: bytes 4-7 are 'TFL3', 'TFL2', or 'TFL1'."""
    if len(header) < 8:
        return False
    return header[4:8] in (b"TFL3", b"TFL2", b"TFL1")


def _confirm_onnx(path: Path) -> bool:
    """Quick check that a protobuf file is ONNX by scanning first 512 bytes."""
    try:
        with open(path, "rb") as f:
            chunk = f.read(512)
        return b"onnx" in chunk.lower() or b"ir_version" in chunk
    except OSError:
        return False


def describe(path: "str | Path") -> dict:
    """Return format, size, and path info as a dict."""
    p = Path(path)
    fmt = detect(p)
    size_bytes = p.stat().st_size if p.is_file() else 0
    return {
        "path":       str(p),
        "format":     fmt,
        "size_bytes": size_bytes,
        "size_human": _human_size(size_bytes),
    }


def _human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"
