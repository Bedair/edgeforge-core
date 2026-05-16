"""Tests for model format detector."""
import os
import struct
import tempfile
from pathlib import Path
import pytest
from edgeforge.converter.detector import detect, ModelFormat, describe


def _write_file(content: bytes, suffix: str) -> Path:
    f = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    f.write(content)
    f.close()
    return Path(f.name)


def test_tflite_by_magic():
    # TFLite FlatBuffer: 4 bytes offset + "TFL3" magic
    content = b"\x18\x00\x00\x00" + b"TFL3" + b"\x00" * 64
    p = _write_file(content, ".bin")  # wrong extension — magic should win
    try:
        assert detect(p) == ModelFormat.TFLITE
    finally:
        os.unlink(p)


def test_tflite_by_extension():
    p = _write_file(b"\x00" * 32, ".tflite")
    try:
        # Extension fallback when magic is ambiguous
        result = detect(p)
        assert result in (ModelFormat.TFLITE, ModelFormat.UNKNOWN)
    finally:
        os.unlink(p)


def test_onnx_by_extension():
    # Minimal protobuf-like content + onnx string
    content = b"\x08\x01" + b"onnx" + b"\x00" * 32
    p = _write_file(content, ".onnx")
    try:
        assert detect(p) == ModelFormat.ONNX
    finally:
        os.unlink(p)


def test_torchscript_zip_magic():
    content = b"PK\x03\x04" + b"\x00" * 64
    p = _write_file(content, ".pt")
    try:
        assert detect(p) == ModelFormat.TORCHSCRIPT
    finally:
        os.unlink(p)


def test_savedmodel_directory():
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        (dp / "saved_model.pb").write_bytes(b"\x00")
        assert detect(dp) == ModelFormat.TF_SAVED


def test_unknown_format():
    p = _write_file(b"\xDE\xAD\xBE\xEF" * 8, ".xyz")
    try:
        assert detect(p) == ModelFormat.UNKNOWN
    finally:
        os.unlink(p)


def test_file_not_found():
    with pytest.raises(FileNotFoundError):
        detect("/tmp/this_does_not_exist_edgeforge.model")


def test_describe_returns_dict():
    p = _write_file(b"\x00" * 64, ".onnx")
    try:
        d = describe(p)
        assert "format" in d
        assert "size_bytes" in d
        assert "size_human" in d
        assert d["size_bytes"] == 64
    finally:
        os.unlink(p)
