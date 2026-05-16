"""EdgeForge converter — model format detection and ONNX IR conversion."""
from .detector import detect, describe, ModelFormat
from .to_onnx  import to_onnx, ConversionError
from .analyzer import analyze
