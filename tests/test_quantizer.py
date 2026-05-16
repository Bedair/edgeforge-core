"""Tests for the INT8 quantizer."""
import tempfile
import numpy as np
import pytest
import onnx
from pathlib import Path
from onnx import helper, TensorProto, numpy_helper


def _make_conv_onnx() -> Path:
    """Conv model — reliable for quantization tests (onnxruntime handles Conv well)."""
    W = numpy_helper.from_array(
        np.random.randn(4, 1, 3, 3).astype(np.float32), name="W"
    )
    conv = helper.make_node("Conv", ["X", "W"], ["Y"], pads=[1,1,1,1])
    graph = helper.make_graph(
        [conv], "conv_quant",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 8, 8])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4, 8, 8])],
        [W],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    # Run shape inference before saving
    model = onnx.shape_inference.infer_shapes(model)
    tmp = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
    onnx.save(model, tmp.name)
    return Path(tmp.name)


def test_quantize_dynamic_produces_output_file():
    from edgeforge.optimizer.quantizer import quantize_dynamic
    src = _make_conv_onnx()
    with tempfile.TemporaryDirectory() as d:
        dst = Path(d) / "quant.onnx"
        report = quantize_dynamic(src, dst)
        assert dst.exists(), "quantized file should be created"
        assert report.flash_before_kb > 0
        assert report.flash_after_kb  > 0
        assert report.mode == "dynamic"


def test_quantize_dynamic_report_fields():
    from edgeforge.optimizer.quantizer import quantize_dynamic
    src = _make_conv_onnx()
    with tempfile.TemporaryDirectory() as d:
        dst = Path(d) / "quant.onnx"
        report = quantize_dynamic(src, dst)
        assert isinstance(report.flash_reduction_pct, float)
        assert isinstance(report.nodes_quantized, int)
        assert isinstance(report.accuracy_delta_est, str)


def test_quantize_static_requires_calibration():
    from edgeforge.optimizer.quantizer import quantize_static
    src = _make_conv_onnx()
    with tempfile.TemporaryDirectory() as d:
        dst = Path(d) / "quant_static.onnx"
        with pytest.raises(ValueError, match="calibration"):
            quantize_static(src, dst)


def test_quantize_static_with_calibration_data():
    from edgeforge.optimizer.quantizer import quantize_static
    src = _make_conv_onnx()
    # Correct shape: [1, 1, 8, 8] matching model input
    calibration = [np.random.randn(1, 1, 8, 8).astype(np.float32) for _ in range(5)]
    with tempfile.TemporaryDirectory() as d:
        dst = Path(d) / "quant_static.onnx"
        report = quantize_static(src, dst, calibration_data=calibration)
        assert dst.exists()
        assert report.mode == "static"


def test_estimate_size_reduction():
    from edgeforge.optimizer.quantizer import quantize_dynamic, estimate_size_reduction
    src = _make_conv_onnx()
    with tempfile.TemporaryDirectory() as d:
        dst = Path(d) / "quant.onnx"
        quantize_dynamic(src, dst)
        stats = estimate_size_reduction(src, dst)
        assert "before_kb"     in stats
        assert "after_kb"      in stats
        assert "reduction_kb"  in stats
        assert "reduction_pct" in stats
        assert stats["before_kb"] > 0
