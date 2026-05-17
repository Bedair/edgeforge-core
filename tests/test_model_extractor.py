"""Tests for model_extractor.py"""
import tempfile
import numpy as np
import pytest
import onnx
from onnx import helper, TensorProto, numpy_helper
from pathlib import Path


def _make_test_model() -> Path:
    W = numpy_helper.from_array(np.random.randn(4,1,3,3).astype(np.float32), name="conv_W")
    b = numpy_helper.from_array(np.zeros(4).astype(np.float32), name="conv_b")
    conv = helper.make_node("Conv", ["X","conv_W","conv_b"], ["Y"], pads=[1,1,1,1])
    relu = helper.make_node("Relu", ["Y"], ["Z"])
    graph = helper.make_graph(
        [conv, relu], "test",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1,1,8,8])],
        [helper.make_tensor_value_info("Z", TensorProto.FLOAT, [1,4,8,8])],
        [W, b],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("",17)])
    model.ir_version = 8
    model = onnx.shape_inference.infer_shapes(model)
    tmp = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
    onnx.save(model, tmp.name)
    return Path(tmp.name)


def test_extract_returns_model_info():
    from edgeforge.codegen.model_extractor import extract
    p = _make_test_model()
    info = extract(p)
    assert info is not None
    assert info.model_name != ""
    assert info.node_count == 2


def test_extract_weights():
    from edgeforge.codegen.model_extractor import extract
    p = _make_test_model()
    info = extract(p)
    assert len(info.weights) >= 1
    for w in info.weights:
        assert w.c_name != ""
        assert w.size_bytes > 0
        assert w.numel > 0


def test_extract_input_output():
    from edgeforge.codegen.model_extractor import extract
    p = _make_test_model()
    info = extract(p)
    assert len(info.inputs)  >= 1
    assert len(info.outputs) >= 1
    assert info.input.shape  == [1, 1, 8, 8]
    assert info.output.shape == [1, 4, 8, 8]


def test_sanitise_c_name():
    from edgeforge.codegen.model_extractor import _sanitise_c_name
    assert _sanitise_c_name("hello/world") == "hello_world"
    assert _sanitise_c_name("123abc")      == "m_123abc"
    assert _sanitise_c_name("a::b::c")     == "a_b_c"
    assert _sanitise_c_name("valid_name")  == "valid_name"


def test_model_name_is_valid_c_identifier():
    from edgeforge.codegen.model_extractor import extract
    import re
    p = _make_test_model()
    info = extract(p)
    assert re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", info.model_name)


def test_weight_hex_data():
    from edgeforge.codegen.model_extractor import extract
    p = _make_test_model()
    info = extract(p)
    for w in info.weights:
        hex_data = w.flat_data_hex
        assert len(hex_data) == w.size_bytes
        assert all(h.startswith("0x") for h in hex_data[:5])
