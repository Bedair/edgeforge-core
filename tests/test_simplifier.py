"""Tests for the ONNX graph simplifier."""
import pytest
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper


def _make_simple_model(n_extra_nodes: int = 0) -> onnx.ModelProto:
    """Build a minimal valid ONNX model for testing."""
    # Input → MatMul → output
    W = numpy_helper.from_array(
        np.random.randn(4, 4).astype(np.float32), name="W"
    )
    matmul = helper.make_node("MatMul", ["X", "W"], ["Y"])
    nodes  = [matmul]

    # Add dead nodes (outputs never consumed)
    for i in range(n_extra_nodes):
        dead = helper.make_node(
            "Relu", ["Y"], [f"dead_out_{i}"],
            name=f"dead_{i}"
        )
        nodes.append(dead)

    graph = helper.make_graph(
        nodes, "test",
        [helper.make_tensor_value_info("X",   TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("Y",   TensorProto.FLOAT, [1, 4])],
        [W],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model


def test_simplify_returns_model_and_report():
    from edgeforge.optimizer.simplifier import simplify
    model  = _make_simple_model()
    result, report = simplify(model)
    assert isinstance(result, onnx.ModelProto)
    assert report.nodes_before >= 1


def test_simplify_reduces_dead_nodes():
    from edgeforge.optimizer.simplifier import simplify, eliminate_dead_nodes
    model = _make_simple_model(n_extra_nodes=3)
    assert len(model.graph.node) == 4  # 1 MatMul + 3 dead Relu

    cleaned, removed = eliminate_dead_nodes(model)
    assert removed == 3
    assert len(cleaned.graph.node) == 1


def test_eliminate_dead_nodes_no_op_on_clean_model():
    from edgeforge.optimizer.simplifier import eliminate_dead_nodes
    model = _make_simple_model(n_extra_nodes=0)
    cleaned, removed = eliminate_dead_nodes(model)
    assert removed == 0
    assert len(cleaned.graph.node) == len(model.graph.node)


def test_simplify_nodes_before_after_consistent():
    from edgeforge.optimizer.simplifier import simplify
    model  = _make_simple_model(n_extra_nodes=2)
    result, report = simplify(model)
    assert report.nodes_before >= report.nodes_after
    assert len(result.graph.node) == report.nodes_after


def test_fold_constants_does_not_crash():
    from edgeforge.optimizer.simplifier import fold_constants
    model = _make_simple_model()
    result, folded = fold_constants(model)
    assert isinstance(result, onnx.ModelProto)
    assert folded >= 0


def test_simplify_report_passes_applied():
    from edgeforge.optimizer.simplifier import simplify
    model = _make_simple_model(n_extra_nodes=2)
    _, report = simplify(model)
    # Should have at least recorded that dead node elimination ran
    assert isinstance(report.passes_applied, list)
