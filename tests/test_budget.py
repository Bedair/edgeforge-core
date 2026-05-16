"""Tests for the MCU budget checker."""
import tempfile
import numpy as np
import pytest
from pathlib import Path


def _get_test_onnx() -> Path:
    """Minimal ONNX model for budget tests."""
    import onnx
    from onnx import helper, TensorProto, numpy_helper

    W = numpy_helper.from_array(
        np.random.randn(8, 4).astype(np.float32), name="W"
    )
    b = numpy_helper.from_array(np.zeros(8).astype(np.float32), name="b")
    graph = helper.make_graph(
        [helper.make_node("Gemm", ["X", "W", "b"], ["Y"])],
        "budget_test",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 8])],
        [W, b],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    tmp = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
    onnx.save(model, tmp.name)
    return Path(tmp.name)


def _make_target(ram_kb: int = 192, flash_kb: int = 1024):
    """Create a minimal TargetProfile for testing."""
    from edgeforge.targets.loader import TargetProfile
    return TargetProfile(
        id="test_target", name="Test MCU", vendor="Test",
        core="cortex-m4f", fpu=True, npu=False,
        ram_kb=ram_kb, flash_kb=flash_kb,
        arena_default_kb=64, cmsis_nn=True,
        runtime="tflite-micro", compiler_flags="",
        rtos_freertos=True, rtos_zephyr=True,
    )


def test_budget_fits_small_model():
    from edgeforge.optimizer.budget import check_budget
    src    = _get_test_onnx()
    target = _make_target(ram_kb=192, flash_kb=1024)
    report = check_budget(src, target)
    assert report.fits is True
    assert report.flash_kb > 0
    assert report.arena_kb > 0


def test_budget_report_fields():
    from edgeforge.optimizer.budget import check_budget
    src    = _get_test_onnx()
    target = _make_target()
    report = check_budget(src, target)
    assert report.target_id       == "test_target"
    assert report.target_ram_kb   == 192
    assert report.target_flash_kb == 1024
    assert isinstance(report.suggestions, list)
    assert isinstance(report.ram_used_pct, float)
    assert isinstance(report.flash_used_pct, float)


def test_budget_fails_tiny_target():
    from edgeforge.optimizer.budget import check_budget
    src    = _get_test_onnx()
    # Absurdly small target — 1 KB RAM, 2 KB Flash
    target = _make_target(ram_kb=1, flash_kb=2)
    report = check_budget(src, target)
    assert report.fits is False
    assert len(report.suggestions) > 0


def test_budget_suggestions_when_fails():
    from edgeforge.optimizer.budget import check_budget
    src    = _get_test_onnx()
    target = _make_target(ram_kb=1, flash_kb=2)
    report = check_budget(src, target)
    assert any("quantis" in s.lower() or "flash" in s.lower() or "ram" in s.lower()
               for s in report.suggestions)


def test_format_bar():
    from edgeforge.optimizer.budget import format_bar
    bar_0   = format_bar(0)
    bar_50  = format_bar(50)
    bar_100 = format_bar(100)
    assert "." in bar_0          # all empty
    assert "#" in bar_50         # half full
    assert "." not in bar_100    # all full
    assert len(bar_50) == 20     # default width
    # Over 100% should be capped -- no overflow
    bar_over = format_bar(200)
    assert len(bar_over) == 20
    assert "." not in bar_over   # fully filled at cap


def test_suggest_strategy_no_warnings_when_fits():
    from edgeforge.optimizer.budget import suggest_strategy
    target = _make_target(ram_kb=192, flash_kb=1024)
    tips = suggest_strategy(
        flash_kb=10, arena_kb=10, target=target,
        ram_fits=True, flash_fits=True, ram_used_pct=5.0,
    )
    assert len(tips) == 0
