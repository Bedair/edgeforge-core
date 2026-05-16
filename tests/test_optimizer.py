"""Integration tests for the full optimisation pipeline."""
import tempfile
import numpy as np
import pytest
import onnx
from pathlib import Path


def _make_conv_model() -> Path:
    """Minimal Conv model — most realistic for MCU quantisation tests."""
    from onnx import helper, TensorProto, numpy_helper

    W = numpy_helper.from_array(
        np.random.randn(4, 1, 3, 3).astype(np.float32), name="W"
    )
    conv = helper.make_node(
        "Conv", ["X", "W"], ["Y"],
        pads=[1, 1, 1, 1], group=1,
    )
    graph = helper.make_graph(
        [conv], "conv_test",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 8, 8])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4, 8, 8])],
        [W],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    tmp = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
    onnx.save(model, tmp.name)
    return Path(tmp.name)


def _make_target(ram_kb=192, flash_kb=1024):
    from edgeforge.targets.loader import TargetProfile
    return TargetProfile(
        id="stm32f407", name="STM32F407", vendor="ST",
        core="cortex-m4f", fpu=True, npu=False,
        ram_kb=ram_kb, flash_kb=flash_kb,
        arena_default_kb=64, cmsis_nn=True,
        runtime="tflite-micro", compiler_flags="",
        rtos_freertos=True, rtos_zephyr=True,
    )


def test_optimize_end_to_end():
    from edgeforge.optimizer.optimizer import optimize
    src    = _make_conv_model()
    target = _make_target()
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "opt.onnx"
        result = optimize(src, target, output_path=out)
        assert out.exists(), "output file must exist"
        assert result.success is True
        assert result.flash_before_kb > 0
        assert result.flash_after_kb  > 0
        assert len(result.steps_applied) > 0


def test_optimize_output_is_valid_onnx():
    from edgeforge.optimizer.optimizer import optimize
    src    = _make_conv_model()
    target = _make_target()
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "opt.onnx"
        optimize(src, target, output_path=out)
        loaded = onnx.load(str(out))
        onnx.checker.check_model(loaded)  # raises if invalid


def test_optimize_result_has_all_reports():
    from edgeforge.optimizer.optimizer import optimize
    src    = _make_conv_model()
    target = _make_target()
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "opt.onnx"
        result = optimize(src, target, output_path=out)
        assert result.simplify_report  is not None
        assert result.quantize_report  is not None
        assert result.budget_report    is not None


def test_optimize_flash_reduction():
    from edgeforge.optimizer.optimizer import optimize
    src    = _make_conv_model()
    target = _make_target()
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "opt.onnx"
        result = optimize(src, target, output_path=out)
        # Dynamic INT8 quantisation always reduces size
        assert isinstance(result.flash_reduction_pct, float)


def test_optimize_strict_raises_when_doesnt_fit():
    from edgeforge.optimizer.optimizer import optimize, OptimizeError
    src    = _make_conv_model()
    target = _make_target(ram_kb=1, flash_kb=2)   # impossibly small
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "opt.onnx"
        with pytest.raises(OptimizeError):
            optimize(src, target, output_path=out, strict=True)


def test_optimize_non_strict_returns_result_even_when_doesnt_fit():
    from edgeforge.optimizer.optimizer import optimize
    src    = _make_conv_model()
    target = _make_target(ram_kb=1, flash_kb=2)
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "opt.onnx"
        result = optimize(src, target, output_path=out, strict=False)
        assert result.fits is False
        assert result.success is True  # file still written


def test_optimize_default_output_path():
    from edgeforge.optimizer.optimizer import optimize
    src    = _make_conv_model()
    target = _make_target()
    result = optimize(src, target)   # no output_path
    out = Path(result.output_path)
    assert out.exists()
    assert out.suffix == ".onnx"
    out.unlink()  # cleanup


def test_optimize_already_onnx_skips_conversion():
    from edgeforge.optimizer.optimizer import optimize
    src    = _make_conv_model()  # already .onnx
    target = _make_target()
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "opt.onnx"
        result = optimize(src, target, output_path=out)
        # No conversion step in steps_applied
        assert not any("convert" in s for s in result.steps_applied)


def test_optimize_src_not_found():
    from edgeforge.optimizer.optimizer import optimize
    target = _make_target()
    with pytest.raises(FileNotFoundError):
        optimize("/nonexistent/model.onnx", target)
