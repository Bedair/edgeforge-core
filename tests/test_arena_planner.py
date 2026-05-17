"""Tests for arena_planner.py"""
import tempfile
import numpy as np
import pytest
import onnx
from onnx import helper, TensorProto, numpy_helper
from pathlib import Path


def _make_model() -> Path:
    W = numpy_helper.from_array(np.random.randn(4,1,3,3).astype(np.float32), name="W")
    conv = helper.make_node("Conv", ["X","W"], ["Y"], pads=[1,1,1,1])
    graph = helper.make_graph(
        [conv], "arena_test",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1,1,8,8])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1,4,8,8])],
        [W],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("",17)])
    model.ir_version = 8
    model = onnx.shape_inference.infer_shapes(model)
    tmp = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
    onnx.save(model, tmp.name)
    return Path(tmp.name)


def _make_target(ram_kb=192):
    from edgeforge.targets.loader import TargetProfile
    return TargetProfile(
        id="stm32f407", name="STM32F407", vendor="ST",
        core="cortex-m4f", fpu=True, npu=False,
        ram_kb=ram_kb, flash_kb=1024,
        arena_default_kb=64, cmsis_nn=True,
        runtime="tflite-micro", compiler_flags="-mcpu=cortex-m4",
        rtos_freertos=True, rtos_zephyr=True,
    )


def test_plan_arena_returns_config():
    from edgeforge.codegen.arena_planner import plan_arena
    p = _make_model()
    t = _make_target()
    config = plan_arena(p, t)
    assert config.total_bytes > 0
    assert config.total_kb > 0
    assert config.alignment in (8, 16)


def test_arena_is_aligned():
    from edgeforge.codegen.arena_planner import plan_arena
    p = _make_model()
    t = _make_target()
    config = plan_arena(p, t)
    assert config.total_bytes_aligned % config.alignment == 0


def test_arena_fits_in_ram():
    from edgeforge.codegen.arena_planner import plan_arena
    p = _make_model()
    t = _make_target(ram_kb=192)
    config = plan_arena(p, t)
    assert config.total_bytes <= t.ram_kb * 1024


def test_stm32_ccm_eligible():
    from edgeforge.codegen.arena_planner import plan_arena
    p = _make_model()
    t = _make_target()
    config = plan_arena(p, t)
    assert config.ccm_eligible is True
    assert config.ccm_size_kb == 64


def test_arena_has_headroom():
    from edgeforge.codegen.arena_planner import plan_arena
    p = _make_model()
    t = _make_target(ram_kb=192)
    config = plan_arena(p, t)
    assert config.ram_headroom_kb >= 0


def test_align_up():
    from edgeforge.codegen.arena_planner import _align_up
    assert _align_up(0,  8) == 0
    assert _align_up(1,  8) == 8
    assert _align_up(8,  8) == 8
    assert _align_up(9,  8) == 16
    assert _align_up(15, 8) == 16
    assert _align_up(16, 8) == 16
