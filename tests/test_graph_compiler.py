"""Tests for the CMSIS-NN graph compiler."""
import tempfile
import numpy as np
import pytest
import onnx
from onnx import helper, TensorProto, numpy_helper
from pathlib import Path


def _make_conv_relu_model() -> Path:
    W = numpy_helper.from_array(
        np.random.randn(8, 1, 3, 3).astype(np.float32) * 0.1, name="W"
    )
    b = numpy_helper.from_array(np.zeros(8).astype(np.float32), name="b")
    conv = helper.make_node("Conv", ["X","W","b"], ["C"], pads=[1,1,1,1])
    relu = helper.make_node("Relu", ["C"], ["R"])
    gap  = helper.make_node("GlobalAveragePool", ["R"], ["P"])
    sh   = numpy_helper.from_array(np.array([1,-1], dtype=np.int64), name="sh")
    resh = helper.make_node("Reshape", ["P","sh"], ["F"])
    fcW  = numpy_helper.from_array(
        np.random.randn(4, 8).astype(np.float32) * 0.1, name="fcW"
    )
    fcB  = numpy_helper.from_array(np.zeros(4).astype(np.float32), name="fcB")
    gemm = helper.make_node("Gemm", ["F","fcW","fcB"], ["out"], transB=1)
    sm   = helper.make_node("Softmax", ["out"], ["prob"], axis=1)
    graph = helper.make_graph(
        [conv, relu, gap, resh, gemm, sm], "test_graph",
        [helper.make_tensor_value_info("X",    TensorProto.FLOAT, [1,1,8,8])],
        [helper.make_tensor_value_info("prob", TensorProto.FLOAT, [1,4])],
        [W, b, sh, fcW, fcB],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("",17)])
    model.ir_version = 8
    model = onnx.shape_inference.infer_shapes(model)
    tmp = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
    onnx.save(model, tmp.name)
    return Path(tmp.name)


def test_compile_graph_returns_plan():
    from edgeforge.codegen.graph_compiler import compile_graph
    p    = _make_conv_relu_model()
    plan = compile_graph(p)
    assert plan is not None
    assert plan.model_name != ""
    assert len(plan.layers) > 0


def test_compile_graph_finds_conv_layer():
    from edgeforge.codegen.graph_compiler import compile_graph
    p    = _make_conv_relu_model()
    plan = compile_graph(p)
    fn_names = [l.fn_name for l in plan.layers]
    assert "arm_convolve_s8" in fn_names


def test_compile_graph_finds_fc_layer():
    from edgeforge.codegen.graph_compiler import compile_graph
    p    = _make_conv_relu_model()
    plan = compile_graph(p)
    fn_names = [l.fn_name for l in plan.layers]
    assert "arm_fully_connected_s8" in fn_names


def test_compile_graph_scratch_bytes_positive():
    from edgeforge.codegen.graph_compiler import compile_graph
    p    = _make_conv_relu_model()
    plan = compile_graph(p)
    assert plan.scratch_bytes > 0


def test_emit_inference_runner_c_no_rtos():
    from edgeforge.codegen.graph_compiler import compile_graph, emit_inference_runner_c
    p    = _make_conv_relu_model()
    plan = compile_graph(p)
    code = emit_inference_runner_c(plan, target_id="stm32f407", rtos="none")
    assert "arm_nnfunctions.h" in code
    assert "edgeforge_init"    in code
    assert "edgeforge_infer"   in code
    assert "edgeforge_deinit"  in code
    assert "TensorFlow"        not in code   # NO TFLite Micro
    assert "MicroInterpreter"  not in code   # NO TFLite Micro


def test_emit_no_tflite_micro_dependency():
    from edgeforge.codegen.graph_compiler import compile_graph, emit_inference_runner_c
    p    = _make_conv_relu_model()
    plan = compile_graph(p)
    code = emit_inference_runner_c(plan, target_id="stm32f407", rtos="none")
    # Must NOT reference TFLite Micro
    assert "tensorflow" not in code.lower()
    assert "tflite"     not in code.lower()
    # MUST reference CMSIS-NN
    assert "arm_nn" in code.lower() or "cmsis_nn" in code.lower()


def test_emit_freertos_glue():
    from edgeforge.codegen.graph_compiler import compile_graph, emit_inference_runner_c
    p    = _make_conv_relu_model()
    plan = compile_graph(p)
    code = emit_inference_runner_c(plan, target_id="stm32f407", rtos="freertos")
    assert "FreeRTOS.h"       in code
    assert "xSemaphoreTake"   in code
    assert "xSemaphoreGive"   in code


def test_emit_zephyr_glue():
    from edgeforge.codegen.graph_compiler import compile_graph, emit_inference_runner_c
    p    = _make_conv_relu_model()
    plan = compile_graph(p)
    code = emit_inference_runner_c(plan, target_id="nrf52840", rtos="zephyr")
    assert "zephyr/kernel.h" in code
    assert "k_mutex"          in code


def test_emit_contains_layer_calls():
    from edgeforge.codegen.graph_compiler import compile_graph, emit_inference_runner_c
    p    = _make_conv_relu_model()
    plan = compile_graph(p)
    code = emit_inference_runner_c(plan, target_id="stm32f407", rtos="none")
    assert "arm_convolve_s8"          in code
    assert "arm_fully_connected_s8"   in code
    assert "arm_relu_s8"              in code


def test_emit_contains_scratch_buffer():
    from edgeforge.codegen.graph_compiler import compile_graph, emit_inference_runner_c
    p    = _make_conv_relu_model()
    plan = compile_graph(p)
    code = emit_inference_runner_c(plan, target_id="stm32f407", rtos="none")
    assert "_scratch" in code
    assert "cmsis_nn_context" in code


def test_generate_uses_graph_compiler():
    """Full integration: generate() should produce inference_runner.c using CMSIS-NN."""
    from edgeforge.codegen.codegen import generate
    from edgeforge.targets.loader  import TargetProfile
    p = _make_conv_relu_model()
    t = TargetProfile(
        id="stm32f407", name="STM32F407", vendor="ST",
        core="cortex-m4f", fpu=True, npu=False,
        ram_kb=192, flash_kb=1024,
        arena_default_kb=64, cmsis_nn=True,
        runtime="tflite-micro", compiler_flags="-mcpu=cortex-m4",
        rtos_freertos=True, rtos_zephyr=True,
    )
    with tempfile.TemporaryDirectory() as d:
        result = generate(p, t, rtos="none", output_dir=d)
        runner_c = (Path(d) / "inference_runner.c").read_text(encoding="utf-8")
        assert "arm_nnfunctions.h"   in runner_c
        assert "arm_convolve_s8"     in runner_c
        assert "MicroInterpreter"    not in runner_c
        assert "tensorflow"          not in runner_c.lower()
