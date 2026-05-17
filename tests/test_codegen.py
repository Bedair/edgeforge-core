"""Integration tests for codegen.py — full template rendering."""
import tempfile
import shutil
import numpy as np
import pytest
import onnx
from onnx import helper, TensorProto, numpy_helper
from pathlib import Path


def _make_model() -> Path:
    W = numpy_helper.from_array(np.random.randn(4,1,3,3).astype(np.float32), name="conv_W")
    b = numpy_helper.from_array(np.zeros(4).astype(np.float32), name="conv_b")
    conv = helper.make_node("Conv", ["X","conv_W","conv_b"], ["Y"], pads=[1,1,1,1])
    relu = helper.make_node("Relu", ["Y"], ["Z"])
    gap  = helper.make_node("GlobalAveragePool", ["Z"], ["P"])
    shape = numpy_helper.from_array(np.array([1,-1], dtype=np.int64), name="shape")
    resh  = helper.make_node("Reshape", ["P","shape"], ["F"])
    fc_W  = numpy_helper.from_array(np.random.randn(10,4).astype(np.float32), name="fc_W")
    fc_b  = numpy_helper.from_array(np.zeros(10).astype(np.float32), name="fc_b")
    gemm  = helper.make_node("Gemm", ["F","fc_W","fc_b"], ["out"], transB=1)
    graph = helper.make_graph(
        [conv, relu, gap, resh, gemm], "codegen_test",
        [helper.make_tensor_value_info("X",   TensorProto.FLOAT, [1,1,8,8])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1,10])],
        [W, b, shape, fc_W, fc_b],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("",17)])
    model.ir_version = 8
    model = onnx.shape_inference.infer_shapes(model)
    tmp = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
    onnx.save(model, tmp.name)
    return Path(tmp.name)


def _make_target():
    from edgeforge.targets.loader import TargetProfile
    return TargetProfile(
        id="stm32f407", name="STM32F407", vendor="ST",
        core="cortex-m4f", fpu=True, npu=False,
        ram_kb=192, flash_kb=1024,
        arena_default_kb=64, cmsis_nn=True,
        runtime="tflite-micro", compiler_flags="-mcpu=cortex-m4",
        rtos_freertos=True, rtos_zephyr=True,
    )


def test_generate_returns_result():
    from edgeforge.codegen.codegen import generate
    src = _make_model()
    t   = _make_target()
    with tempfile.TemporaryDirectory() as d:
        result = generate(src, t, rtos="none", output_dir=d)
        assert result.success is True
        assert len(result.files_written) > 0


def test_generate_creates_all_files_no_rtos():
    from edgeforge.codegen.codegen import generate
    src = _make_model(); t = _make_target()
    with tempfile.TemporaryDirectory() as d:
        result = generate(src, t, rtos="none", output_dir=d)
        out = Path(d)
        for fname in ["model.h","model.c","memory_config.h",
                      "inference_runner.h","inference_runner.c",
                      "CMakeLists.txt","README.md"]:
            assert (out / fname).exists(), f"Missing: {fname}"
        # rtos_glue should NOT be present
        assert not (out / "rtos_glue.c").exists()


def test_generate_creates_rtos_glue_freertos():
    from edgeforge.codegen.codegen import generate
    src = _make_model(); t = _make_target()
    with tempfile.TemporaryDirectory() as d:
        result = generate(src, t, rtos="freertos", output_dir=d)
        assert (Path(d) / "rtos_glue.c").exists()
        assert "rtos_glue.c" in result.files_written


def test_generate_creates_rtos_glue_zephyr():
    from edgeforge.codegen.codegen import generate
    src = _make_model(); t = _make_target()
    with tempfile.TemporaryDirectory() as d:
        result = generate(src, t, rtos="zephyr", output_dir=d)
        assert (Path(d) / "rtos_glue.c").exists()


def test_model_h_contains_arena_define():
    from edgeforge.codegen.codegen import generate
    src = _make_model(); t = _make_target()
    with tempfile.TemporaryDirectory() as d:
        generate(src, t, rtos="none", output_dir=d)
        content = (Path(d) / "memory_config.h").read_text()
        assert "EDGEFORGE_ARENA_SIZE" in content
        assert "EDGEFORGE_ARENA_ALIGNMENT" in content


def test_model_h_contains_input_output_defines():
    from edgeforge.codegen.codegen import generate
    src = _make_model(); t = _make_target()
    with tempfile.TemporaryDirectory() as d:
        generate(src, t, rtos="none", output_dir=d)
        content = (Path(d) / "model.h").read_text()
        assert "EDGEFORGE_INPUT_SIZE"  in content
        assert "EDGEFORGE_OUTPUT_SIZE" in content
        assert "EDGEFORGE_IS_QUANTIZED" in content


def test_model_c_contains_weight_arrays():
    from edgeforge.codegen.codegen import generate
    src = _make_model(); t = _make_target()
    with tempfile.TemporaryDirectory() as d:
        generate(src, t, rtos="none", output_dir=d)
        content = (Path(d) / "model.c").read_text()
        assert "#include" in content
        assert "0x" in content           # hex weight data
        assert "const" in content


def test_inference_runner_h_has_api():
    from edgeforge.codegen.codegen import generate
    src = _make_model(); t = _make_target()
    with tempfile.TemporaryDirectory() as d:
        generate(src, t, rtos="none", output_dir=d)
        content = (Path(d) / "inference_runner.h").read_text()
        assert "edgeforge_init"   in content
        assert "edgeforge_infer"  in content
        assert "edgeforge_deinit" in content
        assert "edgeforge_status_t" in content


def test_cmake_contains_target_name():
    from edgeforge.codegen.codegen import generate
    src = _make_model(); t = _make_target()
    with tempfile.TemporaryDirectory() as d:
        generate(src, t, rtos="none", output_dir=d)
        content = (Path(d) / "CMakeLists.txt").read_text()
        assert "edgeforge_" in content
        assert "model.c" in content


def test_readme_contains_integration_guide():
    from edgeforge.codegen.codegen import generate
    src = _make_model(); t = _make_target()
    with tempfile.TemporaryDirectory() as d:
        generate(src, t, rtos="none", output_dir=d)
        content = (Path(d) / "README.md").read_text(encoding="utf-8")
        assert "edgeforge_init"   in content
        assert "edgeforge_infer"  in content
        assert "arena" in content.lower()
        assert "STM32F407"        in content


def test_generate_raises_on_missing_model():
    from edgeforge.codegen.codegen import generate
    t = _make_target()
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(FileNotFoundError):
            generate("/nonexistent/model.onnx", t, output_dir=d)


def test_generate_raises_on_invalid_rtos():
    from edgeforge.codegen.codegen import generate, CodegenError
    src = _make_model(); t = _make_target()
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(CodegenError, match="Unknown RTOS"):
            generate(src, t, rtos="threadx", output_dir=d)


def test_freertos_glue_contains_semaphore():
    from edgeforge.codegen.codegen import generate
    src = _make_model(); t = _make_target()
    with tempfile.TemporaryDirectory() as d:
        generate(src, t, rtos="freertos", output_dir=d)
        content = (Path(d) / "rtos_glue.c").read_text()
        assert "xSemaphore" in content or "SemaphoreHandle" in content
        assert "edgeforge_rtos_start" in content


def test_zephyr_glue_contains_k_mutex():
    from edgeforge.codegen.codegen import generate
    src = _make_model(); t = _make_target()
    with tempfile.TemporaryDirectory() as d:
        generate(src, t, rtos="zephyr", output_dir=d)
        content = (Path(d) / "rtos_glue.c").read_text()
        assert "k_mutex" in content or "k_msgq" in content
        assert "edgeforge_rtos_start" in content
