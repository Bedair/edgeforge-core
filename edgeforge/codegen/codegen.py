"""
EdgeForge — Code Generation Orchestrator
Renders all Jinja2 templates and writes the output directory.

Entry point: generate(model_path, target, rtos, output_dir)
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from edgeforge.targets.loader import TargetProfile

from .model_extractor import extract, ModelInfo
from .arena_planner   import plan_arena, ArenaConfig

__version__ = "0.3.0-alpha"


@dataclass
class CodegenResult:
    output_dir:    str
    files_written: list[str]
    model_info:    ModelInfo
    arena_config:  ArenaConfig

    @property
    def success(self) -> bool:
        return len(self.files_written) > 0


class CodegenError(Exception):
    """Raised when code generation fails."""
    pass


# Map ONNX op_type → TFLite Micro resolver method
_OP_TO_TFLITE: dict[str, str] = {
    "Conv":                 "Conv2D",
    "ConvInteger":          "Conv2D",
    "QLinearConv":          "Conv2D",
    "DepthwiseConv":        "DepthwiseConv2D",
    "Gemm":                 "FullyConnected",
    "MatMul":               "FullyConnected",
    "MatMulInteger":        "FullyConnected",
    "Relu":                 "Relu",
    "Relu6":                "Relu6",
    "Softmax":              "Softmax",
    "Sigmoid":              "Logistic",
    "GlobalAveragePool":    "AveragePool2D",
    "AveragePool":          "AveragePool2D",
    "MaxPool":              "MaxPool2D",
    "Reshape":              "Reshape",
    "Transpose":            "Transpose",
    "BatchNormalization":   "None",           # fused by simplifier
    "Add":                  "Add",
    "Mul":                  "Mul",
    "Pad":                  "Pad",
    "Concat":               "Concatenation",
    "Split":                "Split",
    "Squeeze":              "Squeeze",
    "Flatten":              "Reshape",
    "QuantizeLinear":       "Quantize",
    "DequantizeLinear":     "Dequantize",
    "DynamicQuantizeLinear": "Quantize",
    "Cast":                 "Cast",
    "Clip":                 "Relu6",
    "Gather":               "Gather",
    "Slice":                "StridedSlice",
    "Pow":                  "None",
    "Sqrt":                 "None",
    "Log":                  "None",
}


def generate(
    model_path:  str | Path,
    target:      "TargetProfile",
    rtos:        str = "none",
    output_dir:  str | Path = "edgeforge_output",
) -> CodegenResult:
    """
    Generate C/C++ firmware integration files for the given model and target.

    Args:
        model_path:  Path to optimised .onnx model.
        target:      MCU target profile.
        rtos:        RTOS integration: "none", "freertos", or "zephyr".
        output_dir:  Directory to write generated files into.

    Returns:
        CodegenResult with file list and model metadata.

    Raises:
        CodegenError: If generation fails.
    """
    src = Path(model_path)
    if not src.exists():
        raise FileNotFoundError(f"Model not found: {src}")

    if rtos not in ("none", "freertos", "zephyr"):
        raise CodegenError(f"Unknown RTOS: {rtos}. Choose: none, freertos, zephyr")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Extract model info ────────────────────────────────────────────────────
    try:
        info = extract(src)
    except Exception as e:
        raise CodegenError(f"Model extraction failed: {e}") from e

    # ── Plan arena ────────────────────────────────────────────────────────────
    try:
        arena = plan_arena(src, target)
    except Exception as e:
        raise CodegenError(f"Arena planning failed: {e}") from e

    # ── Build template context ────────────────────────────────────────────────
    # Collect unique TFLite ops needed by this model
    import onnx as _onnx
    model_proto = _onnx.load(str(src))
    op_set: list[str] = []
    seen: set[str] = set()
    for node in model_proto.graph.node:
        tfl_op = _OP_TO_TFLITE.get(node.op_type, "None")
        if tfl_op != "None" and tfl_op not in seen:
            op_set.append(tfl_op)
            seen.add(tfl_op)

    ctx = {
        "info":    info,
        "target":  target,
        "arena":   arena,
        "rtos":    rtos,
        "ops":     op_set,
        "version": __version__,
    }

    # ── Render templates ──────────────────────────────────────────────────────
    env = _make_jinja_env()
    files_written: list[str] = []

    renders: list[tuple[str, str]] = [
        ("model_h.jinja2",           "model.h"),
        ("model_c.jinja2",           "model.c"),
        ("memory_config_h.jinja2",   "memory_config.h"),
        ("inference_runner_h.jinja2","inference_runner.h"),
        ("inference_runner_c.jinja2","inference_runner.c"),
        ("CMakeLists.jinja2",        "CMakeLists.txt"),
        ("README.jinja2",            "README.md"),
    ]

    for tmpl_name, out_name in renders:
        try:
            rendered = env.get_template(tmpl_name).render(**ctx)
            out_file = out / out_name
            out_file.write_text(rendered, encoding="utf-8")
            files_written.append(out_name)
        except Exception as e:
            raise CodegenError(f"Template {tmpl_name} failed: {e}") from e

    # RTOS glue — only if RTOS is not "none"
    if rtos != "none":
        try:
            rendered = env.get_template("rtos_glue_c.jinja2").render(**ctx)
            out_file = out / "rtos_glue.c"
            out_file.write_text(rendered, encoding="utf-8")
            files_written.append("rtos_glue.c")
        except Exception as e:
            raise CodegenError(f"RTOS glue template failed: {e}") from e

    return CodegenResult(
        output_dir=str(out),
        files_written=files_written,
        model_info=info,
        arena_config=arena,
    )


# ── Internal helpers ──────────────────────────────────────────────────────────

def _make_jinja_env():
    """Create a Jinja2 Environment pointing at the templates/ directory."""
    try:
        from jinja2 import Environment, FileSystemLoader, StrictUndefined
    except ImportError:
        raise CodegenError("jinja2 not installed. Run: pip install jinja2")

    # Templates directory is two levels up from this file: codegen/ → edgeforge/ → repo → templates/
    # Try multiple candidate paths to support both installed and dev layouts
    candidates = [
        Path(__file__).parent.parent.parent / "templates",   # dev: repo root/templates
        Path(__file__).parent.parent / "templates",           # installed package
        Path(__file__).parent / "templates",
    ]

    tmpl_dir = None
    for c in candidates:
        if c.is_dir() and (c / "model_h.jinja2").exists():
            tmpl_dir = c
            break

    if tmpl_dir is None:
        raise CodegenError(
            "Cannot locate templates/ directory. "
            "Ensure you are running from the edgeforge-core repo root."
        )

    return Environment(
        loader=FileSystemLoader(str(tmpl_dir)),
        undefined=StrictUndefined,
        keep_trailing_newline=True,
        trim_blocks=True,
        lstrip_blocks=True,
    )
