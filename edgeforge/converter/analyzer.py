"""
EdgeForge — Model Analyzer
Extracts operator list, tensor shapes, parameter count, and
RAM / flash footprint estimates from an ONNX model.

RAM estimate  = peak activation memory (largest intermediate tensor)
Flash estimate = weight storage (all parameters, quantised to INT8)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TensorInfo:
    name:  str
    shape: list[int]
    dtype: str
    bytes: int


@dataclass
class AnalysisResult:
    # Model identity
    model_path:      str
    format_detected: str
    onnx_opset:      int

    # Graph summary
    op_counts:       dict[str, int]      # e.g. {"Conv": 4, "Relu": 4}
    total_ops:       int
    parameter_count: int

    # Memory estimates (bytes)
    flash_bytes:     int                  # weights at INT8
    ram_bytes:       int                  # peak activation buffer
    arena_bytes:     int                  # TFLite Micro arena (RAM + overhead)

    # Tensor details
    input_tensors:   list[TensorInfo] = field(default_factory=list)
    output_tensors:  list[TensorInfo] = field(default_factory=list)

    # Convenience
    @property
    def flash_kb(self) -> float:
        return self.flash_bytes / 1024

    @property
    def ram_kb(self) -> float:
        return self.ram_bytes / 1024

    @property
    def arena_kb(self) -> float:
        return self.arena_bytes / 1024


# ONNX dtype → bytes per element
_DTYPE_BYTES: dict[int, int] = {
    1:  4,   # FLOAT
    2:  1,   # UINT8
    3:  1,   # INT8
    4:  2,   # UINT16
    5:  2,   # INT16
    6:  4,   # INT32
    7:  8,   # INT64
    10: 2,   # FLOAT16
    11: 8,   # DOUBLE
    12: 4,   # UINT32
}

_DTYPE_NAMES: dict[int, str] = {
    1: "float32", 2: "uint8", 3: "int8", 4: "uint16",
    5: "int16", 6: "int32", 7: "int64", 10: "float16",
    11: "float64", 12: "uint32",
}


def analyze(onnx_path: str | Path, original_format: str = "unknown") -> AnalysisResult:
    """
    Analyze an ONNX model and return memory and graph statistics.

    Args:
        onnx_path:       Path to the .onnx file.
        original_format: Original format before conversion (for display).

    Returns:
        AnalysisResult dataclass.

    Raises:
        ImportError: If onnx is not installed.
        ValueError:  If the model cannot be loaded or analyzed.
    """
    try:
        import onnx
        from onnx import numpy_helper, TensorProto
    except ImportError:
        raise ImportError("onnx not installed. Run: pip install onnx")

    p = Path(onnx_path)
    model = onnx.load(str(p))
    graph = model.graph

    # ── Opset ────────────────────────────────────────────────────────────────
    opset = model.opset_import[0].version if model.opset_import else 0

    # ── Operators ─────────────────────────────────────────────────────────────
    op_counts: dict[str, int] = {}
    for node in graph.node:
        op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1
    total_ops = sum(op_counts.values())

    # ── Parameters (initializers = weights + biases) ──────────────────────────
    parameter_count = 0
    flash_bytes = 0
    for init in graph.initializer:
        arr = numpy_helper.to_array(init)
        parameter_count += arr.size
        # Flash = INT8 storage (1 byte per param after quantisation)
        flash_bytes += arr.size

    # ── Activation memory (RAM) ───────────────────────────────────────────────
    # Build a shape map for all value_info tensors
    shape_map: dict[str, tuple[list[int], int]] = {}
    for vi in list(graph.value_info) + list(graph.input) + list(graph.output):
        t = vi.type.tensor_type
        if not t.HasField("elem_type"):
            continue
        shape = []
        for d in t.shape.dim:
            shape.append(max(d.dim_value, 1))  # treat dynamic dims as 1
        dtype_bytes = _DTYPE_BYTES.get(t.elem_type, 4)
        shape_map[vi.name] = (shape, dtype_bytes)

    # Peak RAM = largest single intermediate tensor
    max_activation = 0
    for name, (shape, dbytes) in shape_map.items():
        size = math.prod(shape) * dbytes if shape else 0
        if size > max_activation:
            max_activation = size

    ram_bytes = max_activation
    # TFLite Micro arena overhead: ~1.5x peak activation + 4KB bookkeeping
    arena_bytes = int(ram_bytes * 1.5) + 4096

    # ── Input / output tensors ────────────────────────────────────────────────
    def _parse_tensors(tensors) -> list[TensorInfo]:
        result = []
        for t in tensors:
            tt = t.type.tensor_type
            shape = [max(d.dim_value, 1) for d in tt.shape.dim]
            dtype_id = tt.elem_type
            dbytes = _DTYPE_BYTES.get(dtype_id, 4)
            total = math.prod(shape) * dbytes if shape else 0
            result.append(TensorInfo(
                name=t.name,
                shape=shape,
                dtype=_DTYPE_NAMES.get(dtype_id, f"dtype_{dtype_id}"),
                bytes=total,
            ))
        return result

    inputs  = _parse_tensors(graph.input)
    outputs = _parse_tensors(graph.output)

    return AnalysisResult(
        model_path=str(p),
        format_detected=original_format,
        onnx_opset=opset,
        op_counts=op_counts,
        total_ops=total_ops,
        parameter_count=parameter_count,
        flash_bytes=flash_bytes,
        ram_bytes=ram_bytes,
        arena_bytes=arena_bytes,
        input_tensors=inputs,
        output_tensors=outputs,
    )
