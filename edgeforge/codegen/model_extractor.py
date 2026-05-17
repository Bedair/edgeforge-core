"""
EdgeForge — Model Extractor
Pulls weights, tensor metadata, and quantisation parameters from an
optimised ONNX model for use by the code generation templates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import numpy_helper, TensorProto


# ONNX elem_type → C type string
_DTYPE_TO_C: dict[int, str] = {
    1:  "float",
    2:  "uint8_t",
    3:  "int8_t",
    4:  "uint16_t",
    5:  "int16_t",
    6:  "int32_t",
    7:  "int64_t",
    10: "uint16_t",   # float16 — stored as uint16
    11: "double",
    12: "uint32_t",
}

_DTYPE_BYTES: dict[int, int] = {
    1: 4, 2: 1, 3: 1, 4: 2, 5: 2,
    6: 4, 7: 8, 10: 2, 11: 8, 12: 4,
}

_DTYPE_NAMES: dict[int, str] = {
    1: "float32", 2: "uint8", 3: "int8", 4: "uint16",
    5: "int16",   6: "int32", 7: "int64", 10: "float16",
    11: "float64", 12: "uint32",
}


@dataclass
class WeightTensor:
    name:       str           # original ONNX name
    c_name:     str           # sanitised C identifier
    shape:      list[int]
    dtype:      str           # numpy dtype string
    c_type:     str           # C type (int8_t, float, etc.)
    data:       np.ndarray    # raw numpy array
    size_bytes: int

    @property
    def numel(self) -> int:
        return int(np.prod(self.shape)) if self.shape else 1

    @property
    def shape_str(self) -> str:
        return ", ".join(str(d) for d in self.shape)

    @property
    def flat_data_hex(self) -> list[str]:
        """Return data as list of hex strings for C array initialiser."""
        raw = self.data.flatten().tobytes()
        return [f"0x{b:02x}" for b in raw]


@dataclass
class IOTensor:
    name:       str
    c_name:     str
    shape:      list[int]
    dtype_id:   int
    dtype:      str
    c_type:     str
    size_bytes: int
    # Quantisation params (if INT8)
    scale:      float = 1.0
    zero_point: int   = 0
    is_quantized: bool = False

    @property
    def numel(self) -> int:
        return int(np.prod(self.shape)) if self.shape else 1

    @property
    def shape_str(self) -> str:
        return ", ".join(str(d) for d in self.shape)


@dataclass
class ModelInfo:
    # Identity
    model_path:   str
    model_name:   str           # sanitised C identifier for the model

    # Graph
    node_count:   int
    op_summary:   str           # e.g. "Conv x5, Relu x5, ..."
    is_quantized: bool

    # Weights
    weights:      list[WeightTensor] = field(default_factory=list)
    total_weight_bytes: int = 0

    # I/O
    inputs:       list[IOTensor] = field(default_factory=list)
    outputs:      list[IOTensor] = field(default_factory=list)

    # Convenience
    @property
    def input(self) -> IOTensor:
        return self.inputs[0]

    @property
    def output(self) -> IOTensor:
        return self.outputs[0]


def extract(model_path: str | Path) -> ModelInfo:
    """
    Extract all information needed for code generation from an ONNX model.

    Args:
        model_path: Path to optimised .onnx file.

    Returns:
        ModelInfo dataclass with weights, tensors, and metadata.
    """
    p = Path(model_path)
    model = onnx.load(str(p))
    graph = model.graph

    # ── Model identity ────────────────────────────────────────────────────────
    raw_name  = p.stem
    model_name = _sanitise_c_name(raw_name)

    # ── Graph metadata ────────────────────────────────────────────────────────
    op_counts: dict[str, int] = {}
    for node in graph.node:
        op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1
    op_summary = ", ".join(
        f"{op} x{n}" for op, n in
        sorted(op_counts.items(), key=lambda x: -x[1])
    )

    is_quantized = any(
        n.op_type in {"DequantizeLinear", "QuantizeLinear",
                      "QLinearConv", "QLinearMatMul",
                      "ConvInteger", "MatMulInteger"}
        for n in graph.node
    )

    # ── Extract weights ───────────────────────────────────────────────────────
    weights: list[WeightTensor] = []
    total_bytes = 0

    for init in graph.initializer:
        try:
            arr = numpy_helper.to_array(init)
        except Exception:
            continue

        # Skip scalar and very small constants (shape tensors etc.)
        if arr.size < 2:
            continue

        dtype_id  = init.data_type
        c_type    = _DTYPE_TO_C.get(dtype_id, "uint8_t")
        size_bytes = arr.nbytes

        wt = WeightTensor(
            name=init.name,
            c_name=_sanitise_c_name(init.name),
            shape=list(arr.shape),
            dtype=str(arr.dtype),
            c_type=c_type,
            data=arr,
            size_bytes=size_bytes,
        )
        weights.append(wt)
        total_bytes += size_bytes

    # ── Extract I/O tensors ───────────────────────────────────────────────────
    def _parse_io(tensors) -> list[IOTensor]:
        result = []
        for t in tensors:
            tt    = t.type.tensor_type
            dtype = tt.elem_type
            shape = [max(d.dim_value, 1) for d in tt.shape.dim]
            dbytes = _DTYPE_BYTES.get(dtype, 4)
            size  = math.prod(shape) * dbytes if shape else dbytes

            # Quantisation params from graph metadata (if present)
            scale, zp, is_q = _extract_quant_params(graph, t.name, dtype)

            result.append(IOTensor(
                name=t.name,
                c_name=_sanitise_c_name(t.name),
                shape=shape,
                dtype_id=dtype,
                dtype=_DTYPE_NAMES.get(dtype, "uint8"),
                c_type=_DTYPE_TO_C.get(dtype, "float"),
                size_bytes=size,
                scale=scale,
                zero_point=zp,
                is_quantized=is_q,
            ))
        return result

    inputs  = _parse_io(graph.input)
    outputs = _parse_io(graph.output)

    return ModelInfo(
        model_path=str(p),
        model_name=model_name,
        node_count=len(graph.node),
        op_summary=op_summary,
        is_quantized=is_quantized,
        weights=weights,
        total_weight_bytes=total_bytes,
        inputs=inputs,
        outputs=outputs,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sanitise_c_name(name: str) -> str:
    """Convert any string to a valid C identifier."""
    import re
    # Replace non-alphanumeric with underscore
    s = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    # Must not start with digit
    if s and s[0].isdigit():
        s = "m_" + s
    # Collapse multiple underscores
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "tensor"


def _extract_quant_params(
    graph: onnx.GraphProto,
    tensor_name: str,
    dtype_id: int,
) -> tuple[float, int, bool]:
    """
    Try to extract quantisation scale and zero_point for a tensor.
    Returns (scale, zero_point, is_quantized).
    """
    # INT8 / UINT8 tensors in a quantised model
    if dtype_id not in (2, 3):
        return 1.0, 0, False

    # Look for QuantizeLinear node that produces this tensor
    for node in graph.node:
        if node.op_type == "QuantizeLinear" and tensor_name in node.output:
            # inputs: [x, y_scale, y_zero_point]
            try:
                from onnx import numpy_helper as nh
                inits = {i.name: nh.to_array(i) for i in graph.initializer}
                scale = float(inits.get(node.input[1], np.array([1.0]))[0])
                zp    = int(inits.get(node.input[2], np.array([0]))[0])
                return scale, zp, True
            except Exception:
                pass

    return 1.0, 0, dtype_id in (2, 3)
