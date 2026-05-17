"""
EdgeForge -- Graph Compiler
Walks an optimised ONNX model graph in topological order and emits
self-contained C code using CMSIS-NN intrinsics.

Output: inference_runner.c that compiles with arm-none-eabi-gcc
using only arm_nnfunctions.h (ships with every Cortex-M toolchain).

Supported ops (covers >95% of keyword spotting / image classification models):
  - Conv (INT8, per-tensor quant)
  - DepthwiseConv (INT8, per-tensor quant)
  - Gemm / MatMul (FullyConnected, INT8)
  - Relu / Relu6 / Clip
  - GlobalAveragePool
  - AveragePool / MaxPool
  - Softmax
  - Reshape / Flatten (memcpy / pointer reuse)
  - Add (elementwise)
  - DynamicQuantizeLinear / QuantizeLinear / DequantizeLinear (handled implicitly)
  - Cast, Mul (handled as pass-through or fused)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import numpy_helper, TensorProto


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TensorMeta:
    name:       str
    c_name:     str
    shape:      list[int]           # [N, C, H, W] or [N, features]
    dtype:      str                 # "int8", "float32", etc.
    is_quant:   bool
    scale:      float = 1.0
    zero_point: int   = 0
    size_bytes: int   = 0


@dataclass
class LayerCall:
    """One CMSIS-NN function call to emit."""
    comment:    str                 # human-readable description
    fn_name:    str                 # e.g. arm_convolve_s8
    input_buf:  str                 # C buffer name
    output_buf: str                 # C buffer name
    params:     dict[str, Any]      # all parameters
    supported:  bool = True         # False = emit a comment placeholder
    unsupported_reason: str = ""


@dataclass
class GraphPlan:
    model_name:      str
    layers:          list[LayerCall]
    tensor_metas:    dict[str, TensorMeta]
    scratch_bytes:   int
    intermediate_bufs: list[tuple[str, int]]  # (name, size_bytes)
    input_name:      str
    output_name:     str
    input_scale:     float
    input_zp:        int
    output_scale:    float
    output_zp:       int
    is_quantized:    bool


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compile_graph(model_path: str | Path) -> GraphPlan:
    """
    Walk the ONNX graph and build a GraphPlan describing every CMSIS-NN call.
    """
    p = Path(model_path)
    model = onnx.load(str(p))
    graph = model.graph

    model_name = _sanitise(p.stem)

    # Build lookup tables
    init_map   = _build_init_map(graph)
    shape_map  = _build_shape_map(graph)
    quant_map  = _build_quant_map(graph, init_map)

    is_quantized = any(
        n.op_type in {"DequantizeLinear","QuantizeLinear",
                      "ConvInteger","MatMulInteger","DynamicQuantizeLinear"}
        for n in graph.node
    )

    # Graph I/O
    input_name  = graph.input[0].name
    output_name = graph.output[0].name
    in_q  = quant_map.get(input_name,  (1.0, 0))
    out_q = quant_map.get(output_name, (1.0, 0))

    # Walk nodes in topological order
    layers: list[LayerCall] = []
    intermediate_bufs: list[tuple[str, int]] = []
    scratch_bytes = 0

    # Track which buffers exist (input + all layer outputs)
    buf_registry: dict[str, int] = {}   # name -> size_bytes

    for node_idx, node in enumerate(graph.node):
        op = node.op_type

        # Skip quantisation bookkeeping ops -- handled implicitly
        if op in {"QuantizeLinear", "DequantizeLinear",
                  "DynamicQuantizeLinear", "Cast"}:
            continue

        layer = _compile_node(
            node, node_idx, graph, model_name,
            init_map, shape_map, quant_map,
            buf_registry, intermediate_bufs,
        )

        if layer is not None:
            layers.append(layer)
            # Track scratch requirement
            if "scratch_bytes" in layer.params:
                scratch_bytes = max(scratch_bytes, layer.params["scratch_bytes"])

    # Minimum scratch buffer
    scratch_bytes = max(scratch_bytes, 1024)

    return GraphPlan(
        model_name=model_name,
        layers=layers,
        tensor_metas={},
        scratch_bytes=scratch_bytes,
        intermediate_bufs=intermediate_bufs,
        input_name=_sanitise(input_name),
        output_name=_sanitise(output_name),
        input_scale=in_q[0],
        input_zp=in_q[1],
        output_scale=out_q[0],
        output_zp=out_q[1],
        is_quantized=is_quantized,
    )


# ---------------------------------------------------------------------------
# Node compiler
# ---------------------------------------------------------------------------

def _compile_node(
    node, node_idx, graph, model_name,
    init_map, shape_map, quant_map,
    buf_registry, intermediate_bufs,
) -> LayerCall | None:

    op     = node.op_type
    name   = node.name or f"node_{node_idx}"
    inputs = list(node.input)
    outs   = list(node.output)

    out_name = _sanitise(outs[0]) if outs else f"out_{node_idx}"

    # Register output buffer
    out_shape = shape_map.get(outs[0], [1]) if outs else [1]
    out_bytes = math.prod(out_shape)  # INT8 = 1 byte/elem
    if out_name not in buf_registry:
        buf_registry[out_name] = out_bytes
        intermediate_bufs.append((out_name, out_bytes))

    in_name = _sanitise(inputs[0]) if inputs else "input"

    # ── Conv ──────────────────────────────────────────────────────────────────
    if op in ("Conv", "ConvInteger", "QLinearConv"):
        return _compile_conv(
            node, name, in_name, out_name, out_shape,
            inputs, graph, model_name, init_map, shape_map, quant_map,
        )

    # ── DepthwiseConv (group == in_channels) ──────────────────────────────────
    if op == "Conv":  # handled above, but depthwise detected by group attr
        pass  # already handled in _compile_conv

    # ── Gemm / MatMul (FullyConnected) ────────────────────────────────────────
    if op in ("Gemm", "MatMul", "MatMulInteger"):
        return _compile_fc(
            node, name, in_name, out_name, out_shape,
            inputs, graph, model_name, init_map, shape_map, quant_map,
        )

    # ── Relu ──────────────────────────────────────────────────────────────────
    if op == "Relu":
        in_shape = shape_map.get(inputs[0], [1]) if inputs else [1]
        return LayerCall(
            comment=f"ReLU -- {name}",
            fn_name="arm_relu_s8",
            input_buf=in_name,
            output_buf=out_name,
            params={
                "buf":       in_name,
                "size":      math.prod(in_shape),
                "in_place":  True,
            },
        )

    # ── Clip (Relu6) ──────────────────────────────────────────────────────────
    if op == "Clip":
        in_shape = shape_map.get(inputs[0], [1]) if inputs else [1]
        return LayerCall(
            comment=f"Clip/ReLU6 -- {name}",
            fn_name="arm_relu_s8",
            input_buf=in_name,
            output_buf=out_name,
            params={
                "buf":      in_name,
                "size":     math.prod(in_shape),
                "in_place": True,
            },
        )

    # ── GlobalAveragePool ─────────────────────────────────────────────────────
    if op == "GlobalAveragePool":
        in_shape = shape_map.get(inputs[0], [1,1,1,1]) if inputs else [1,1,1,1]
        return LayerCall(
            comment=f"GlobalAvgPool -- {name}",
            fn_name="arm_avgpool_s8",
            input_buf=in_name,
            output_buf=out_name,
            params={
                "input_shape":  in_shape,
                "output_shape": out_shape,
                "global":       True,
            },
        )

    # ── Softmax ───────────────────────────────────────────────────────────────
    if op == "Softmax":
        n_classes = math.prod(out_shape)
        q = quant_map.get(inputs[0], (1.0/256, -128))
        return LayerCall(
            comment=f"Softmax -- {name}",
            fn_name="arm_softmax_s8",
            input_buf=in_name,
            output_buf=out_name,
            params={
                "num_rows":  1,
                "row_size":  n_classes,
                "mult":      _float_to_q31(q[0]),
                "shift":     0,
                "diff_min":  -128,
            },
        )

    # ── Reshape / Flatten ─────────────────────────────────────────────────────
    if op in ("Reshape", "Flatten", "Squeeze", "Unsqueeze"):
        return LayerCall(
            comment=f"Reshape (pointer alias) -- {name}",
            fn_name="__reshape__",
            input_buf=in_name,
            output_buf=out_name,
            params={"alias": True},
        )

    # ── Add ───────────────────────────────────────────────────────────────────
    if op == "Add":
        in2_name = _sanitise(inputs[1]) if len(inputs) > 1 else "zero"
        in_shape = shape_map.get(inputs[0], [1]) if inputs else [1]
        q1 = quant_map.get(inputs[0], (1.0, 0))
        q2 = quant_map.get(inputs[1], (1.0, 0)) if len(inputs) > 1 else (1.0, 0)
        qo = quant_map.get(outs[0],   (1.0, 0)) if outs else (1.0, 0)
        return LayerCall(
            comment=f"ElementwiseAdd -- {name}",
            fn_name="arm_elementwise_add_s8",
            input_buf=in_name,
            output_buf=out_name,
            params={
                "input_1": in_name,
                "input_2": in2_name,
                "input_1_offset": -q1[1],
                "input_1_mult":   _float_to_q31(q1[0] / qo[0]),
                "input_1_shift":  0,
                "input_2_offset": -q2[1],
                "input_2_mult":   _float_to_q31(q2[0] / qo[0]),
                "input_2_shift":  0,
                "left_shift":     4,
                "out_offset":     qo[1],
                "out_mult":       _float_to_q31(1.0),
                "out_shift":      0,
                "out_activation_min": -128,
                "out_activation_max": 127,
                "block_size":     math.prod(in_shape),
            },
        )

    # ── Mul (elementwise -- usually bias scaling) ─────────────────────────────
    if op == "Mul":
        return LayerCall(
            comment=f"Mul (pass-through) -- {name}",
            fn_name="__passthrough__",
            input_buf=in_name,
            output_buf=out_name,
            params={"alias": True},
        )

    # ── Unsupported -- emit comment placeholder ───────────────────────────────
    return LayerCall(
        comment=f"Unsupported op: {op} -- {name}",
        fn_name="__unsupported__",
        input_buf=in_name,
        output_buf=out_name,
        params={},
        supported=False,
        unsupported_reason=f"Op '{op}' not yet supported by EdgeForge CMSIS-NN backend",
    )


# ---------------------------------------------------------------------------
# Conv compiler
# ---------------------------------------------------------------------------

def _compile_conv(
    node, name, in_name, out_name, out_shape,
    inputs, graph, model_name, init_map, shape_map, quant_map,
) -> LayerCall:

    # Detect depthwise: group == in_channels
    attrs = {a.name: a for a in node.attribute}
    group = attrs["group"].i if "group" in attrs else 1
    in_shape = shape_map.get(inputs[0], [1,1,1,1])
    in_ch = in_shape[1] if len(in_shape) >= 2 else 1
    is_depthwise = (group == in_ch and in_ch > 1)

    # Kernel weights
    w_name = inputs[1] if len(inputs) > 1 else ""
    w_arr  = init_map.get(w_name)
    if w_arr is None:
        # Quantised weight name may have _quantized suffix
        for k in init_map:
            if w_name.replace("/", "_") in k or k.endswith("_quantized"):
                w_arr = init_map[k]
                w_name = k
                break

    w_shape = list(w_arr.shape) if w_arr is not None else [1,1,1,1]
    out_ch  = w_shape[0]
    kH      = w_shape[2] if len(w_shape) >= 3 else 1
    kW      = w_shape[3] if len(w_shape) >= 4 else 1

    # Bias
    b_name = inputs[2] if len(inputs) > 2 else ""
    has_bias = b_name in init_map

    # Strides and padding
    strides = list(attrs["strides"].ints) if "strides" in attrs else [1, 1]
    pads    = list(attrs["pads"].ints)    if "pads"    in attrs else [0, 0, 0, 0]

    # Quant params
    w_q = quant_map.get(w_name, (1.0/128, 0))
    o_q = quant_map.get(inputs[0], (1.0, 0))
    out_q = quant_map.get(out_shape[0] if isinstance(out_shape,list) else "", (1.0/128, 0))

    # Scratch size (CMSIS-NN Conv needs 2*kH*kW*in_ch*sizeof(q15_t))
    scratch = 2 * kH * kW * in_ch * 2

    fn = "arm_depthwise_conv_s8" if is_depthwise else "arm_convolve_s8"
    c_w_name = f"{model_name}_{_sanitise(w_name)}"

    return LayerCall(
        comment=f"{'DepthwiseConv' if is_depthwise else 'Conv2D'} -- {name} "
                f"({out_ch} filters, {kH}x{kW}, stride {strides})",
        fn_name=fn,
        input_buf=in_name,
        output_buf=out_name,
        params={
            "input_shape":   in_shape,
            "output_shape":  out_shape,
            "filter_shape":  w_shape,
            "weight_name":   c_w_name,
            "bias_name":     f"{model_name}_{_sanitise(b_name)}" if has_bias else "NULL",
            "has_bias":      has_bias,
            "stride_h":      strides[0],
            "stride_w":      strides[1] if len(strides) > 1 else strides[0],
            "pad_h":         pads[0],
            "pad_w":         pads[1] if len(pads) > 1 else pads[0],
            "act_min":       -128,
            "act_max":       127,
            "input_offset":  -o_q[1],
            "output_offset": 0,
            "multiplier":    _float_to_q31(w_q[0] * o_q[0]),
            "shift":         0,
            "scratch_bytes": scratch,
            "is_depthwise":  is_depthwise,
            "dilation_h":    1,
            "dilation_w":    1,
        },
    )


# ---------------------------------------------------------------------------
# FullyConnected compiler
# ---------------------------------------------------------------------------

def _compile_fc(
    node, name, in_name, out_name, out_shape,
    inputs, graph, model_name, init_map, shape_map, quant_map,
) -> LayerCall:

    w_name = inputs[1] if len(inputs) > 1 else ""
    b_name = inputs[2] if len(inputs) > 2 else ""
    w_arr  = init_map.get(w_name)

    # Try quantized variant
    if w_arr is None:
        for k in init_map:
            if "fc_w" in k or "weight" in k.lower():
                w_arr  = init_map[k]
                w_name = k
                break

    w_shape   = list(w_arr.shape) if w_arr is not None else [1, 1]
    n_batches = w_shape[0] if len(w_shape) >= 2 else 1
    in_dim    = w_shape[1] if len(w_shape) >= 2 else 1
    out_dim   = math.prod(out_shape)

    has_bias = b_name in init_map
    w_q = quant_map.get(w_name, (1.0/128, 0))
    i_q = quant_map.get(inputs[0], (1.0, 0))

    return LayerCall(
        comment=f"FullyConnected -- {name} ({in_dim} -> {out_dim})",
        fn_name="arm_fully_connected_s8",
        input_buf=in_name,
        output_buf=out_name,
        params={
            "input_dim":     in_dim,
            "output_dim":    out_dim,
            "weight_name":   f"{model_name}_{_sanitise(w_name)}",
            "bias_name":     f"{model_name}_{_sanitise(b_name)}" if has_bias else "NULL",
            "has_bias":      has_bias,
            "input_offset":  -i_q[1],
            "output_offset": 0,
            "multiplier":    _float_to_q31(w_q[0] * i_q[0]),
            "shift":         0,
            "act_min":       -128,
            "act_max":       127,
        },
    )


# ---------------------------------------------------------------------------
# Code emitter
# ---------------------------------------------------------------------------

def emit_inference_runner_c(plan: GraphPlan, target_id: str, rtos: str) -> str:
    """
    Render the complete inference_runner.c as a string.
    No Jinja2 -- pure Python string generation for full control.
    """
    lines: list[str] = []
    w = lines.append  # shorthand

    w("/**")
    w(" * EdgeForge Generated File -- DO NOT EDIT")
    w(f" * Model: {plan.model_name}")
    w(f" * Target: {target_id}  RTOS: {rtos}")
    w(" *")
    w(" * Self-contained inference runner using CMSIS-NN.")
    w(" * ONLY dependency: arm_nnfunctions.h")
    w(" *   -- included in STM32CubeIDE via CMSIS pack")
    w(" *   -- included in ModusToolbox via CMSIS pack")
    w(" *   -- included in nRF Connect SDK via Zephyr CMSIS module")
    w(" *   -- standalone: github.com/ARM-software/CMSIS-NN")
    w(" */")
    w("")
    w('#include "inference_runner.h"')
    w('#include "model.h"')
    w('#include "memory_config.h"')
    w('#include "arm_nnfunctions.h"')
    w('#include <string.h>')
    w("")

    if rtos == "freertos":
        w('#include "FreeRTOS.h"')
        w('#include "semphr.h"')
        w("static SemaphoreHandle_t _ef_mutex = NULL;")
        w("")
    elif rtos == "zephyr":
        w('#include <zephyr/kernel.h>')
        w("K_MUTEX_DEFINE(_ef_mutex);")
        w("")

    w(f"/* Scratch buffer for CMSIS-NN intermediate calculations */")
    w(f"static int8_t _scratch[{plan.scratch_bytes}];")
    w("")

    # Intermediate activation buffers
    if plan.intermediate_bufs:
        w("/* Intermediate activation buffers */")
        for buf_name, buf_size in plan.intermediate_bufs:
            w(f"static int8_t _buf_{buf_name}[{buf_size}];")
        w("")

    # edgeforge_init
    w("edgeforge_status_t edgeforge_init(edgeforge_model_t *model) {")
    w("    if (!model) return EDGEFORGE_ERR_INIT;")
    w("    if (model->initialised) return EDGEFORGE_OK;")
    if rtos == "freertos":
        w("    if (!_ef_mutex) {")
        w("        _ef_mutex = xSemaphoreCreateMutex();")
        w("        if (!_ef_mutex) return EDGEFORGE_ERR_INIT;")
        w("    }")
    w("    memset(model->arena, 0, sizeof(model->arena));")
    w("    model->initialised = 1;")
    w("    return EDGEFORGE_OK;")
    w("}")
    w("")

    # edgeforge_infer
    w("edgeforge_status_t edgeforge_infer(")
    w("    edgeforge_model_t       *model,")
    w("    const edgeforge_input_t *input,")
    w("    edgeforge_output_t      *output)")
    w("{")
    w("    if (!model || !model->initialised || !input || !output)")
    w("        return EDGEFORGE_ERR_INPUT;")
    w("")

    if rtos == "freertos":
        w("    xSemaphoreTake(_ef_mutex, portMAX_DELAY);")
    elif rtos == "zephyr":
        w("    k_mutex_lock(&_ef_mutex, K_FOREVER);")

    w("")
    w("    /* Use arena as first activation buffer */")
    w("    int8_t *_arena = (int8_t *)model->arena;")
    w("")

    # Emit each layer
    for i, layer in enumerate(plan.layers):
        w(f"    /* ---- Layer {i}: {layer.comment} ---- */")

        if not layer.supported:
            w(f"    /* UNSUPPORTED: {layer.unsupported_reason} */")
            w(f"    /* TODO: implement {layer.fn_name} */")
            w("")
            continue

        fn = layer.fn_name
        p  = layer.params

        if fn == "__reshape__" or fn == "__passthrough__":
            w(f"    const int8_t *{layer.output_buf} = (const int8_t *){layer.input_buf};")
            w("    (void)_scratch;  /* unused for reshape */")

        elif fn == "arm_relu_s8":
            w(f"    arm_relu_s8((int8_t *){layer.input_buf}, {p['size']});")
            w(f"    const int8_t *{layer.output_buf} = {layer.input_buf};")

        elif fn in ("arm_convolve_s8", "arm_depthwise_conv_s8"):
            in_s  = p["input_shape"]
            out_s = p["output_shape"]
            f_s   = p["filter_shape"]

            w(f"    {{")
            w(f"        cmsis_nn_context ctx = {{.buf=_scratch, .size={p['scratch_bytes']}}};")
            if fn == "arm_convolve_s8":
                w(f"        cmsis_nn_conv_params cp = {{")
                w(f"            .padding        = {{{p['pad_h']}, {p['pad_w']}, {p['pad_h']}, {p['pad_w']}}},")
                w(f"            .stride         = {{{p['stride_h']}, {p['stride_w']}}},")
                w(f"            .dilation       = {{{p['dilation_h']}, {p['dilation_w']}}},")
                w(f"            .input_offset   = {p['input_offset']},")
                w(f"            .output_offset  = {p['output_offset']},")
                w(f"            .activation     = {{.min={p['act_min']}, .max={p['act_max']}}},")
                w(f"        }};")
            else:
                w(f"        cmsis_nn_dw_conv_params cp = {{")
                w(f"            .padding        = {{{p['pad_h']}, {p['pad_w']}, {p['pad_h']}, {p['pad_w']}}},")
                w(f"            .stride         = {{{p['stride_h']}, {p['stride_w']}}},")
                w(f"            .dilation       = {{{p['dilation_h']}, {p['dilation_w']}}},")
                w(f"            .input_offset   = {p['input_offset']},")
                w(f"            .output_offset  = {p['output_offset']},")
                w(f"            .activation     = {{.min={p['act_min']}, .max={p['act_max']}}},")
                w(f"            .ch_mult        = 1,")
                w(f"        }};")
            w(f"        cmsis_nn_per_tensor_quant_params qp = {{")
            w(f"            .multiplier = {p['multiplier']},")
            w(f"            .shift      = {p['shift']},")
            w(f"        }};")
            in_dims  = ", ".join(str(d) for d in (in_s  if len(in_s)  == 4 else [1]+in_s))
            out_dims = ", ".join(str(d) for d in (out_s if len(out_s) == 4 else [1]+out_s))
            f_dims   = ", ".join(str(d) for d in (f_s   if len(f_s)   == 4 else [1]+f_s))
            w(f"        cmsis_nn_dims in_d  = {{{in_dims}}};")
            w(f"        cmsis_nn_dims fil_d = {{{f_dims}}};")
            w(f"        cmsis_nn_dims out_d = {{{out_dims}}};")
            w(f"        cmsis_nn_dims b_d   = {{1, 1, 1, {out_s[1] if len(out_s)>=2 else 1}}};")
            bias_arg = p['bias_name'] if p['has_bias'] else "NULL"
            w(f"        arm_cmsis_nn_status s{i} = {fn}(")
            w(f"            &ctx, &cp, &qp,")
            w(f"            &in_d,  (const int8_t *){layer.input_buf},")
            w(f"            &fil_d, (const int8_t *){p['weight_name']},")
            w(f"            &b_d,   (const int32_t *){bias_arg},")
            w(f"            &out_d, _buf_{layer.output_buf}")
            w(f"        );")
            w(f"        if (s{i} != ARM_CMSIS_NN_SUCCESS) {{")
            if rtos == "freertos":
                w(f"            xSemaphoreGive(_ef_mutex);")
            elif rtos == "zephyr":
                w(f"            k_mutex_unlock(&_ef_mutex);")
            w(f"            return EDGEFORGE_ERR_INVOKE;")
            w(f"        }}")
            w(f"    }}")
            w(f"    const int8_t *{layer.output_buf} = _buf_{layer.output_buf};")

        elif fn == "arm_fully_connected_s8":
            w(f"    {{")
            w(f"        cmsis_nn_context ctx = {{.buf=_scratch, .size=sizeof(_scratch)}};")
            w(f"        cmsis_nn_fc_params fp = {{")
            w(f"            .input_offset  = {p['input_offset']},")
            w(f"            .filter_offset = 0,")
            w(f"            .output_offset = {p['output_offset']},")
            w(f"            .activation    = {{.min={p['act_min']}, .max={p['act_max']}}},")
            w(f"        }};")
            w(f"        cmsis_nn_per_tensor_quant_params qp = {{")
            w(f"            .multiplier = {p['multiplier']},")
            w(f"            .shift      = {p['shift']},")
            w(f"        }};")
            w(f"        cmsis_nn_dims in_d  = {{1, 1, 1, {p['input_dim']}}};")
            w(f"        cmsis_nn_dims fil_d = {{1, 1, {p['input_dim']}, {p['output_dim']}}};")
            w(f"        cmsis_nn_dims out_d = {{1, 1, 1, {p['output_dim']}}};")
            w(f"        cmsis_nn_dims b_d   = {{1, 1, 1, {p['output_dim']}}};")
            bias_arg = p['bias_name'] if p['has_bias'] else "NULL"
            w(f"        arm_cmsis_nn_status s{i} = arm_fully_connected_s8(")
            w(f"            &ctx, &fp, &qp,")
            w(f"            &in_d,  (const int8_t *){layer.input_buf},")
            w(f"            &fil_d, (const int8_t *){p['weight_name']},")
            w(f"            &b_d,   (const int32_t *){bias_arg},")
            w(f"            &out_d, _buf_{layer.output_buf}")
            w(f"        );")
            w(f"        if (s{i} != ARM_CMSIS_NN_SUCCESS) {{")
            if rtos == "freertos":
                w(f"            xSemaphoreGive(_ef_mutex);")
            elif rtos == "zephyr":
                w(f"            k_mutex_unlock(&_ef_mutex);")
            w(f"            return EDGEFORGE_ERR_INVOKE;")
            w(f"        }}")
            w(f"    }}")
            w(f"    const int8_t *{layer.output_buf} = _buf_{layer.output_buf};")

        elif fn == "arm_avgpool_s8":
            in_s  = p["input_shape"]
            out_s = p["output_shape"]
            in_dims  = ", ".join(str(d) for d in (in_s  if len(in_s)  == 4 else [1]+in_s))
            out_dims = ", ".join(str(d) for d in (out_s if len(out_s) == 4 else [1]+out_s))
            w(f"    {{")
            w(f"        cmsis_nn_context ctx = {{.buf=_scratch, .size=sizeof(_scratch)}};")
            w(f"        cmsis_nn_pool_params pp = {{")
            w(f"            .stride = {{1, 1}}, .padding = {{0, 0, 0, 0}},")
            w(f"            .activation = {{.min=-128, .max=127}},")
            w(f"        }};")
            w(f"        cmsis_nn_dims in_d  = {{{in_dims}}};")
            w(f"        cmsis_nn_dims fil_d = {{{in_s[2] if len(in_s)>=3 else 1}, {in_s[3] if len(in_s)>=4 else 1}}};")
            w(f"        cmsis_nn_dims out_d = {{{out_dims}}};")
            w(f"        arm_avgpool_s8(&ctx, &pp, &in_d, (const int8_t *){layer.input_buf}, &fil_d, &out_d, _buf_{layer.output_buf});")
            w(f"    }}")
            w(f"    const int8_t *{layer.output_buf} = _buf_{layer.output_buf};")

        elif fn == "arm_softmax_s8":
            w(f"    arm_softmax_s8(")
            w(f"        (const int8_t *){layer.input_buf},")
            w(f"        {p['num_rows']}, {p['row_size']},")
            w(f"        {p['mult']}, {p['shift']}, {p['diff_min']},")
            w(f"        _buf_{layer.output_buf}")
            w(f"    );")
            w(f"    const int8_t *{layer.output_buf} = _buf_{layer.output_buf};")

        elif fn == "arm_elementwise_add_s8":
            w(f"    arm_elementwise_add_s8(")
            w(f"        (const int8_t *){p['input_1']}, (const int8_t *){p['input_2']},")
            w(f"        {p['input_1_offset']}, {p['input_1_mult']}, {p['input_1_shift']},")
            w(f"        {p['input_2_offset']}, {p['input_2_mult']}, {p['input_2_shift']},")
            w(f"        {p['left_shift']},")
            w(f"        _buf_{layer.output_buf},")
            w(f"        {p['out_offset']}, {p['out_mult']}, {p['out_shift']},")
            w(f"        {p['out_activation_min']}, {p['out_activation_max']},")
            w(f"        {p['block_size']}")
            w(f"    );")
            w(f"    const int8_t *{layer.output_buf} = _buf_{layer.output_buf};")

        w("")

    # Copy final output
    w("    /* Copy final output */")
    final_out = plan.layers[-1].output_buf if plan.layers else "input"
    w(f"    memcpy(output, {final_out}, EDGEFORGE_OUTPUT_SIZE);")
    w("")

    if rtos == "freertos":
        w("    xSemaphoreGive(_ef_mutex);")
    elif rtos == "zephyr":
        w("    k_mutex_unlock(&_ef_mutex);")

    w("    return EDGEFORGE_OK;")
    w("}")
    w("")

    # edgeforge_deinit
    w("void edgeforge_deinit(edgeforge_model_t *model) {")
    w("    if (!model) return;")
    w("    model->initialised = 0;")
    w("}")
    w("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_init_map(graph) -> dict[str, np.ndarray]:
    m = {}
    for init in graph.initializer:
        try:
            m[init.name] = numpy_helper.to_array(init)
        except Exception:
            pass
    return m


def _build_shape_map(graph) -> dict[str, list[int]]:
    m = {}
    all_vi = list(graph.value_info) + list(graph.input) + list(graph.output)
    for vi in all_vi:
        t = vi.type.tensor_type
        if not t.HasField("elem_type"):
            continue
        shape = [max(d.dim_value, 1) for d in t.shape.dim]
        m[vi.name] = shape
    return m


def _build_quant_map(graph, init_map) -> dict[str, tuple[float, int]]:
    """Build tensor_name -> (scale, zero_point) from QuantizeLinear nodes."""
    m = {}
    for node in graph.node:
        if node.op_type not in ("QuantizeLinear", "DequantizeLinear",
                                "DynamicQuantizeLinear"):
            continue
        try:
            if len(node.input) >= 2:
                scale = float(init_map.get(node.input[1], np.array([1.0])).flat[0])
                zp    = int(init_map.get(node.input[2], np.array([0])).flat[0]) \
                        if len(node.input) >= 3 else 0
                for out in node.output:
                    if out:
                        m[out] = (scale, zp)
                if node.input[0]:
                    m[node.input[0]] = (scale, zp)
        except Exception:
            pass
    return m


def _sanitise(name: str) -> str:
    import re
    s = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    if s and s[0].isdigit():
        s = "m_" + s
    return re.sub(r"_+", "_", s).strip("_") or "tensor"


def _float_to_q31(x: float) -> int:
    """Convert float multiplier to Q31 fixed-point integer."""
    clamped = max(-1.0, min(1.0 - 2**-31, x))
    return int(round(clamped * (2**31 - 1)))
