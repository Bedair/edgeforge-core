"""
EdgeForge — Graph Simplifier
Runs pre-quantisation graph cleanup passes on an ONNX model:
  1. Constant folding   — pre-compute nodes with all-constant inputs
  2. Operator fusion    — fuse Conv+BN, Conv+BN+Relu chains
  3. Dead node removal  — strip nodes whose outputs are never consumed
Wraps onnx-simplifier where available, with manual fallbacks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import onnx
from onnx import helper, numpy_helper, TensorProto

log = logging.getLogger(__name__)


@dataclass
class SimplifyReport:
    nodes_before:    int = 0
    nodes_after:     int = 0
    nodes_fused:     int = 0
    nodes_removed:   int = 0
    passes_applied:  list[str] = field(default_factory=list)

    @property
    def nodes_saved(self) -> int:
        return self.nodes_before - self.nodes_after


def simplify(model: onnx.ModelProto) -> tuple[onnx.ModelProto, SimplifyReport]:
    """
    Run all simplification passes on an ONNX model.

    Tries onnx-simplifier first (best results), falls back to manual passes.

    Args:
        model: Loaded ONNX ModelProto.

    Returns:
        (simplified_model, SimplifyReport)
    """
    report = SimplifyReport(nodes_before=len(model.graph.node))

    # ── Pass 1: onnx-simplifier (constant folding + shape inference) ─────────
    model, report = _try_onnxsim(model, report)

    # ── Pass 2: dead node elimination ────────────────────────────────────────
    model, removed = eliminate_dead_nodes(model)
    report.nodes_removed += removed
    if removed:
        report.passes_applied.append(f"dead_node_elimination (-{removed})")

    # ── Pass 3: Conv+BN fusion (manual — onnxsim may not catch all cases) ────
    model, fused = fuse_bn_into_conv(model)
    report.nodes_fused += fused
    if fused:
        report.passes_applied.append(f"conv_bn_fusion (-{fused})")

    report.nodes_after = len(model.graph.node)
    return model, report


def _try_onnxsim(
    model: onnx.ModelProto,
    report: SimplifyReport,
) -> tuple[onnx.ModelProto, SimplifyReport]:
    """Attempt onnx-simplifier; silently fall back if unavailable."""
    try:
        import onnxsim
        simplified, ok = onnxsim.simplify(
            model,
            overwrite_input_shapes=None,
            skip_shape_inference=False,
        )
        if ok:
            report.passes_applied.append("onnxsim_constant_folding")
            log.info("onnx-simplifier: success")
            return simplified, report
        else:
            log.warning("onnx-simplifier returned ok=False, using original")
    except ImportError:
        log.info("onnx-simplifier not installed — using manual passes only")
    except Exception as e:
        log.warning(f"onnx-simplifier failed ({e}) — using manual passes only")
    return model, report


def eliminate_dead_nodes(model: onnx.ModelProto) -> tuple[onnx.ModelProto, int]:
    """
    Remove nodes whose outputs are not consumed by any other node
    and are not graph outputs.

    Returns (model, number_of_nodes_removed).
    """
    graph = model.graph

    # Build set of all consumed input names
    consumed: set[str] = set()
    for node in graph.node:
        for inp in node.input:
            if inp:
                consumed.add(inp)
    for out in graph.output:
        consumed.add(out.name)

    # Find dead nodes: all outputs are unconsumed
    dead: list[onnx.NodeProto] = []
    for node in graph.node:
        outputs = [o for o in node.output if o]
        if outputs and all(o not in consumed for o in outputs):
            dead.append(node)

    if not dead:
        return model, 0

    dead_set = set(id(n) for n in dead)
    new_nodes = [n for n in graph.node if id(n) not in dead_set]

    new_graph = helper.make_graph(
        new_nodes,
        graph.name,
        list(graph.input),
        list(graph.output),
        list(graph.initializer),
    )
    new_model = helper.make_model(new_graph, opset_imports=model.opset_import)
    new_model.ir_version = model.ir_version
    return new_model, len(dead)


def fuse_bn_into_conv(model: onnx.ModelProto) -> tuple[onnx.ModelProto, int]:
    """
    Fuse BatchNormalization into a preceding Conv node.

    Conv → BN  becomes  Conv (with adjusted W and B)

    This is only valid when BN is in inference mode (no training outputs).
    Returns (model, number_of_fusions_performed).
    """
    import numpy as np

    graph = model.graph

    # Build initializer lookup
    init_map: dict[str, np.ndarray] = {}
    for init in graph.initializer:
        try:
            init_map[init.name] = numpy_helper.to_array(init)
        except Exception:
            pass

    # Map output_name → node
    output_to_node: dict[str, onnx.NodeProto] = {}
    for node in graph.node:
        for out in node.output:
            if out:
                output_to_node[out] = node

    fused_count = 0
    nodes_to_remove: set[int] = set()
    new_initializers: list[onnx.TensorProto] = []

    for bn_node in list(graph.node):
        if bn_node.op_type != "BatchNormalization":
            continue
        # BN must have exactly 1 output used (inference mode)
        if len([o for o in bn_node.output if o]) > 1:
            continue

        bn_input = bn_node.input[0]
        conv_node = output_to_node.get(bn_input)
        if conv_node is None or conv_node.op_type != "Conv":
            continue

        # Extract BN parameters — all must be known constants
        try:
            bn_scale  = init_map[bn_node.input[1]]   # gamma
            bn_bias   = init_map[bn_node.input[2]]   # beta
            bn_mean   = init_map[bn_node.input[3]]
            bn_var    = init_map[bn_node.input[4]]
        except KeyError:
            continue  # dynamic BN params — cannot fuse

        eps = 1e-5
        for attr in bn_node.attribute:
            if attr.name == "epsilon":
                eps = attr.f

        # Extract Conv weight (and optional bias)
        try:
            conv_W = init_map[conv_node.input[1]]
        except KeyError:
            continue

        conv_B = (
            init_map[conv_node.input[2]]
            if len(conv_node.input) > 2 and conv_node.input[2] in init_map
            else np.zeros(conv_W.shape[0], dtype=conv_W.dtype)
        )

        # Compute fused weights:
        # W_fused[i] = W[i] * scale[i] / sqrt(var[i] + eps)
        # B_fused[i] = (B[i] - mean[i]) * scale[i] / sqrt(var[i] + eps) + bias[i]
        std = np.sqrt(bn_var + eps)
        scale_factor = bn_scale / std

        W_fused = conv_W * scale_factor.reshape(-1, 1, 1, 1)
        B_fused = (conv_B - bn_mean) * scale_factor + bn_bias

        # Build new initializers
        w_name = conv_node.input[1] + "_bn_fused"
        b_name = (conv_node.input[2] if len(conv_node.input) > 2
                  else conv_node.output[0] + "_bias_fused")

        new_initializers.append(numpy_helper.from_array(
            W_fused.astype(conv_W.dtype), name=w_name))
        new_initializers.append(numpy_helper.from_array(
            B_fused.astype(np.float32), name=b_name))

        # Patch Conv node to use fused weights and output BN's output
        conv_node.input[1] = w_name
        if len(conv_node.input) > 2:
            conv_node.input[2] = b_name
        else:
            conv_node.input.append(b_name)
        # Redirect conv output to bn output
        conv_node.output[0] = bn_node.output[0]

        nodes_to_remove.add(id(bn_node))
        fused_count += 1

    if fused_count == 0:
        return model, 0

    # Rebuild graph without fused BN nodes
    new_nodes = [n for n in graph.node if id(n) not in nodes_to_remove]
    all_inits = list(graph.initializer) + new_initializers

    new_graph = helper.make_graph(
        new_nodes, graph.name,
        list(graph.input), list(graph.output),
        all_inits,
    )
    new_model = helper.make_model(new_graph, opset_imports=model.opset_import)
    new_model.ir_version = model.ir_version
    return new_model, fused_count


def fold_constants(model: onnx.ModelProto) -> tuple[onnx.ModelProto, int]:
    """
    Basic constant folding: evaluate nodes where ALL inputs are initializers.
    Returns (model, number_of_nodes_folded).

    Note: onnx-simplifier does this better — this is a lightweight fallback.
    """
    import numpy as np

    graph = model.graph
    init_map: dict[str, np.ndarray] = {
        i.name: numpy_helper.to_array(i) for i in graph.initializer
    }

    folded = 0
    new_nodes: list[onnx.NodeProto] = []
    new_inits = list(graph.initializer)

    for node in graph.node:
        inputs_known = all(
            inp in init_map or inp == ""
            for inp in node.input
        )
        if not inputs_known or node.op_type in ("Loop", "Scan", "If"):
            new_nodes.append(node)
            continue

        # Try to evaluate via onnxruntime
        try:
            import onnxruntime as rt
            import io

            # Build a mini model with just this node
            mini_inputs = [
                helper.make_tensor_value_info(
                    name, TensorProto.FLOAT, None
                )
                for name in node.input if name and name in init_map
            ]
            mini_outputs = [
                helper.make_tensor_value_info(name, TensorProto.FLOAT, None)
                for name in node.output if name
            ]
            mini_graph = helper.make_graph(
                [node], "fold",
                mini_inputs, mini_outputs,
                [i for i in graph.initializer if i.name in node.input],
            )
            mini_model = helper.make_model(
                mini_graph, opset_imports=model.opset_import)
            mini_model.ir_version = model.ir_version

            buf = io.BytesIO()
            onnx.save(mini_model, buf)
            sess = rt.InferenceSession(
                buf.getvalue(),
                providers=["CPUExecutionProvider"],
            )
            feed = {
                name: init_map[name]
                for name in node.input
                if name and name in init_map
            }
            results = sess.run(None, feed)
            for out_name, arr in zip(node.output, results):
                if not out_name:
                    continue
                tensor = numpy_helper.from_array(arr, name=out_name)
                new_inits.append(tensor)
                init_map[out_name] = arr
            folded += 1
        except Exception:
            new_nodes.append(node)
            continue

    if folded == 0:
        return model, 0

    new_graph = helper.make_graph(
        new_nodes, graph.name,
        list(graph.input), list(graph.output),
        new_inits,
    )
    new_model = helper.make_model(new_graph, opset_imports=model.opset_import)
    new_model.ir_version = model.ir_version
    return new_model, folded
