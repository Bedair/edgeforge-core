"""
generate_test_model.py
Generates a realistic keyword spotting-like CNN model for testing EdgeForge.

Architecture: DS-CNN (Depthwise Separable CNN) — similar to what runs
on Cortex-M4 devices for audio classification. Small enough to fit on
all three v1 target boards.

Output: test_model.onnx (~120KB float32, ~30KB after INT8 quantisation)

Usage:
    python generate_test_model.py
    edgeforge analyze  test_model.onnx
    edgeforge optimize test_model.onnx --mcu=stm32f407
    edgeforge analyze  test_model_opt.onnx
"""

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import onnx.shape_inference


def make_conv_weights(name: str, out_ch: int, in_ch: int, kH: int, kW: int) -> onnx.TensorProto:
    w = np.random.randn(out_ch, in_ch, kH, kW).astype(np.float32) * 0.1
    return numpy_helper.from_array(w, name=name)


def make_bias(name: str, size: int) -> onnx.TensorProto:
    b = np.zeros(size, dtype=np.float32)
    return numpy_helper.from_array(b, name=name)


def make_bn_params(name: str, size: int):
    """BatchNorm: scale, bias, mean, var"""
    scale = numpy_helper.from_array(np.ones(size,  dtype=np.float32), name=f"{name}_scale")
    bias  = numpy_helper.from_array(np.zeros(size, dtype=np.float32), name=f"{name}_bias")
    mean  = numpy_helper.from_array(np.zeros(size, dtype=np.float32), name=f"{name}_mean")
    var   = numpy_helper.from_array(np.ones(size,  dtype=np.float32), name=f"{name}_var")
    return scale, bias, mean, var


def conv_bn_relu(
    input_name: str,
    output_name: str,
    prefix: str,
    in_ch: int,
    out_ch: int,
    kH: int = 3,
    kW: int = 3,
    stride: int = 1,
    pad: int = 1,
) -> tuple[list, list]:
    """Returns (nodes, initializers) for a Conv → BN → ReLU block."""
    conv_w  = make_conv_weights(f"{prefix}_w",   out_ch, in_ch, kH, kW)
    conv_b  = make_bias(f"{prefix}_b", out_ch)
    bn_s, bn_b, bn_m, bn_v = make_bn_params(prefix, out_ch)

    conv_out = f"{prefix}_conv_out"
    bn_out   = f"{prefix}_bn_out"

    conv_node = helper.make_node(
        "Conv",
        inputs=[input_name, f"{prefix}_w", f"{prefix}_b"],
        outputs=[conv_out],
        pads=[pad, pad, pad, pad],
        strides=[stride, stride],
        name=f"{prefix}_conv",
    )
    bn_node = helper.make_node(
        "BatchNormalization",
        inputs=[conv_out, f"{prefix}_scale", f"{prefix}_bias",
                f"{prefix}_mean", f"{prefix}_var"],
        outputs=[bn_out],
        epsilon=1e-5,
        name=f"{prefix}_bn",
    )
    relu_node = helper.make_node(
        "Relu",
        inputs=[bn_out],
        outputs=[output_name],
        name=f"{prefix}_relu",
    )

    nodes = [conv_node, bn_node, relu_node]
    inits = [conv_w, conv_b, bn_s, bn_b, bn_m, bn_v]
    return nodes, inits


def build_ds_cnn() -> onnx.ModelProto:
    """
    DS-CNN-S style model for keyword spotting.
    Input:  [1, 1, 49, 10]  — 49 time frames, 10 MFCC features
    Output: [1, 10]         — 10 keyword classes (yes/no/up/down/left/right/on/off/stop/go)
    """
    all_nodes = []
    all_inits = []

    # ── Block 1: standard Conv 3x3, 1→32 channels ──────────────────────────
    nodes, inits = conv_bn_relu("input", "block1_out", "b1", 1, 32, 3, 3, stride=2, pad=1)
    all_nodes += nodes; all_inits += inits

    # ── Block 2: depthwise Conv 3x3 (groups=32) ────────────────────────────
    dw2_w = numpy_helper.from_array(
        (np.random.randn(32, 1, 3, 3) * 0.1).astype(np.float32), name="dw2_w"
    )
    dw2_b = make_bias("dw2_b", 32)
    bn2_s, bn2_b, bn2_m, bn2_v = make_bn_params("bn2", 32)

    all_inits += [dw2_w, dw2_b, bn2_s, bn2_b, bn2_m, bn2_v]
    all_nodes += [
        helper.make_node("Conv", ["block1_out","dw2_w","dw2_b"], ["dw2_conv"],
                         pads=[1,1,1,1], group=32, name="dw2_conv"),
        helper.make_node("BatchNormalization",
                         ["dw2_conv","bn2_scale","bn2_bias","bn2_mean","bn2_var"],
                         ["dw2_bn"], name="dw2_bn"),
        helper.make_node("Relu", ["dw2_bn"], ["dw2_out"], name="dw2_relu"),
    ]

    # ── Block 2 pointwise: 32→64 ────────────────────────────────────────────
    nodes, inits = conv_bn_relu("dw2_out", "pw2_out", "pw2", 32, 64, 1, 1, pad=0)
    all_nodes += nodes; all_inits += inits

    # ── Block 3: depthwise Conv 3x3 (groups=64) ────────────────────────────
    dw3_w = numpy_helper.from_array(
        (np.random.randn(64, 1, 3, 3) * 0.1).astype(np.float32), name="dw3_w"
    )
    dw3_b = make_bias("dw3_b", 64)
    bn3_s, bn3_b, bn3_m, bn3_v = make_bn_params("bn3", 64)

    all_inits += [dw3_w, dw3_b, bn3_s, bn3_b, bn3_m, bn3_v]
    all_nodes += [
        helper.make_node("Conv", ["pw2_out","dw3_w","dw3_b"], ["dw3_conv"],
                         pads=[1,1,1,1], group=64, name="dw3_conv"),
        helper.make_node("BatchNormalization",
                         ["dw3_conv","bn3_scale","bn3_bias","bn3_mean","bn3_var"],
                         ["dw3_bn"], name="dw3_bn"),
        helper.make_node("Relu", ["dw3_bn"], ["dw3_out"], name="dw3_relu"),
    ]

    # ── Block 3 pointwise: 64→64 ────────────────────────────────────────────
    nodes, inits = conv_bn_relu("dw3_out", "pw3_out", "pw3", 64, 64, 1, 1, pad=0)
    all_nodes += nodes; all_inits += inits

    # ── Global Average Pooling ───────────────────────────────────────────────
    all_nodes.append(
        helper.make_node("GlobalAveragePool", ["pw3_out"], ["gap_out"], name="gap")
    )

    # ── Flatten ─────────────────────────────────────────────────────────────
    shape_init = numpy_helper.from_array(
        np.array([1, -1], dtype=np.int64), name="flatten_shape"
    )
    all_inits.append(shape_init)
    all_nodes.append(
        helper.make_node("Reshape", ["gap_out", "flatten_shape"], ["flat_out"], name="flatten")
    )

    # ── Fully Connected → 10 classes ────────────────────────────────────────
    fc_w = numpy_helper.from_array(
        (np.random.randn(10, 64) * 0.1).astype(np.float32), name="fc_w"
    )
    fc_b = make_bias("fc_b", 10)
    all_inits += [fc_w, fc_b]
    all_nodes.append(
        helper.make_node("Gemm", ["flat_out", "fc_w", "fc_b"], ["logits"],
                         transB=1, name="fc")
    )

    # ── Softmax ─────────────────────────────────────────────────────────────
    all_nodes.append(
        helper.make_node("Softmax", ["logits"], ["output"], axis=1, name="softmax")
    )

    # ── Build graph ─────────────────────────────────────────────────────────
    graph = helper.make_graph(
        all_nodes,
        "ds_cnn_keyword_spotting",
        inputs=[helper.make_tensor_value_info("input",  TensorProto.FLOAT, [1, 1, 49, 10])],
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10])],
        initializer=all_inits,
    )

    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 8
    model.doc_string = (
        "DS-CNN keyword spotting model — generated by EdgeForge test script. "
        "Input: [1,1,49,10] MFCC features. Output: [1,10] class probabilities."
    )

    # Run shape inference so quantisation tools can read tensor types
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    return model


if __name__ == "__main__":
    import sys

    np.random.seed(42)
    print("Building DS-CNN keyword spotting model...")
    model = build_ds_cnn()

    out_path = "test_model.onnx"
    onnx.save(model, out_path)

    size_kb = __import__("os").path.getsize(out_path) / 1024
    n_nodes  = len(model.graph.node)
    n_params = sum(
        __import__("numpy").prod(i.dims)
        for i in model.graph.initializer
    )

    print(f"")
    print(f"  Saved:       {out_path}")
    print(f"  Size:        {size_kb:.1f} KB")
    print(f"  Nodes:       {n_nodes}")
    print(f"  Parameters:  {n_params:,}")
    print(f"  Input:       [1, 1, 49, 10]  (MFCC spectrogram)")
    print(f"  Output:      [1, 10]          (keyword class probabilities)")
    print(f"")
    print(f"Now run:")
    print(f"  edgeforge analyze  {out_path}")
    print(f"  edgeforge optimize {out_path} --mcu=stm32f407")
    print(f"  edgeforge analyze  test_model_opt.onnx")