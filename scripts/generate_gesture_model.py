"""
generate_gesture_model_v2.py
Gesture model without BatchNormalization -- converts cleanly to TFLite.
BN is replaced with bias terms directly in each Conv layer.
Input:  [1, 1, 50, 3]  -- 50 accelerometer samples x (ax, ay, az)
Output: [1, 3]          -- idle / shake / tap
"""
import numpy as np
import onnx
import onnx.shape_inference
from onnx import helper, TensorProto, numpy_helper

np.random.seed(42)

def w(name, shape, scale=0.05):
    return numpy_helper.from_array(
        (np.random.randn(*shape) * scale).astype(np.float32), name=name)

def b(name, size):
    return numpy_helper.from_array(np.zeros(size, dtype=np.float32), name=name)

nodes, inits = [], []

# Block 1: Conv 3x3, 1->16, stride 2, with bias (no BN)
inits += [w("b1_w", (16,1,3,3)), b("b1_b", 16)]
nodes += [
    helper.make_node("Conv", ["X","b1_w","b1_b"], ["b1_c"],
                     pads=[1,1,1,1], strides=[2,2]),
    helper.make_node("Relu", ["b1_c"], ["b1"]),
]

# Block 2: Depthwise 3x3 groups=16
inits += [w("dw2_w", (16,1,3,3)), b("dw2_b", 16)]
nodes += [
    helper.make_node("Conv", ["b1","dw2_w","dw2_b"], ["dw2_c"],
                     pads=[1,1,1,1], group=16),
    helper.make_node("Relu", ["dw2_c"], ["dw2"]),
]

# Pointwise 1x1, 16->32
inits += [w("pw2_w", (32,16,1,1)), b("pw2_b", 32)]
nodes += [
    helper.make_node("Conv", ["dw2","pw2_w","pw2_b"], ["pw2_c"], pads=[0,0,0,0]),
    helper.make_node("Relu", ["pw2_c"], ["pw2"]),
]

# Block 3: Depthwise 3x3 groups=32
inits += [w("dw3_w", (32,1,3,3)), b("dw3_b", 32)]
nodes += [
    helper.make_node("Conv", ["pw2","dw3_w","dw3_b"], ["dw3_c"],
                     pads=[1,1,1,1], group=32),
    helper.make_node("Relu", ["dw3_c"], ["dw3"]),
]

# Pointwise 1x1, 32->32
inits += [w("pw3_w", (32,32,1,1)), b("pw3_b", 32)]
nodes += [
    helper.make_node("Conv", ["dw3","pw3_w","pw3_b"], ["pw3_c"], pads=[0,0,0,0]),
    helper.make_node("Relu", ["pw3_c"], ["pw3"]),
]

# Global Average Pool
nodes.append(helper.make_node("GlobalAveragePool", ["pw3"], ["gap"]))

# Flatten
sh = numpy_helper.from_array(np.array([1,-1], dtype=np.int64), name="sh")
inits.append(sh)
nodes.append(helper.make_node("Reshape", ["gap","sh"], ["flat"]))

# FC 32->3
inits += [w("fc_w", (3,32)), b("fc_b", 3)]
nodes.append(helper.make_node("Gemm", ["flat","fc_w","fc_b"], ["logits"], transB=1))

# Softmax
nodes.append(helper.make_node("Softmax", ["logits"], ["output"], axis=1))

graph = helper.make_graph(
    nodes, "gesture_v2",
    [helper.make_tensor_value_info("X",      TensorProto.FLOAT, [1,1,50,3])],
    [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1,3])],
    initializer=inits,
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("",17)])
model.ir_version = 8
model = onnx.shape_inference.infer_shapes(model)
onnx.checker.check_model(model)
onnx.save(model, "gesture_model_v2.onnx")

n_params = sum(np.prod(i.dims) for i in model.graph.initializer)
print(f"gesture_model_v2.onnx saved")
print(f"  Nodes:      {len(model.graph.node)}")
print(f"  Parameters: {n_params:,}")
print(f"  Input:      [1, 1, 50, 3]")
print(f"  Output:     [1, 3] -- idle/shake/tap")
print(f"  No BatchNorm -- converts cleanly to TFLite")
