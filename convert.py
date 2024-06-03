import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np
from time import time
from network import UNet_plusplus

# Intialize model and input
model = UNet_plusplus(num_classes=8).cuda()
model.eval()
input = torch.rand(3, 1, 128, 128).to(torch.float32).cuda()

# Export model to ONNX
torch.onnx.export(model, input, "onnx_model.onnx",
                  input_names=["input"], output_names=["output"],
                  dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}})

# Load and verify the ONNX model
onnx_model = onnx.load("onnx_model.onnx")
onnx.checker.check_model(onnx_model)
# print(onnx.helper.printable_graph(onnx_model.graph))

# # Run inference with ONNX runtime
ort_session = ort.InferenceSession("onnx_model.onnx")

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# # Generate input for sequence
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(input)}
start = time()
ort_outs = ort_session.run(None, ort_inputs)
end = time()

print(f"ONNX Runtime output: {len(ort_outs)} - {ort_outs[0].shape} - {end-start}")
