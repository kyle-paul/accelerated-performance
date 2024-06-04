import torch
from networks.UNet_plusplus.UNet_plusplus import UNet_plusplus
from networks.RotCAtt.RotCAtt import RotCAtt
from networks.RotCAtt.config import get_config

# Intialize model and input
model = torch.load("TorchModels/model1.pth")
model = RotCAtt(config=get_config()).cuda()
model.eval()

input = torch.rand(3, 1, 128, 128).to(torch.float32).cuda()
output = model(input)

# Export model to ONNX
torch.onnx.export(model, input, "OnnxModels/onnx_model2.onnx",
                  input_names=["input"], output_names=["output"],
                  dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}})