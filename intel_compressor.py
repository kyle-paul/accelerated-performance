import torch
import onnx
from neural_compressor import mix_precision
from neural_compressor.config import MixedPrecisionConfig


# model = onnx.load("OnnxModels/onnx_model_sim2.onnx")

# conf = MixedPrecisionConfig(
#     backend="onnxrt_cuda_ep",
#     device="gpu",
#     precisions="fp16",
# )
# converted_model = mix_precision.fit(model, conf=conf)
# converted_model.save("OnnxModels/onnx_model_sim_compressed2.onnx")

conf = MixedPrecisionConfig(
    backend="ipex",
    device="cpu",
    precisions="fp16",
) 
torch_model = torch.load("TorchModels/model1.pth")
converted_model = mix_precision.fit(torch_model, conf=conf)