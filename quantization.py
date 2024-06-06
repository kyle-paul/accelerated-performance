import torch
import torch.quantization

model = torch.load('TorchModels/model1.pth')
model.eval()

quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

torch.save(quantized_model, 'TorchModels/quantized_model1.pth')