import torch


model = torch.load('TorchModels\model1.pth')
scripted_model = torch.jit.script(model)
print(scripted_model)
scripted_model.save('ScriptedModels/unet_plusplus.pt')