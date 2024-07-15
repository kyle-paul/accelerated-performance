import torch
import openvino.torch


model = torch.jit.load('ScriptedModels/resunet.pt')
model = torch.compile(
    model, backend="openvino", 
    options = {
        "device" : "GPU.0", 
        "model_caching" : True, 
        "cache_dir": "./model_cache",
        "PERFORMANCE_HINT" : "LATENCY"
    })