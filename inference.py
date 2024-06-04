import onnx
import onnxruntime as ort
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import zoom

# Input and preprocessing
image = np.load('samples/0001_0170.npy')
img_size = 128
x, y = image.shape
if x != img_size and y != img_size:
    inputs = zoom(image, (img_size / x, img_size / y), order=0)
    
inputs = inputs[np.newaxis,np.newaxis,:,:]

# Inference
EP_list = [
    ('TensorrtExecutionProvider', {
        'device_id': 0,                       # Select GPU to execute
        'trt_max_workspace_size': 2147483648, # Set GPU memory usage limit
        'trt_fp16_enable': True,              # Enable FP16 precision for faster inference  
    }),
    
    'TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']

ort_session = ort.InferenceSession("onnx_model_sim.onnx", providers=EP_list)
inp = {ort_session.get_inputs()[0].name: inputs}
out = ort_session.run(None, inp)

out = np.argmax(out[0], axis=1)
plt.imshow(out[0])
plt.show()