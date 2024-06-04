from onnxconverter_common import auto_mixed_precision
import onnx
import onnxruntime as ort

import numpy as np
from scipy.ndimage import zoom

def test_data():
    image = np.load('samples/0001_0170.npy')
    img_size = 128
    x, y = image.shape
    if x != img_size and y != img_size:
        inputs = zoom(image, (img_size / x, img_size / y), order=0)
        
    inputs = inputs[np.newaxis, np.newaxis, :, :]
    inputs = np.float32(inputs)
    provider = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    ort_session = ort.InferenceSession("OnnxModels/onnx_model2.onnx", providers=provider)
    inp = {ort_session.get_inputs()[0].name: inputs}
    return inp

model = onnx.load("OnnxModels/onnx_model2.onnx")
model_fp16 = auto_mixed_precision.auto_convert_mixed_precision(model, test_data(), rtol=0.01, atol=0.001, keep_io_types=True)
onnx.save(model_fp16, "OnnxModels/onnx_model_sim2_float16.onnx")