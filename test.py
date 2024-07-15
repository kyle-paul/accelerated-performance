import torch
import onnxruntime as ort
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import zoom
from networks.UNet_plusplus.UNet_plusplus import UNet_plusplus
from tqdm import tqdm
import SimpleITK as sitk
import os
from skimage.transform import resize
import nibabel as nib



def save_vol(vol, type, path):
    vol = np.transpose(vol, (2, 1, 0))
    affine = np.eye(4)
    nifti_file = nib.Nifti1Image(vol.astype(np.int8), affine) if type=='labels' else nib.Nifti1Image(vol, affine)
    nib.save(nifti_file, path)  

# # Input and preprocessing
# img_size = 128
# vol = []
# for index in range(166, 178, 1):
#     slice = np.load(f'samples/0001_{index:04d}.npy')
#     x, y = slice.shape
#     if x != img_size and y != img_size:
#         slice = zoom(slice, (img_size / x, img_size / y), order=0)
#     vol.append(slice)

# vol = np.array(vol)
# inputs = np.array(vol)[:, np.newaxis, :, :]
# # inputs = np.float16(inputs)
def preprocessing(data):
    max_data = np.max(data)
    min_data = np.min(data)    
    data = (data - min_data) / (max_data - min_data)
    data = data[:12, np.newaxis, :, :]
    return data
    
data = sitk.GetArrayFromImage(sitk.ReadImage("samples/MMWHS.nii.gz"))
data = resize(data, (data.shape[0], 128, 128), mode='reflect', anti_aliasing=True)
inputs = preprocessing(data)


print(inputs.shape)

# Inference
providers = [
    ('TensorrtExecutionProvider', {
        'device_id': 0,                       # Select GPU to execute
        'trt_max_workspace_size': 2147483648, # Set GPU memory usage limit
        'trt_fp16_enable': True,              # Enable FP16 precision for faster inference  
        'trt_engine_cache_enable': True,
        'trt_engine_cache_path': 'Engine/onnx_model_sim_engine_2',
    }),
    ('CUDAExecutionProvider', {
    })
]

ort_session = ort.InferenceSession("OnnxModels/onnx_model_sim2.onnx", providers=providers)
inp = {ort_session.get_inputs()[0].name: inputs}
out = ort_session.run(None, inp)
print(out[0].shape)
out = np.argmax(out[0], axis=1)
save_vol(out, "labels", "temp.nii.gz")
  


# # warmup
# print("warming up")
# inp = {ort_session.get_inputs()[0].name: inputs}
# out = ort_session.run(None, inp)


# for i in tqdm(range(1000)):
#     inp = {ort_session.get_inputs()[0].name: inputs}
#     out = ort_session.run(None, inp)


# out = np.argmax(out[0], axis=1)
# print(out.shape)