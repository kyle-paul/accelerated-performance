from onnxconverter_common import auto_mixed_precision
import onnx

model = onnx.load("OnnxModels/onnx_model_sim2.onnx")
model_fp16 = auto_convert_mixed_precision(model, test_data, rtol=0.01, atol=0.001, keep_io_types=True)
onnx.save(model_fp16, "OnnxModels/onnx_model_sim2_float16.onnx")
