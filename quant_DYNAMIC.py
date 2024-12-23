from onnxruntime.quantization import quantize_dynamic, QuantType

# model_fp32 = 'depth_anything_v2.onnx'
# model_quan_dyn = 'DA_quantized_dyn.onnx'
model_fp32 = 'MiDaS_small.onnx'
model_quan_dyn = 'MiDaS_quantized_dyn.onnx'

quantized_model_dyn = quantize_dynamic(
    model_fp32,
    model_quan_dyn,
    weight_type=QuantType.QUInt8,
)

print("Dynamic quantization completed and saved to:", model_quan_dyn)