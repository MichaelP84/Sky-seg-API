from onnxruntime.quantization import quantize_dynamic, QuantType

# Define the path to your original ONNX model
input_model_path = 'ckpt/isnetis.onnx'
# Define the path where the quantized model will be saved
output_model_path = 'ckpt/isnetis_quantized.onnx'
# Perform dynamic quantization
quantize_dynamic(model_input=input_model_path,
                 model_output=output_model_path,
                 weight_type=QuantType.QUInt8)  # Use QUInt8 or QInt8 based on your preference

print("Quantization complete! Model was saved to:", output_model_path)
