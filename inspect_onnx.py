import onnx
import onnxruntime as ort

# Load the ONNX model
model_path = 'ckpt/isnetis.onnx'
onnx_model = onnx.load(model_path)

# Print the model
print(onnx.helper.printable_graph(onnx_model.graph))

# Optionally, check the model
onnx.checker.check_model(onnx_model)

# To print input and output names and shapes using ONNX Runtime
session = ort.InferenceSession(model_path)
print("Inputs:")
for input in session.get_inputs():
    print(input.name, input.shape, input.type)

print("Outputs:")
for output in session.get_outputs():
    print(output.name, output.shape, output.type)
