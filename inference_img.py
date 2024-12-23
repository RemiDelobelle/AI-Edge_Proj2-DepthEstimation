import onnx
import onnxruntime
import numpy as np
import cv2
import os

# Path to the ONNX model and input image
# onnx_model_path = 'depth_anything_v2.onnx'
onnx_model_path = 'depth_anything_quantized_static.onnx'
output_dir = 'assets/vis_depth'
image_path = 'assets/examples/demo09.jpg'

# Load the ONNX model
session = onnxruntime.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])

model = onnx.load(onnx_model_path)
print(onnx.helper.printable_graph(model.graph))


# Load the input image
image = cv2.imread(image_path)

# Step 1: Preprocess the image
# Resize to the input size expected by the model (example: 224x224)
input_size = (518, 518)  # Modify based on your model's expected input size
image_resized = cv2.resize(image, input_size)

# Convert the image to a float32 array and normalize (adjust normalization based on your model's needs)
image_normalized = image_resized.astype(np.float32) / 255.0  # Normalize between 0 and 1
image_transposed = np.transpose(image_normalized, (2, 0, 1))  # Change to (C, H, W) format
image_input = np.expand_dims(image_transposed, axis=0)  # Add batch dimension

# Step 2: Run inference
# Get the input name and output name from the ONNX model
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Run the model with the preprocessed image
output = session.run([output_name], {input_name: image_input})

# Step 3: Post-process the output (depends on the model)
# For simplicity, let's assume the output is a single-channel depth map
output_image = output[0][0]  # Access the first output tensor, and the first element in the batch
output_image = np.squeeze(output_image)  # Remove extra dimensions if necessary

# Step 4: Save the output
output_path = os.path.join(output_dir, 'output_image_quan_INT.png')  # Path to save the output

# Normalize the output for visualization (if it's a depth map or image)
output_image_normalized = cv2.normalize(output_image, None, 0, 255, cv2.NORM_MINMAX)
output_image_normalized = np.uint8(output_image_normalized)

# Save the output image
cv2.imwrite(output_path, output_image_normalized)

print(f"Inference completed! Output saved to: {output_path}")
