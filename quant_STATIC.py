import os
import cv2
import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_static, QuantType
from onnxruntime.quantization import CalibrationDataReader
from onnxruntime.quantization import QuantFormat, quantize_static, CalibrationMethod
from onnx.helper import make_tensor_value_info, set_model_props

class CalibrationImageReader(CalibrationDataReader):
    def __init__(self, image_folder, input_size=(256, 256)):
        """
        Args:
            image_folder (str): Path to the folder containing calibration images.
            input_size (tuple): Size to which input images should be resized.
        """
        self.image_folder = image_folder
        self.input_size = input_size
        self.image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.jpg')]
        self.index = 0
    
    def preprocess_image(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Ensure correct color format
        image_resized = cv2.resize(image, (256, 256))  # Resize to match model input
        image_normalized = image_resized.astype(np.float32) / 255.0  # Normalize to [0, 1]
        image_normalized = np.transpose(image_normalized, (2, 0, 1))  # HWC -> CHW
        image_batch = np.expand_dims(image_normalized, axis=0)  # Add batch dimension
        print(f"Processed Image Shape: {image_batch.shape}")
        print(f"Pixel Value Range: Min={image_batch.min()}, Max={image_batch.max()}")
        return image_batch


    def get_next(self):
        """Return the next preprocessed image from the calibration folder."""
        if self.index >= len(self.image_paths):
            return None  # No more images
        
        # Preprocess the next image
        image_path = self.image_paths[self.index]
        self.index += 1
        return {self.input_name: self.preprocess_image(image_path)}

    def rewind(self):
        """Reset the reader to start from the first image."""
        self.index = 0


# Function to modify input and output tensors to int8
def modify_model_io_to_int8(model_path, output_path, input_scale, input_zero_point, output_scale, output_zero_point):
    model = onnx.load(model_path)
    graph = model.graph

    # Modify input tensor
    for input_tensor in graph.input:
        input_tensor.type.tensor_type.elem_type = onnx.TensorProto.INT8
        # Add quantization information for input
        quant_param_metadata = f"scale:{input_scale},zero_point:{input_zero_point}"
        set_model_props(model, {f"quantization.input.{input_tensor.name}": quant_param_metadata})

    # Modify output tensor
    for output_tensor in graph.output:
        output_tensor.type.tensor_type.elem_type = onnx.TensorProto.INT8
        # Add quantization information for output
        quant_param_metadata = f"scale:{output_scale},zero_point:{output_zero_point}"
        set_model_props(model, {f"quantization.output.{output_tensor.name}": quant_param_metadata})

    # Save the updated model
    onnx.save(model, output_path)
    print(f"Modified model input and output tensors to INT8 and saved to {output_path}")


# Set paths for your model and calibration dataset
# model_fp32_path = 'depth_anything_v2.onnx'  # Path to your FP32 model
model_fp32_path = 'assets/models/MiDaS_small.onnx'
model_quantized_path = 'assets/models/MiDaS_quant_STAT1.onnx'  # Path to save the quantized model
calibration_folder = 'assets/examples/'  # Path to your calibration image folder
input_size = (256, 256)  # Model's expected input size (height, width)

# Load the ONNX model to get the input name
onnx_model = onnx.load(model_fp32_path)
session = ort.InferenceSession(model_fp32_path, providers=ort.get_available_providers())
input_name = session.get_inputs()[0].name  # Assume single input for simplicity

# Create the calibration data reader
calibration_data_reader = CalibrationImageReader(calibration_folder, input_size=input_size)
calibration_data_reader.input_name = input_name  # Pass the model's input name to the reader

# Inspect the input shape of the model
print("Model Input Details:")
for input_tensor in session.get_inputs():
    print(f"Name: {input_tensor.name}, Shape: {input_tensor.shape}, Type: {input_tensor.type}")


# Perform static quantization
quantize_static(
    model_input=model_fp32_path,
    model_output=model_quantized_path,
    calibration_data_reader=calibration_data_reader,
    quant_format=QuantFormat.QOperator,
    op_types_to_quantize=["Conv", "MatMul"],  # Limit to supported ops
    activation_type=QuantType.QInt8,  # Quantize activations to INT8
    weight_type=QuantType.QInt8,      # Quantize weights to INT8
    per_channel=True  # Use per-channel quantization if needed for better accuracy
)