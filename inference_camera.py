import onnx
import onnxruntime
import numpy as np
import cv2
import os
import time

# Path to the ONNX model and input video
# onnx_model_path = 'assets/models/DA_quant_DYN.onnx'
# onnx_model_path = 'assets/models/DA_quant_STAT3.onnx'
# onnx_model_path = 'assets/models/depth_anything_v2.onnx'
onnx_model_path = 'assets/models/MiDaS_small.onnx'
# onnx_model_path = 'assets/models/MiDaS_quantized_dyn.onnx'
# onnx_model_path = 'assets/models/MiDaS_quant_STAT1.onnx'
video_path = 'assets/examples_video/ferris_wheel.mp4'
# video_path = 'assets/examples_video/basketball.mp4'

# Load the ONNX model
session = onnxruntime.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])

# model = onnx.load(onnx_model_path)
# print(onnx.helper.printable_graph(model.graph))

# Open video capture
# Uncomment the line below to use a video file instead of a webcam feed
# cap = cv2.VideoCapture(video_path)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties (for FPS calculation)
fps = cap.get(cv2.CAP_PROP_FPS)

while True:
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        break  # Break the loop if no more frames are available

    # Start the timer for FPS calculation
    start_time = time.time()

    # Step 1: Preprocess the frame (resize and normalize)
    input_size = (256, 256)  # Modify based on your model's expected input size
    frame_resized = cv2.resize(frame, input_size)

    # Convert the frame to a float32 array and normalize (adjust normalization based on your model's needs)
    frame_normalized = frame_resized.astype(np.float32) / 255.0  # Normalize between 0 and 1
    frame_transposed = np.transpose(frame_normalized, (2, 0, 1))  # Change to (C, H, W) format
    frame_input = np.expand_dims(frame_transposed, axis=0)  # Add batch dimension

    # Step 2: Run inference
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    output = session.run([output_name], {input_name: frame_input})

    # Step 3: Post-process the output (depth map)
    depth_map = output[0][0]  # Access the first output tensor and the first element in the batch
    depth_map = np.squeeze(depth_map)  # Remove extra dimensions if necessary

    # Normalize the depth map to 0-255 for visualization
    depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_map_normalized = np.uint8(depth_map_normalized)

    # Apply a color map to the depth map to create a colorized version
    depth_map_colored = cv2.applyColorMap(depth_map_normalized, cv2.COLORMAP_JET)

    # Resize the original frame to match the depth map size (518x518) for visualization
    frame_resized_for_display = cv2.resize(frame, (256, 256))

    # Stack the resized original frame and the colorized depth map side by side
    combined_frame = np.hstack((frame_resized_for_display, depth_map_colored))

    # Calculate FPS
    end_time = time.time()
    fps_display = int(1 / (end_time - start_time))

    # Display the combined frame with FPS
    cv2.putText(combined_frame, f"FPS: {fps_display}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Original + Depth Map', combined_frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release the video capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()