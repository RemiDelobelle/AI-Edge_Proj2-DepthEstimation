# AI Edge Computing Project - Depth Estimation

## Overview
This project was developed as part of the AI Edge Computing course. The primary focus is on researching the possibilities for EMVI, a company providing solutions for visually impaired individuals, to implement depth estimation models on edge devices, particularly the Raspberry Pi 5. The goal was to explore the feasibility of running a depth estimation model efficiently and accurately on edge hardware, optimizing for real-time performance.

The paper discusses the methodology, results, and conclusions of this project in detail.

## Goals
The primary objectives of this project are:

1. **Evaluate Depth Estimation Models:** Research and test depth estimation models to determine their feasibility for real-time execution on an edge device, specifically Raspberry Pi 5.
2. **Optimize Model Performance:** Explore methods for improving the model's performance on edge devices, including quantization techniques and using execution providers.

## Repository Structure
```
.
├── convert_to_onnx.py          # Converts a .pt file to ONNX format
├── inference_camera.py         # Runs inference using live camera feed or video
├── inference_img.py            # Runs inference on static images
├── quant_DYNAMIC.py            # Quantizes ONNX model using dynamic method
├── quant_STATIC.py             # Quantizes ONNX model using static method
├── assets/
│   ├── examples/               # Folder containing test images
│   ├── examples_video/         # Folder containing test videos
│   ├── models/                 # Folder containing all used models
│   └── vis_depth/              # Folder for saving output images
```

## Getting Started
### Prerequisites
- Python 3.8+
- Required Python packages (install using `pip install -r requirements.txt`):
    - numpy
    - opencv-python
    - torch
    - onnx
    - onnxruntime
- A Raspberry Pi 5 or a local machine for testing.

### Setup
1. Clone this repository:
   ```bash
   git clone https://github.com/RemiDelobelle/AI-Edge_Proj2-DepthEstimation
   ```
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your Raspberry Pi 5 or local machine for testing.

## Usage
To run any of the scripts, just execute them directly. You may need to adjust paths to models or input files within the code (e.g., specify the correct path to the ONNX models inside the script). 

## Conclusion
This project demonstrates the feasibility of running depth estimation models on edge devices like Raspberry Pi 5. The use of quantization and execution providers significantly improved performance, making real-time assistance for visually disabled individuals a step closer to reality.
