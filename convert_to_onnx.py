import torch
import torch.onnx
from depth_anything_v2.dpt import DepthAnythingV2
encoder = 'vits'  # Choose from ['vits', 'vitb', 'vitl', 'vitg']
model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }


depth_anything = DepthAnythingV2(**model_configs[encoder])
depth_anything.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
depth_anything = depth_anything.eval()

# Input sample (image tensor)
input_size = 518
dummy_input = torch.randn(1, 3, input_size, input_size)  # Create a dummy input

# Export model to ONNX format
onnx_path = './depth_anything_v2.onnx'
torch.onnx.export(depth_anything, dummy_input, onnx_path, opset_version=11)
