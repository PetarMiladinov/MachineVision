import torch
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from vggt_hub.vggt.models import VGGT

# Instantiate VGGT model
model = VGGT.from_pretrained("facebook/VGGT-1B")
model.eval()

# Load and preprocess image
image_path = 'images/sample_chair.jpg'
image = Image.open(image_path).convert('RGB')
image = image.resize((518, 518))
image_np = np.array(image) / 255.0
image_np = image_np.transpose(2, 0, 1)  # HWC to CHW
image_tensor = torch.tensor(image_np, dtype=torch.float32).unsqueeze(0)  # [1, 3, H, W]

# VGGT expects [S, 3, H, W] or [B, S, 3, H, W]
image_tensor = image_tensor.unsqueeze(0)  # [1, 1, 3, H, W]

with torch.no_grad():
    predictions = model(image_tensor)

# Show depth map for verification
depth = predictions.get('depth')
if depth is not None:
    # depth: [B, S, H, W, 1]
    depth_map = depth.squeeze().cpu().numpy()  # [H, W]
    plt.imshow(depth_map, cmap='plasma')
    plt.title('Predicted Depth Map')
    plt.colorbar()
    plt.show()
    np.save('predicted_depth.npy', depth_map)
    plt.imsave('predicted_depth.png', depth_map, cmap='plasma')
else:
    print('No depth map returned by VGGT model.')

# Get reference object info from user
ref_length_pixels = int(input("Enter reference object length in pixels: "))
ref_length_cm = float(input("Enter reference object length in cm: "))
pixel_per_cm = ref_length_pixels / ref_length_cm

# Estimate furniture bounding box from depth map (simple thresholding)
if depth is not None:
    # Threshold depth to find foreground (furniture)
    mask = (depth_map < np.percentile(depth_map, 50)).astype(np.uint8)  # crude foreground mask
    indices = np.argwhere(mask)
    if indices.size == 0:
        print('No furniture detected in depth map.')
    else:
        ymin, xmin = indices.min(axis=0)
        ymax, xmax = indices.max(axis=0)
        width_px = xmax - xmin
        height_px = ymax - ymin
        width_cm = width_px / pixel_per_cm
        height_cm = height_px / pixel_per_cm
        # Estimate average depth in region
        furniture_depth = np.mean(depth_map[mask == 1])
        depth_cm = furniture_depth / pixel_per_cm
        volume = width_cm * height_cm * depth_cm
        print(f"Approximate furniture volume: {volume:.2f} cm³")
else:
    print('Cannot estimate volume without depth map.')