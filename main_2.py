import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from vggt.models.vggt import VGGT
from scipy.spatial import ConvexHull

# Instantiate VGGT model
model = VGGT.from_pretrained("facebook/VGGT-1B")
model.eval()

def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((518, 518))
    image_np = np.array(image) / 255.0
    image_np = image_np.transpose(2, 0, 1)  # HWC to CHW
    image_tensor = torch.tensor(image_np, dtype=torch.float32).unsqueeze(0)  # [1, 3, H, W]
    image_tensor = image_tensor.unsqueeze(0)  # [1, 1, 3, H, W]
    return image_tensor

image_path = 'images/mebel.jpg'
image_tensor = preprocess_image(image_path)

with torch.no_grad():
    # Use VGGT heads to extract advanced outputs
    aggregated_tokens, ps_idx = model.aggregator(image_tensor)
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    pose_enc = model.camera_head(aggregated_tokens)[-1]
    extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, image_tensor.shape[-2:])
    depth_map, depth_conf = model.depth_head(aggregated_tokens, image_tensor, ps_idx)
    point_map = model.point_head(aggregated_tokens, image_tensor, ps_idx)

# Show depth map for verification
if depth_map is not None:
    depth_map_np = depth_map.squeeze().cpu().numpy()
    plt.imshow(depth_map_np, cmap='plasma')
    plt.title('Predicted Depth Map (Advanced)')
    plt.colorbar()
    plt.show()
    np.save('predicted_depth_main2.npy', depth_map_np)
    plt.imsave('predicted_depth_main2.png', depth_map_np, cmap='plasma')
else:
    print('No depth map returned by VGGT model.')

# Print camera parameters and point map
print('Extrinsic matrix (camera pose):')
print(extrinsic)
print('Intrinsic matrix (camera intrinsics):')
print(intrinsic)
print('Point map:')
print(point_map)

# Estimate volume using the point map via convex hull
# Extract 3D points from point_map (first tensor in tuple)
points_3d = point_map[0].squeeze().cpu().numpy()  # shape: [H, W, 3]
points_3d_flat = points_3d.reshape(-1, 3)

# Remove any NaNs or infs
points_3d_flat = points_3d_flat[np.isfinite(points_3d_flat).all(axis=1)]

if points_3d_flat.shape[0] >= 4:
    hull = ConvexHull(points_3d_flat)
    volume = hull.volume
    print(f"Convex hull volume estimate (in model units): {volume:.2f}")
    # Interactive reference selection for scale conversion
    import matplotlib.pyplot as plt
    image = Image.open(image_path).convert('RGB')
    plt.imshow(image)
    plt.title('Click TOP and BOTTOM (or endpoints) of your reference object in the image window.')
    print('Please click the TOP and BOTTOM (or endpoints) of your reference object in the image window.')
    pts = plt.ginput(2)
    plt.close()
    height, width = points_3d.shape[:2]
    x_ref1 = min(max(int(pts[0][0]), 0), width - 1)
    y_ref1 = min(max(int(pts[0][1]), 0), height - 1)
    x_ref2 = min(max(int(pts[1][0]), 0), width - 1)
    y_ref2 = min(max(int(pts[1][1]), 0), height - 1)
    # Get corresponding 3D points from point map
    ref_3d_1 = points_3d[y_ref1, x_ref1]
    ref_3d_2 = points_3d[y_ref2, x_ref2]
    model_length = np.linalg.norm(ref_3d_1 - ref_3d_2)
    # Prompt user for real-world length
    real_world_length = float(input('Enter the real-world length (in meters) of the reference object: '))
    scale = real_world_length / model_length
    volume_m3 = volume * (scale ** 3)
    print(f"Estimated object volume in m³: {volume_m3:.4f}")
else:
    print("Not enough valid 3D points for convex hull volume estimation.")

# Visualize the 3D point map as a scatter plot
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
valid_points = points_3d_flat[np.isfinite(points_3d_flat).all(axis=1)]
ax.scatter(valid_points[:, 0], valid_points[:, 1], valid_points[:, 2], s=0.5, c=valid_points[:, 2], cmap='plasma')
ax.set_title('3D Point Map')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

# You can further process camera_params, point_map, and point_tracks for advanced 3D reconstruction and volume estimation.
