# Furniture Volume Estimation using VGGT

This project uses the VGGT deep learning model to predict depth maps and reconstruct 3D point maps from a single RGB image of furniture. The main goal is to estimate the real-world volume of furniture objects using the predicted depth and a reference object for scale calibration.

## How to Use
1. Place an image of furniture in the folder: `VGGT Furniture/MV/images/`  
   > **Note:** If running commands or scripts, ensure paths with spaces are properly quoted (e.g., `"VGGT Furniture/MV/images/"`).
2. Run `main_2.py` in your Python environment (see `requirements.txt` for dependencies).
3. When prompted, click the endpoints of your reference object in the image window, then enter its real-world length in meters.
4. The script will output the estimated volume for all objects (or a selected object if segmented).
5. Inspect the 3D point map visualization to verify the reconstruction and object coverage.

## Example Output

Predicted Depth Map:
![Predicted Depth Map](VGGT%20Furniture/MV/predicted_depth_main2.png)
See [`project_documentation.txt`](VGGT%20Furniture/MV/project_documentation.txt) or [`project_documentation.pdf`](VGGT%20Furniture/MV/project_documentation.pdf) for full workflow and instructions.
See [`project_documentation.pdf`](VGGT%20Furniture/MV/project_documentation.pdf) for full workflow and instructions.

## Files Included
- `main_2.py`: Main script for image processing, depth prediction, 3D point map extraction, and volume estimation.
- `requirements.txt`: Python dependencies for recreating the environment.
- `project_documentation.pdf`: Project documentation and workflow.
- `predicted_depth_main2.npy` and `predicted_depth_main2.png`: Example depth map outputs.
- `convert_doc_to_pdf.py`: Script to convert documentation to PDF.
- `images/`: Folder for input images.

## Notes
- Do NOT include the `vggt_env/` virtual environment folder in the repo.
- All dependencies are listed in `requirements.txt`.
- For single-object volume estimation, segment or mask the point map before running the convex hull calculation.

---
Author: Petar
Date: September 4, 2025
