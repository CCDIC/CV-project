CV-project: Computer Vision Projects
A computer vision project repository containing implementations of various computer vision algorithmsns, including feature detection, corner detection, blob detection, and image filtering techniques.
Project Structure
plaintext
CV-project/
├── common.py               # Common utility functions for image processing
├── corners.py              # Corner detection implementations (Harris detector, etc.)
├── blob_detection.py       # Blob detection using Difference of Gaussians (DoG)
├── filters.py              # Image filtering operations (convolution, edge detection, etc.)
├── feature_detection/      # Output directory for feature detection results
├── gaussian_filter/        # Output directory for Gaussian filter results
├── log_filter/             # Output directory for Laplacian of Gaussian results
├── sobel_operator/         # Output directory for Sobel operator results
├── image_patches/          # Output directory for image patch results
├── polka_detections/       # Output directory for polka dot detection results
└── cell_detections/        # Output directory for cell detection results
Key Features
1. Image Processing Utilities (common.py)
Image reading/writing with grayscale conversion
Scale space visualization
Maxima detection in scale space
Visualization of detected features with overlay
2. Corner Detection (corners.py)
Implementation of corner score calculation
Harris corner detector
Visualization of corner responses with heatmaps
3. Blob Detection (blob_detection.py)
Difference of Gaussians (DoG) for blob detection
Multi-scale blob detection
Cell counting application using blob detection
4. Image Filtering (filters.py)
Image patch extraction
Convolution operations
Edge detection
Sobel operator
Laplacian of Gaussian (LoG) filtering
Usage
Clone the repository:
bash
git clone https://github.com/CCDIC/CV-project.git
cd CV-project
Run individual components:
bash
# Run corner detection
python corners.py

# Run blob detection
python blob_detection.py

# Run filtering operations
python filters.py
Check the corresponding output directories for results.
Dependencies
NumPy
SciPy
PIL (Pillow)
Matplotlib
Notes
The project uses various image filtering techniques to detect features in images
Results are saved in corresponding output directories
Parameters can be adjusted in the respective scripts for different detection behaviors
This repository provides practical implementations of fundamental computer vision algorithms, serving as a good reference for learning and experimenting with image processing techniques.
