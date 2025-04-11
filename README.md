# OpenOpticalFlow

OpenOpticalFlow is an open source package for extraction of high-resolution velocity fields from various flow visualization images. Originally developed in MATLAB, it is now available as a Python package.

## Overview

OpenOpticalFlow implements optical flow algorithms, particularly focusing on the Horn-Schunck and Liu-Shen methods for estimating fluid motion from image sequences. The package provides tools for:

- Preprocessing images for optical flow analysis
- Estimating optical flow using the Horn-Schunck method
- Refining optical flow estimates using the physics-based Liu-Shen method
- Calculating vorticity and other flow diagnostics
- Visualizing flow fields

For more details, see the [original paper](https://github.com/alexlib/OpenOpticalFlow/blob/master/Open_Optical_Flow_Paper_v1.pdf).

## Installation

### Prerequisites

- Python 3.7 or higher
- NumPy
- SciPy
- Matplotlib
- scikit-image

### Installing from Source

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/OpenOpticalFlow.git
   cd OpenOpticalFlow
   ```

2. Install the package in development mode:
   ```bash
   pip install -e .
   ```

   This will install the package in development mode, allowing you to modify the code and have the changes immediately available.

3. Alternatively, install with all development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

## Usage

Here's a basic example of how to use the package:

```python
import numpy as np
from openopticalflow.optical_flow_physics import optical_flow_physics
from openopticalflow.vis_flow import vis_flow
import matplotlib.pyplot as plt

# Load two consecutive images
image1 = ...  # Load your first image
image2 = ...  # Load your second image

# Calculate optical flow
u, v, vorticity, ux_horn, uy_horn, error = optical_flow_physics(
    image1, image2, lambda_1=10.0, lambda_2=100.0
)

# Visualize the flow field
plt.figure(figsize=(10, 8))
ax = vis_flow(v, u, gx=20, offset=0, mag=2, color='red')
plt.title('Optical Flow Field')
plt.show()

# Visualize the vorticity
plt.figure(figsize=(10, 8))
plt.imshow(vorticity, cmap='RdBu_r')
plt.colorbar(label='Vorticity')
plt.title('Vorticity Field')
plt.show()
```

## Testing

The package includes a comprehensive test suite using pytest. To run the tests:

1. Make sure you have the development dependencies installed:
   ```bash
   pip install -e ".[dev]"
   ```

2. Run all tests:
   ```bash
   python -m pytest
   ```

3. Run tests with coverage information:
   ```bash
   python -m pytest --cov=openopticalflow
   ```

4. Run a specific test file:
   ```bash
   python -m pytest tests/test_optical_flow_synthetic.py -v
   ```

### Test Suite Overview

The test suite includes:

- Unit tests for individual components (Horn-Schunck estimator, vorticity calculation, etc.)
- Integration tests for the complete optical flow pipeline
- Synthetic flow tests with known ground truth (translation and dipole patterns)
- Visualization tests

## Coordinate System

The package uses the following coordinate system conventions:

- Image origin (0,0) is at the top-left corner
- The y-axis increases downward
- The x-axis increases to the right
- In the optical flow output, `u` is the y-component (vertical) and `v` is the x-component (horizontal)

## License

[MIT License](LICENSE)

## Acknowledgments

This package is based on the optical flow algorithms developed by Horn & Schunck and Liu & Shen.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
