"""
Test file demonstrating the use of example images with the optical flow algorithms.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import pytest

from openopticalflow.optical_flow_physics import optical_flow_physics
from openopticalflow.vis_flow import vis_flow


def test_vortex_pair_basic():
    """Test basic optical flow on vortex pair images."""
    # Check if images exist
    img1_path = os.path.join('img', 'vortex_pair_particles_1.tif')
    img2_path = os.path.join('img', 'vortex_pair_particles_2.tif')
    
    assert os.path.exists(img1_path), f"Image file not found: {img1_path}"
    assert os.path.exists(img2_path), f"Image file not found: {img2_path}"
    
    # Load images
    img1 = io.imread(img1_path)
    img2 = io.imread(img2_path)
    
    # Verify images loaded correctly
    assert img1.shape == img2.shape, "Images have different dimensions"
    assert img1.dtype == img2.dtype, "Images have different data types"
    
    # Downsample for faster processing in tests
    img1_small = img1[::4, ::4]
    img2_small = img2[::4, ::4]
    
    # Calculate optical flow
    u, v, vorticity, _, _, _ = optical_flow_physics(
        img1_small, img2_small, lambda_1=10.0, lambda_2=100.0
    )
    
    # Basic assertions about the flow field
    assert u.shape == img1_small.shape, "Flow field u has wrong shape"
    assert v.shape == img1_small.shape, "Flow field v has wrong shape"
    assert vorticity.shape == img1_small.shape, "Vorticity field has wrong shape"
    
    # Check that there is some non-zero flow
    assert np.any(u != 0), "Flow field u is all zeros"
    assert np.any(v != 0), "Flow field v is all zeros"
    
    # Check that vorticity has both positive and negative values (for vortex pair)
    assert np.any(vorticity > 0), "No positive vorticity found"
    assert np.any(vorticity < 0), "No negative vorticity found"
    
    return u, v, vorticity, img1_small, img2_small


def test_wall_jet_basic():
    """Test basic optical flow on wall jet images."""
    # Check if images exist
    img1_path = os.path.join('img', 'wall_jet_1.tif')
    img2_path = os.path.join('img', 'wall_jet_2.tif')
    
    assert os.path.exists(img1_path), f"Image file not found: {img1_path}"
    assert os.path.exists(img2_path), f"Image file not found: {img2_path}"
    
    # Load images
    img1 = io.imread(img1_path)
    img2 = io.imread(img2_path)
    
    # Verify images loaded correctly
    assert img1.shape == img2.shape, "Images have different dimensions"
    assert img1.dtype == img2.dtype, "Images have different data types"
    
    # Downsample for faster processing in tests
    img1_small = img1[::4, ::4]
    img2_small = img2[::4, ::4]
    
    # Calculate optical flow
    u, v, vorticity, _, _, _ = optical_flow_physics(
        img1_small, img2_small, lambda_1=10.0, lambda_2=100.0
    )
    
    # Basic assertions about the flow field
    assert u.shape == img1_small.shape, "Flow field u has wrong shape"
    assert v.shape == img1_small.shape, "Flow field v has wrong shape"
    assert vorticity.shape == img1_small.shape, "Vorticity field has wrong shape"
    
    # Check that there is some non-zero flow
    assert np.any(u != 0), "Flow field u is all zeros"
    assert np.any(v != 0), "Flow field v is all zeros"
    
    return u, v, vorticity, img1_small, img2_small


def test_white_oval_basic():
    """Test basic optical flow on white oval images."""
    # Check if images exist
    img1_path = os.path.join('img', 'White_Oval_1.tif')
    img2_path = os.path.join('img', 'White_Oval_2.tif')
    
    assert os.path.exists(img1_path), f"Image file not found: {img1_path}"
    assert os.path.exists(img2_path), f"Image file not found: {img2_path}"
    
    # Load images
    img1 = io.imread(img1_path)
    img2 = io.imread(img2_path)
    
    # Verify images loaded correctly
    assert img1.shape == img2.shape, "Images have different dimensions"
    assert img1.dtype == img2.dtype, "Images have different data types"
    
    # Downsample for faster processing in tests
    img1_small = img1[::4, ::4]
    img2_small = img2[::4, ::4]
    
    # Calculate optical flow
    u, v, vorticity, _, _, _ = optical_flow_physics(
        img1_small, img2_small, lambda_1=10.0, lambda_2=100.0
    )
    
    # Basic assertions about the flow field
    assert u.shape == img1_small.shape, "Flow field u has wrong shape"
    assert v.shape == img1_small.shape, "Flow field v has wrong shape"
    assert vorticity.shape == img1_small.shape, "Vorticity field has wrong shape"
    
    # Check that there is some non-zero flow
    assert np.any(u != 0), "Flow field u is all zeros"
    assert np.any(v != 0), "Flow field v is all zeros"
    
    return u, v, vorticity, img1_small, img2_small


def test_2d_vortices_basic():
    """Test basic optical flow on 2D vortices images."""
    # Check if images exist
    img1_path = os.path.join('img', '2D_vortices_1.tif')
    img2_path = os.path.join('img', '2D_vortices_2.tif')
    
    assert os.path.exists(img1_path), f"Image file not found: {img1_path}"
    assert os.path.exists(img2_path), f"Image file not found: {img2_path}"
    
    # Load images
    img1 = io.imread(img1_path)
    img2 = io.imread(img2_path)
    
    # Verify images loaded correctly
    assert img1.shape == img2.shape, "Images have different dimensions"
    assert img1.dtype == img2.dtype, "Images have different data types"
    
    # Downsample for faster processing in tests
    img1_small = img1[::4, ::4]
    img2_small = img2[::4, ::4]
    
    # Calculate optical flow
    u, v, vorticity, _, _, _ = optical_flow_physics(
        img1_small, img2_small, lambda_1=10.0, lambda_2=100.0
    )
    
    # Basic assertions about the flow field
    assert u.shape == img1_small.shape, "Flow field u has wrong shape"
    assert v.shape == img1_small.shape, "Flow field v has wrong shape"
    assert vorticity.shape == img1_small.shape, "Vorticity field has wrong shape"
    
    # Check that there is some non-zero flow
    assert np.any(u != 0), "Flow field u is all zeros"
    assert np.any(v != 0), "Flow field v is all zeros"
    
    # Check that vorticity has both positive and negative values (for vortices)
    assert np.any(vorticity > 0), "No positive vorticity found"
    assert np.any(vorticity < 0), "No negative vorticity found"
    
    return u, v, vorticity, img1_small, img2_small


if __name__ == "__main__":
    # Run the tests and visualize the results
    print("Testing vortex pair images...")
    u, v, vorticity, img1, img2 = test_vortex_pair_basic()
    
    # Visualize the results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img1, cmap='gray')
    plt.title('First Image')
    
    plt.subplot(1, 3, 2)
    plt.imshow(img2, cmap='gray')
    plt.title('Second Image')
    
    plt.subplot(1, 3, 3)
    ax = vis_flow(v, u, gx=10, offset=0, mag=2, color='red')
    plt.title('Optical Flow Field')
    
    plt.tight_layout()
    plt.show()
    
    # Vorticity visualization
    plt.figure(figsize=(10, 8))
    plt.imshow(vorticity, cmap='RdBu_r')
    plt.colorbar(label='Vorticity')
    plt.title('Vorticity Field')
    plt.tight_layout()
    plt.show()
    
    # Repeat for other image sets if desired
    print("Testing wall jet images...")
    test_wall_jet_basic()
    
    print("Testing white oval images...")
    test_white_oval_basic()
    
    print("Testing 2D vortices images...")
    test_2d_vortices_basic()
