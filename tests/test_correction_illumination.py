import numpy as np
import pytest
from openopticalflow.correction_illumination import correction_illumination

def test_global_illumination_correction():
    """Test that global illumination correction works correctly."""
    # Create test images with different brightness levels
    img1 = np.ones((100, 100)) * 100  # Bright image
    img2 = np.ones((100, 100)) * 50   # Darker image
    window = [0, 100, 0, 100]  # Full image window
    
    # Apply global correction only (size_average=0)
    _, corrected_img2 = correction_illumination(img1, img2, window, 0)
    
    # The mean of corrected_img2 should be close to img1's mean
    assert np.isclose(np.mean(corrected_img2), np.mean(img1), rtol=1e-5)

def test_local_illumination_correction():
    """Test that local illumination correction works correctly."""
    # Create test images with local illumination differences
    img1 = np.ones((100, 100)) * 100
    img2 = np.ones((100, 100)) * 100
    
    # Add a bright spot to img2
    img2[40:60, 40:60] = 150
    
    window = [0, 100, 0, 100]
    
    # Apply local correction with a window size of 10
    _, corrected_img2 = correction_illumination(img1, img2, window, 10)
    
    # The bright spot should be reduced (not completely eliminated due to edge effects)
    # Check that the difference between the center and surroundings is smaller
    original_diff = img2[50, 50] - img2[30, 30]
    corrected_diff = corrected_img2[50, 50] - corrected_img2[30, 30]
    
    assert corrected_diff < original_diff

def test_no_change_for_identical_images():
    """Test that identical images remain unchanged."""
    img = np.random.rand(50, 50) * 255
    window = [0, 50, 0, 50]
    
    # Apply correction to identical images
    img1_out, img2_out = correction_illumination(img, img.copy(), window, 5)
    
    # Both output images should be very close to the input
    assert np.allclose(img, img1_out, rtol=1e-5)
    assert np.allclose(img, img2_out, rtol=1e-5)
