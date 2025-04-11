import numpy as np
import pytest
from openopticalflow.optical_flow_physics import optical_flow_physics, laplacian, vorticity

def test_laplacian():
    """Test the Laplacian function."""
    # Create a simple test image
    img = np.zeros((10, 10))
    img[4:7, 4:7] = 1.0  # Create a square in the middle

    # Calculate Laplacian
    lap = laplacian(img, dx=1.0)

    # The Laplacian should be non-zero at the edges of the square
    # and zero elsewhere
    assert np.sum(np.abs(lap)) > 0
    assert np.all(lap[5, 5] == 0)  # Center of square should be zero

    # Check that the sum of Laplacian is close to zero for a closed shape
    assert np.isclose(np.sum(lap), 0, atol=1e-10)

def test_vorticity():
    """Test the vorticity function."""
    # Create a simple rotational velocity field (rigid body rotation)
    x, y = np.meshgrid(np.linspace(-1, 1, 20), np.linspace(-1, 1, 20))
    ux = -y  # Clockwise rotation
    uy = x

    # Calculate vorticity
    vor = vorticity(ux, uy)

    # For rigid body rotation, vorticity should be negative
    # The sign is negative due to the coordinate system convention
    assert np.all(vor < 0)

    # The magnitude should be approximately constant
    assert np.std(vor) / np.abs(np.mean(vor)) < 0.5

def test_optical_flow_physics_zero_motion():
    """Test optical flow calculation with zero motion."""
    # Create two identical images
    img = np.random.rand(20, 20)

    # Calculate optical flow
    u, v, vor, _, _, _ = optical_flow_physics(img, img, 1.0, 1.0)

    # With identical images, flow should be close to zero
    assert np.allclose(u, 0, atol=1e-3)
    assert np.allclose(v, 0, atol=1e-3)
    assert np.allclose(vor, 0, atol=1e-3)

def test_optical_flow_physics_translation():
    """Test optical flow calculation with simple translation."""
    # Create a simple pattern
    x, y = np.meshgrid(np.linspace(0, 1, 20), np.linspace(0, 1, 20))
    img1 = np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)

    # Shift the pattern slightly
    dx, dy = 0.05, 0.05
    x_shifted = x - dx
    y_shifted = y - dy
    img2 = np.sin(2 * np.pi * x_shifted) * np.sin(2 * np.pi * y_shifted)

    # Calculate optical flow
    u, v, _, _, _, _ = optical_flow_physics(img1, img2, 10.0, 100.0)

    # The mean flow should be approximately in the direction of the shift
    # Note: The values might not match exactly due to regularization
    assert np.mean(u) > 0  # Positive y-flow (downward in image coordinates)
    assert np.mean(v) > 0  # Positive x-flow (rightward in image coordinates)
