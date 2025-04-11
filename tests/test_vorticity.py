"""
Tests for the vorticity calculation module.
"""
import numpy as np
import pytest
from openopticalflow.vorticity import vorticity

def test_zero_vorticity():
    """Test that uniform flow has zero vorticity."""
    # Create a uniform flow field
    shape = (20, 20)
    vx = np.ones(shape)
    vy = np.ones(shape)

    # Calculate vorticity
    vor = vorticity(vx, vy)

    # Uniform flow should have zero vorticity
    assert np.allclose(vor, 0, atol=1e-10)

def test_rigid_body_rotation():
    """Test vorticity calculation for rigid body rotation."""
    # Create a rigid body rotation flow field
    n = 20
    x, y = np.meshgrid(np.linspace(-1, 1, n), np.linspace(-1, 1, n))

    # Clockwise rotation around the origin
    vx = -y
    vy = x

    # Calculate vorticity
    vor = vorticity(vx, vy)

    # For rigid body rotation, vorticity should be negative
    # The sign is negative due to the coordinate system convention
    assert np.all(vor < 0)

    # The magnitude should be approximately constant
    assert np.std(vor) / np.abs(np.mean(vor)) < 0.5

def test_shear_flow():
    """Test vorticity calculation for shear flow."""
    # Create a shear flow field (velocity varies in y direction)
    n = 20
    y = np.linspace(0, 1, n)
    vx = np.tile(y, (n, 1)).T  # vx increases with y
    vy = np.zeros((n, n))      # vy is zero everywhere

    # Calculate vorticity
    vor = vorticity(vx, vy)

    # For this shear flow, vorticity should be positive due to the coordinate system
    # The exact value depends on the derivative implementation
    assert np.all(vor > 0)

    # The vorticity should be approximately constant
    assert np.std(vor) / np.mean(vor) < 0.5
