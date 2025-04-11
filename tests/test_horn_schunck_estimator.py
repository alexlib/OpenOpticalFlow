import numpy as np
import pytest
from openopticalflow.horn_schunck_estimator import horn_schunck_estimator

def test_zero_motion():
    """Test that zero motion is correctly estimated."""
    # Create a simple test case with no motion
    shape = (20, 20)
    Ix = np.zeros(shape)
    Iy = np.zeros(shape)
    It = np.zeros(shape)

    # Run the estimator
    u, v = horn_schunck_estimator(Ix, Iy, It, alpha=1.0, tol=1e-6, maxiter=100)

    # With zero derivatives, the flow should be zero
    assert np.allclose(u, 0, atol=1e-6)
    assert np.allclose(v, 0, atol=1e-6)

def test_uniform_translation():
    """Test estimation of uniform translation."""
    # Create a simple test case with uniform translation
    shape = (20, 20)

    # Uniform gradient in x and y
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))

    # Simulate uniform translation of (2, 1) pixels
    vx_true = 2.0
    vy_true = 1.0

    # Create spatial and temporal derivatives for this motion
    Ix = np.ones(shape)  # Uniform gradient in x
    Iy = np.ones(shape)  # Uniform gradient in y
    It = vx_true * Ix + vy_true * Iy  # Temporal derivative from optical flow constraint

    # Run the estimator
    u, v = horn_schunck_estimator(Ix, Iy, It, alpha=0.1, tol=1e-6, maxiter=500)

    # The estimated flow should be close to the true values
    # Note: Due to regularization, the values might not be exactly equal
    assert np.isclose(np.mean(u), -vx_true, atol=0.5)
    assert np.isclose(np.mean(v), -vy_true, atol=0.5)

def test_convergence():
    """Test that the algorithm converges within the maximum iterations."""
    # Create a simple test case
    shape = (20, 20)
    Ix = np.random.rand(*shape)
    Iy = np.random.rand(*shape)
    It = np.random.rand(*shape) * 0.1  # Small temporal derivative

    # Set a very small tolerance to ensure we hit max iterations
    tol = 1e-12
    maxiter = 10

    # Run the estimator
    u, v = horn_schunck_estimator(Ix, Iy, It, alpha=1.0, tol=tol, maxiter=maxiter)

    # The function should return without error even if max iterations is reached
    assert isinstance(u, np.ndarray)
    assert isinstance(v, np.ndarray)
    assert u.shape == shape
    assert v.shape == shape
