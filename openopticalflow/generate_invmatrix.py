import numpy as np
from scipy import ndimage

def generate_invmatrix(i, alpha, h):
    """
    Generate inverse matrix for Liu-Shen optical flow estimation

    Parameters:
        i: Input image
        alpha: Regularization parameter
        h: Spatial step
    """
    # Define kernels
    d = np.array([[0, -1, 0], [0, 0, 0], [0, 1, 0]]) / 2
    m = np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]]) / 4
    d2 = np.array([[0, 1, 0], [0, -2, 0], [0, 1, 0]])
    h_kernel = np.ones((3, 3))

    # Calculate matrix components
    cmtx = ndimage.convolve(np.ones_like(i), h_kernel/(h*h), mode='reflect')

    a11 = i * (ndimage.convolve(i, d2/(h*h), mode='reflect') - 2*i/(h*h)) - alpha*cmtx
    a22 = i * (ndimage.convolve(i, d2.T/(h*h), mode='reflect') - 2*i/(h*h)) - alpha*cmtx
    a12 = i * ndimage.convolve(i, m/(h*h), mode='reflect')

    # Calculate determinant and inverse components
    det_a = a11*a22 - a12*a12

    # Add a small epsilon to avoid division by zero
    epsilon = 1e-10

    # Create a mask for valid determinant values
    valid_mask = np.abs(det_a) > epsilon

    # Initialize inverse matrix components with zeros
    b11 = np.zeros_like(det_a)
    b12 = np.zeros_like(det_a)
    b22 = np.zeros_like(det_a)

    # Calculate inverse only where determinant is valid
    b11[valid_mask] = a22[valid_mask] / det_a[valid_mask]
    b12[valid_mask] = -a12[valid_mask] / det_a[valid_mask]
    b22[valid_mask] = a11[valid_mask] / det_a[valid_mask]

    # For invalid determinants, use regularized values
    # This is a simple regularization that preserves the matrix structure
    invalid_mask = ~valid_mask
    if np.any(invalid_mask):
        # Use identity matrix scaled by the average of valid diagonal elements
        if np.any(valid_mask):
            scale = 0.5 * (np.mean(np.abs(b11[valid_mask])) + np.mean(np.abs(b22[valid_mask])))
        else:
            scale = 1.0
        b11[invalid_mask] = scale
        b22[invalid_mask] = scale
        b12[invalid_mask] = 0.0

    return b11, b12, b22