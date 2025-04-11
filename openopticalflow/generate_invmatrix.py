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

    b11 = a22/det_a
    b12 = -a12/det_a
    b22 = a11/det_a

    return b11, b12, b22