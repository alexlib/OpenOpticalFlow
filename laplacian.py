import numpy as np
from scipy import ndimage

def laplacian(u, h):
    """
    Calculate the Laplacian of an image.
    
    Parameters:
        u: Input image/array
        h: Step size
    
    Returns:
        delu: Laplacian of the input
    """
    # Define kernel
    H = np.array([[1, 1, 1],
                  [1, 0, 1],
                  [1, 1, 1]])
    # Alternative kernel (more natural):
    # H = np.array([[0, 1, 0],
    #               [1, 0, 1],
    #               [0, 1, 0]])
    
    # Calculate Laplacian
    ones_filtered = ndimage.convolve(np.ones_like(u), H/(h*h), mode='reflect')
    u_filtered = ndimage.convolve(u, H/(h*h), mode='reflect')
    delu = -u * ones_filtered + u_filtered
    
    return delu