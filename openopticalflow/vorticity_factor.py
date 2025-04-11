import numpy as np
from scipy import ndimage

def vorticity(vx, vy, factor_x, factor_y):
    """
    Calculate vorticity from velocity components with scaling factors.
    
    Parameters:
        vx: x-component of velocity field
        vy: y-component of velocity field
        factor_x: converting factor from pixel to m (m/pixel) in x
        factor_y: converting factor from pixel to m (m/pixel) in y
    
    Returns:
        omega: Vorticity field
    """
    # Optional smoothing (commented out as in MATLAB)
    # kernel = np.ones((5, 5)) / 25
    # vx = ndimage.convolve(vx, kernel, mode='reflect')
    # vy = ndimage.convolve(vy, kernel, mode='reflect')
    
    # Define derivative kernel
    dx = 1
    d_kernel = np.array([[0, -1, 0], 
                        [0, 0, 0], 
                        [0, 1, 0]]) / 2
    
    # Calculate velocity gradients
    vy_x = ndimage.convolve(vy, d_kernel.T/dx, mode='reflect')
    vx_y = ndimage.convolve(vx, d_kernel/dx, mode='reflect')
    
    # Calculate vorticity with scaling factors
    omega = (vy_x/factor_x - vx_y/factor_y)
    
    return omega