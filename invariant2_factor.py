import numpy as np
from scipy import ndimage

def invariant2_factor(vx, vy, factor_x, factor_y):
    """
    Calculate the second invariant (Q-criterion) of the velocity gradient tensor.
    
    Parameters:
        vx: x-component of velocity field
        vy: y-component of velocity field
        factor_x: converting factor from pixel to m (m/pixel) in x
        factor_y: converting factor from pixel to m (m/pixel) in y
    
    Returns:
        qq: Second invariant field (Q-criterion)
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
    vx_x = ndimage.convolve(vx, d_kernel.T/dx, mode='reflect') / factor_x
    vx_y = ndimage.convolve(vx, d_kernel/dx, mode='reflect') / factor_y
    vy_x = ndimage.convolve(vy, d_kernel.T/dx, mode='reflect') / factor_x
    vy_y = ndimage.convolve(vy, d_kernel/dx, mode='reflect') / factor_y
    
    # Get dimensions
    m, n = vx.shape
    qq = np.zeros((m, n))
    
    # Calculate Q-criterion at each point
    for i in range(m):
        for j in range(n):
            # Construct velocity gradient tensor
            u = np.array([[vx_x[i,j], vx_y[i,j]],
                         [vy_x[i,j], vy_y[i,j]]])
            
            # Calculate symmetric and antisymmetric parts
            s = 0.5 * (u + u.T)  # Symmetric part (strain tensor)
            q = 0.5 * (u - u.T)  # Antisymmetric part (rotation tensor)
            
            # Calculate Q-criterion
            qq[i,j] = 0.5 * (np.trace(q @ q.T) - np.trace(s @ s.T))
    
    return qq

def invariant2_factor_vectorized(vx, vy, factor_x, factor_y):
    """
    Vectorized version of invariant2_factor function.
    This version avoids loops and is much faster for large arrays.
    """
    dx = 1
    d_kernel = np.array([[0, -1, 0], 
                        [0, 0, 0], 
                        [0, 1, 0]]) / 2
    
    # Calculate velocity gradients
    vx_x = ndimage.convolve(vx, d_kernel.T/dx, mode='reflect') / factor_x
    vx_y = ndimage.convolve(vx, d_kernel/dx, mode='reflect') / factor_y
    vy_x = ndimage.convolve(vy, d_kernel.T/dx, mode='reflect') / factor_x
    vy_y = ndimage.convolve(vy, d_kernel/dx, mode='reflect') / factor_y
    
    # Calculate Q-criterion components
    s11 = vx_x
    s12 = 0.5 * (vx_y + vy_x)
    s21 = s12
    s22 = vy_y
    
    q12 = 0.5 * (vx_y - vy_x)
    q21 = -q12
    
    # Calculate Q-criterion
    qq = 2 * (q12 * q21) - (s11**2 + s12*s21 + s21*s12 + s22**2)
    
    return qq

# Example usage
if __name__ == "__main__":
    # Create sample velocity fields
    x, y = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
    vx = -y
    vy = x
    
    # Set conversion factors (example values)
    factor_x = 0.001  # 1 mm/pixel
    factor_y = 0.001  # 1 mm/pixel
    
    # Calculate Q-criterion using both methods
    qq1 = invariant2_factor(vx, vy, factor_x, factor_y)
    qq2 = invariant2_factor_vectorized(vx, vy, factor_x, factor_y)
    
    # Plot results
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(131)
    plt.quiver(x[::5, ::5], y[::5, ::5], vx[::5, ::5], vy[::5, ::5])
    plt.title('Velocity Field')
    plt.axis('equal')
    
    plt.subplot(132)
    plt.imshow(qq1, cmap='RdBu_r')
    plt.colorbar(label='Q-criterion (Loop)')
    plt.title('Q-criterion (Loop)')
    
    plt.subplot(133)
    plt.imshow(qq2, cmap='RdBu_r')
    plt.colorbar(label='Q-criterion (Vectorized)')
    plt.title('Q-criterion (Vectorized)')
    
    plt.tight_layout()
    plt.show()