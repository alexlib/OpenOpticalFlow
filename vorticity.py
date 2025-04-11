import numpy as np
from scipy import ndimage

def vorticity(vx, vy):
    """
    Calculate vorticity from velocity components.
    Direct translation of MATLAB's vorticity.m function.
    
    Parameters:
        vx: x-component of velocity field (numpy array)
        vy: y-component of velocity field (numpy array)
    
    Returns:
        omega: Vorticity field (numpy array)
    """
    # Optional smoothing (commented out as in MATLAB)
    # In MATLAB: Vx = imfilter(Vx, [1 1 1 1 1]'*[1 1 1 1 1]/25,'symmetric')
    # kernel = np.ones((5, 5)) / 25
    # vx = ndimage.convolve(vx, kernel, mode='reflect')
    # vy = ndimage.convolve(vy, kernel, mode='reflect')
    
    # Define spatial step
    dx = 1
    
    # Define derivative kernel (matching MATLAB's D matrix)
    d_kernel = np.array([[0, -1, 0],
                        [0, 0, 0],
                        [0, 1, 0]]) / 2
    
    # Calculate velocity gradients
    # In MATLAB: Vy_x = imfilter(Vy, D'/dx, 'symmetric', 'same')
    vy_x = ndimage.convolve(vy, d_kernel.T/dx, mode='reflect')
    
    # In MATLAB: Vx_y = imfilter(Vx, D/dx, 'symmetric', 'same')
    vx_y = ndimage.convolve(vx, d_kernel/dx, mode='reflect')
    
    # Calculate vorticity (matching MATLAB's omega=Vy_x-Vx_y)
    omega = vy_x - vx_y
    
    return omega

# Example usage and validation
if __name__ == "__main__":
    # Create sample velocity fields (matching MATLAB example)
    x, y = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
    vx = -y  # Example: rigid body rotation
    vy = x
    
    # Calculate vorticity
    omega = vorticity(vx, vy)
    
    # Optional: Visualize results
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(131)
    plt.quiver(x[::5, ::5], y[::5, ::5], vx[::5, ::5], vy[::5, ::5])
    plt.title('Velocity Field')
    plt.axis('equal')
    
    plt.subplot(132)
    plt.imshow(omega, cmap='RdBu_r')
    plt.colorbar(label='Vorticity')
    plt.title('Vorticity Field')
    
    plt.tight_layout()
    plt.show()