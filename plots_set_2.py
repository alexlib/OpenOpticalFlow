import numpy as np
import matplotlib.pyplot as plt
from vorticity import vorticity
from invariant2_factor import invariant2_factor
from vis_flow import vis_flow

def plots_set_2(ux, uy):
    """
    Create visualization plots for flow analysis including velocity magnitude,
    vorticity, Q-criterion, and streamlines.
    """
    # Calculate velocity magnitude
    u_mag = np.sqrt(ux**2 + uy**2)
    u_max = np.max(u_mag)
    u_mag = u_mag / u_max

    # Calculate vorticity
    vor = vorticity(ux, uy)
    vor_max = np.max(np.abs(vor))
    vor = vor/vor_max

    # Calculate Q-criterion
    Q = invariant2_factor(ux, uy, 1, 1)

    # Plot velocity magnitude field with streamlines
    plt.figure(20, figsize=(10, 8))
    plt.imshow(u_mag, extent=[0, u_mag.shape[1], 0, u_mag.shape[0]],
               vmin=0, vmax=1, aspect='equal')

    m, n = ux.shape
    y, x = np.meshgrid(np.arange(m), np.arange(n), indexing='ij')
    plt.streamplot(x, y, ux, uy, density=2, color='yellow')
    plt.xlabel('x (pixels)')
    plt.ylabel('y (pixels)')
    plt.gca().invert_yaxis()
    plt.title('Velocity Magnitude Field')
    plt.colorbar()
    plt.show()

    # Plot vorticity field with velocity vectors
    plt.figure(21, figsize=(10, 8))
    plt.imshow(vor, extent=[0, vor.shape[1], 0, vor.shape[0]],
               vmin=-1, vmax=1, cmap='RdBu_r', aspect='equal')

    # Plot velocity vectors
    gx = max(vor.shape) // 25
    mag = 3  # Fixed value for better visibility
    vis_flow(ux, uy, gx=gx, offset=1, mag=mag, color='black')  # Using normalized vectors

    plt.xlabel('x (pixels)')
    plt.ylabel('y (pixels)')
    plt.gca().invert_yaxis()
    plt.title('Vorticity Field')
    plt.colorbar()
    plt.show()

    # Plot Q-criterion field
    plt.figure(22, figsize=(10, 8))
    plt.imshow(Q, extent=[0, Q.shape[1], 0, Q.shape[0]],
               vmin=0, vmax=0.1, cmap='viridis', aspect='equal')
    plt.xlabel('x (pixels)')
    plt.ylabel('y (pixels)')
    plt.gca().invert_yaxis()
    plt.title('Q-criterion Field')
    plt.colorbar()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Create sample velocity fields (matching MATLAB example)
    x, y = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
    ux = -y  # Example: rigid body rotation
    uy = x
    plots_set_2(ux, uy)
