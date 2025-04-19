import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from vis_flow import vis_flow

def plots_set_1(I_region1, I_region2, ux0, uy0, Im1, Im2, ux, uy):
    """
    Create visualization plots for optical flow analysis.
    Matches MATLAB implementation exactly.

    Parameters:
        I_region1, I_region2: Downsampled images
        ux0, uy0: Initial (coarse) velocity fields
        Im1, Im2: Original images
        ux, uy: Refined velocity fields
    """


    # Plot initial velocity vector field (Figure 3 in MATLAB)
    plt.figure(figsize=(10, 8))
    gx, offset = 30, 1
    ax = vis_flow(ux0, uy0, gx, offset, 2, 'red')  # Using normalized vectors, mag=2 for longer arrows
    plt.xlabel('x (pixels)')
    plt.ylabel('y (pixels)')
    ax.set_aspect('equal')
    ax.invert_yaxis()
    plt.title('Coarse-Grained Velocity Field')
    plt.show()

    # Plot streamlines (Figure 4 in MATLAB)
    plt.figure(4, figsize=(10, 8))
    m, n = ux0.shape
    y, x = np.meshgrid(np.arange(m), np.arange(n), indexing='ij')
    plt.streamplot(x, y, ux0, uy0, density=2, color='blue')
    plt.xlabel('x (pixels)')
    plt.ylabel('y (pixels)')
    plt.axis('image')
    plt.gca().invert_yaxis()
    plt.title('Coarse-Grained Streamlines')
    plt.show()

    # Create an animated figure to show motion between original images
    fig, ax = plt.subplots(figsize=(10, 10))

    # Initialize with the first image
    im = ax.imshow(Im1, cmap='gray', animated=True)
    ax.set_title('Original Image Animation')
    ax.set_xlabel('x (pixels)')
    ax.set_ylabel('y (pixels)')
    ax.axis('image')

    # Function to update the animation
    def update(frame):
        # Alternate between the two images
        if frame % 2 == 0:
            im.set_array(Im1)
            ax.set_title('Original Image 1')
        else:
            im.set_array(Im2)
            ax.set_title('Original Image 2')
        return [im]

    # Create animation with 500ms interval
    _ = FuncAnimation(fig, update, frames=10, interval=500, blit=True, repeat=True)

    plt.tight_layout()
    plt.show()

    # Plot refined velocity vector field (Figure 12 in MATLAB)
    plt.figure(12, figsize=(10, 8))
    gx, offset = 50, 1
    # Plot velocity vectors
    ax = vis_flow(ux, uy, gx, offset, 2, 'red')  # Using normalized vectors, mag=2 for longer arrows
    plt.xlabel('x (pixels)')
    plt.ylabel('y (pixels)')
    ax.set_aspect('equal')
    ax.invert_yaxis()
    plt.title('Refined Velocity Field')
    plt.show()

    # Plot streamlines (Figure 13 in MATLAB)
    plt.figure(13, figsize=(10, 8))
    m, n = ux.shape
    y, x = np.meshgrid(np.arange(m), np.arange(n), indexing='ij')
    plt.streamplot(x, y, ux, uy, density=2, color='blue')
    plt.xlabel('x (pixels)')
    plt.ylabel('y (pixels)')
    plt.axis('image')
    plt.gca().invert_yaxis()
    plt.title('Refined Streamlines')
    plt.show()

# Example usage
if __name__ == "__main__":
    # Load and downsample vortex pair images
    from skimage import io, transform
    Im2 = io.imread('img/vortex_pair_particles_1.tif')
    Im1 = io.imread('img/vortex_pair_particles_2.tif')
    I_region1 = transform.downscale_local_mean(Im1, (4, 4))
    I_region2 = transform.downscale_local_mean(Im2, (4, 4))

    # Perform optical flow analysis
    from optical_flow_physics import optical_flow_physics
    ux0, uy0, vor, ux_horn, uy_horn, error1 = optical_flow_physics(I_region1, I_region2, 20, 2000)
    ux, uy = ux0, uy0


    plots_set_1(I_region1, I_region2, ux0, uy0, Im1, Im2, ux, uy)
