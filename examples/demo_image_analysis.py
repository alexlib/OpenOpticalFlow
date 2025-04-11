"""
Demo script showing how to use OpenOpticalFlow with the example images.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io

from openopticalflow.optical_flow_physics import optical_flow_physics
from openopticalflow.vis_flow import vis_flow
from openopticalflow.preprocessing import pre_processing


def analyze_image_pair(img1_path, img2_path, title, lambda_1=10.0, lambda_2=100.0, downsample=2):
    """
    Analyze a pair of images and visualize the optical flow and vorticity.
    
    Parameters:
        img1_path (str): Path to the first image
        img2_path (str): Path to the second image
        title (str): Title for the plots
        lambda_1 (float): Regularization parameter for Horn-Schunck
        lambda_2 (float): Regularization parameter for Liu-Shen
        downsample (int): Downsampling factor for faster processing
    """
    print(f"Analyzing {title}...")
    
    # Load images
    img1 = io.imread(img1_path)
    img2 = io.imread(img2_path)
    
    # Downsample for faster processing
    if downsample > 1:
        img1 = img1[::downsample, ::downsample]
        img2 = img2[::downsample, ::downsample]
    
    # Preprocess images
    img1_proc, img2_proc = pre_processing(img1, img2, scale_im=1.0, size_filter=3)
    
    # Calculate optical flow
    u, v, vorticity, ux_horn, uy_horn, error = optical_flow_physics(
        img1_proc, img2_proc, lambda_1=lambda_1, lambda_2=lambda_2
    )
    
    # Visualize the results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img1, cmap='gray')
    plt.title(f'{title} - First Image')
    
    plt.subplot(1, 3, 2)
    plt.imshow(img2, cmap='gray')
    plt.title(f'{title} - Second Image')
    
    plt.subplot(1, 3, 3)
    ax = vis_flow(v, u, gx=10, offset=0, mag=2, color='red')
    plt.title(f'{title} - Optical Flow Field')
    
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_').lower()}_flow.png")
    plt.show()
    
    # Vorticity visualization
    plt.figure(figsize=(10, 8))
    plt.imshow(vorticity, cmap='RdBu_r')
    plt.colorbar(label='Vorticity')
    plt.title(f'{title} - Vorticity Field')
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_').lower()}_vorticity.png")
    plt.show()
    
    return u, v, vorticity


def main():
    """Run the demo on all example image pairs."""
    # Create examples directory if it doesn't exist
    os.makedirs('examples/output', exist_ok=True)
    
    # Analyze vortex pair
    analyze_image_pair(
        'img/vortex_pair_particles_1.tif',
        'img/vortex_pair_particles_2.tif',
        'Vortex Pair'
    )
    
    # Analyze wall jet
    analyze_image_pair(
        'img/wall_jet_1.tif',
        'img/wall_jet_2.tif',
        'Wall Jet'
    )
    
    # Analyze white oval
    analyze_image_pair(
        'img/White_Oval_1.tif',
        'img/White_Oval_2.tif',
        'White Oval'
    )
    
    # Analyze 2D vortices
    analyze_image_pair(
        'img/2D_vortices_1.tif',
        'img/2D_vortices_2.tif',
        '2D Vortices'
    )


if __name__ == "__main__":
    main()
