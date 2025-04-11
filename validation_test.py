import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from skimage import io
from optical_flow_physics import optical_flow_physics
from preprocessing import pre_processing, correction_illumination, shift_image_refine

def create_synthetic_case():
    """Create a synthetic test case with known displacement"""
    # Create a random pattern
    np.random.seed(42)  # for reproducibility
    size = (200, 200)
    img = np.zeros(size)
    
    # Add random Gaussian blobs
    for _ in range(50):
        x = np.random.randint(0, size[1])
        y = np.random.randint(0, size[0])
        sigma = np.random.uniform(2, 5)
        amplitude = np.random.uniform(0.5, 1.0)
        
        y_grid, x_grid = np.ogrid[-y:size[0]-y, -x:size[1]-x]
        r2 = x_grid*x_grid + y_grid*y_grid
        img += amplitude * np.exp(-r2/(2.*sigma**2))

    # Normalize to 0-255 range
    img = ((img - img.min()) * (255.0 / (img.max() - img.min()))).astype(np.uint8)
    
    return img

def apply_known_shift(image, dx, dy):
    """Apply known shift to image using Fourier shift theorem"""
    rows, cols = image.shape
    y_grid, x_grid = np.mgrid[:rows, :cols]
    
    shifted = ndimage.shift(image, (dy, dx), mode='reflect', order=3)
    return shifted

def compute_error_metrics(ux, uy, true_dx, true_dy):
    """Compute error metrics between estimated and true displacement"""
    error_x = ux - true_dx
    error_y = uy - true_dy
    
    rmse = np.sqrt(np.mean(error_x**2 + error_y**2))
    mae = np.mean(np.abs(error_x) + np.abs(error_y))
    
    return rmse, mae

def plot_flow_field(img1, img2, ux, uy, true_dx, true_dy):
    """Plot original, shifted images and flow field with normalized vectors"""
    plt.figure(figsize=(15, 5))
    
    # Original image
    plt.subplot(131)
    plt.imshow(img1, cmap='gray')
    plt.title('Original Image')
    
    # Shifted image
    plt.subplot(132)
    plt.imshow(img2, cmap='gray')
    plt.title('Shifted Image')
    
    # Flow field with normalized vectors
    plt.subplot(133)
    plt.imshow(img1, cmap='gray', alpha=0.5)
    
    # Subsample the flow field for better visualization
    sy, sx = ux.shape
    step = max(sx, sy) // 25  # Show about 25 vectors in largest dimension
    
    y, x = np.mgrid[0:sy:step, 0:sx:step]
    u = ux[::step, ::step]
    v = uy[::step, ::step]
    
    # Normalize vectors
    magnitude = np.sqrt(u**2 + v**2)
    if magnitude.max() > 0:
        u = u / magnitude.max()
        v = v / magnitude.max()
    
    plt.quiver(x, y, u, v, color='r', scale=15, width=0.003)
    plt.title('Flow Field\n(normalized & subsampled)')
    
    plt.tight_layout()

def run_test_case(img1_path, img2_path, true_dx=None, true_dy=None, create_synthetic=False):
    """Run optical flow test case for given image pair"""
    # Read images
    img1 = io.imread(img1_path)
    
    if create_synthetic:
        # Create shifted image using known displacement
        img2 = apply_known_shift(img1, true_dx, true_dy)
        # Save shifted image for verification
        base_name = img1_path.rsplit('.', 1)[0]
        io.imsave(f'{base_name}_shifted.tif', img2)
    else:
        img2 = io.imread(img2_path)
        
    # Parameters for optical flow computation - matching MATLAB defaults
    lambda_1 = 20.0    # Horn-Schunck regularization
    lambda_2 = 2000.0  # Liu-Shen regularization
    scale_im = 1.0     # No downscaling
    size_filter = 4    # Gaussian filter size
    size_average = 0   # No illumination correction
    no_iteration = 3   # Number of refinement iterations
    
    # Store originals
    i1_original = img1.astype(float)
    i2_original = img2.astype(float)
    
    # Initial pre-processing
    window_shifting = np.array([1, i1_original.shape[1], 1, i1_original.shape[0]])
    i1, i2 = correction_illumination(i1_original, i2_original, window_shifting, size_average)
    i1, i2 = pre_processing(i1, i2, scale_im, size_filter)
    
    # Initial optical flow
    ux0, uy0, vor, ux_horn, uy_horn, error1 = optical_flow_physics(i1, i2, lambda_1, lambda_2)
    
    # Convert to uint8 for shift operation
    im1 = i1_original.astype(np.uint8)
    im2 = i2_original.astype(np.uint8)
    
    ux_corr = ux0.copy()
    uy_corr = uy0.copy()
    
    # Iterative refinement
    for k in range(no_iteration):
        # Shift image based on current estimate
        im1_shift, uxi, uyi = shift_image_refine(ux_corr, uy_corr, im1, im2)
        
        # Convert to float for processing
        i1 = im1_shift.astype(float)
        i2 = im2.astype(float)
        
        # Additional pre-processing with smaller filter
        size_filter_1 = 2
        i1, i2 = pre_processing(i1, i2, 1, size_filter_1)
        
        # Calculate correction
        dux, duy, vor, dux_horn, duy_horn, error2 = optical_flow_physics(
            i1, i2, lambda_1, lambda_2)
        
        # Update flow fields
        ux_corr = uxi + dux
        uy_corr = uyi + duy
    
    # Final velocity field
    ux = ux_corr
    uy = uy_corr
    
    # Plot results with normalized vectors
    plot_flow_field(img1, img2, ux, uy, true_dx, true_dy)
    
    if true_dx is not None and true_dy is not None:
        # Compute error metrics
        rmse, mae = compute_error_metrics(ux, uy, true_dx, true_dy)
        print(f"True displacement: dx={true_dx}, dy={true_dy}")
        print(f"Mean estimated displacement: dx={np.mean(ux):.3f}, dy={np.mean(uy):.3f}")
        print(f"RMSE: {rmse:.3f}")
        print(f"MAE: {mae:.3f}")
    else:
        print(f"Mean displacement: dx={np.mean(ux):.3f}, dy={np.mean(uy):.3f}")
    
    return ux, uy, vor

def main():
    # Test Case 1: White Oval with synthetic shift
    print("\nTest Case 1: White Oval (Synthetic Shift)")
    print("-----------------------------------------")
    ux1, uy1, vor1 = run_test_case(
        img1_path='White_Oval_1.tif',
        img2_path=None,
        true_dx=2.5,
        true_dy=1.5,
        create_synthetic=True
    )
    
    # Test Case 2: Wall Jet (real image pair)
    print("\nTest Case 2: Wall Jet")
    print("--------------------")
    ux2, uy2, vor2 = run_test_case(
        img1_path='wall_jet_1.tif',
        img2_path='wall_jet_2.tif',
        create_synthetic=False
    )
    
    # Additional visualization for Wall Jet case
    plt.figure(figsize=(10, 5))
    
    # Plot vorticity field
    plt.subplot(121)
    vor_normalized = vor2 / np.max(np.abs(vor2))
    plt.imshow(vor_normalized, cmap='RdBu_r', vmin=-1, vmax=1)
    plt.colorbar(label='Normalized Vorticity')
    plt.title('Vorticity Field')
    
    # Plot velocity magnitude
    plt.subplot(122)
    vel_mag = np.sqrt(ux2**2 + uy2**2)
    vel_mag_normalized = vel_mag / np.max(vel_mag)
    plt.imshow(vel_mag_normalized, cmap='viridis')
    plt.colorbar(label='Normalized Velocity Magnitude')
    plt.title('Velocity Magnitude')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
