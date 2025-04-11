import numpy as np
from scipy import ndimage
from skimage.transform import resize

def shift_image_fun_refine_1(ux, uy, Im1, Im2):
    """
    Shift and refine image based on velocity fields.
    
    Parameters:
        ux, uy: Initial velocity fields in pixels/unit time in coarse image
        Im1, Im2: Finer images at times 1 and 2
    
    Returns:
        Im1_shift: Shifted image from Im1 based on uxI and uyI
        uxI, uyI: Interpolated velocity fields in pixels/unit time in finer image
    """
    # Convert inputs to float
    Im1 = Im1.astype(float)
    Im2 = Im2.astype(float)
    ux = ux.astype(float)
    uy = uy.astype(float)

    # Get dimensions of velocity field
    m0, n0 = ux.shape
    
    # Normalized size of coarse image
    x0_normalized = np.linspace(1/n0, 1, n0)
    y0_normalized = np.linspace(1/m0, 1, m0)
    X, Y = np.meshgrid(x0_normalized, y0_normalized)

    # Get dimensions of images
    m1, n1 = Im1.shape
    
    # Normalized size of finer image
    x1_normalized = np.linspace(1/n1, 1, n1)
    y1_normalized = np.linspace(1/m1, 1, m1)
    XI, YI = np.meshgrid(x1_normalized, y1_normalized)

    # Obtain interpolated velocity field in the finer image
    # Using resize instead of interp2 as it's more efficient for uniform scaling
    uxI = (n1/n0) * resize(ux, (len(y1_normalized), len(x1_normalized)), 
                          preserve_range=True)
    uyI = (m1/m0) * resize(uy, (len(y1_normalized), len(x1_normalized)), 
                          preserve_range=True)

    # Generate shifted image from Im1 based on rounded velocity field
    Im1_shift0 = Im2.copy()
    
    # Create coordinate arrays
    y_coords, x_coords = np.mgrid[0:m1, 0:n1]
    
    # Calculate shifted coordinates
    y_shift = y_coords + np.round(uyI).astype(int)
    x_shift = x_coords + np.round(uxI).astype(int)
    
    # Create mask for valid coordinates
    valid_mask = ((y_shift >= 0) & (y_shift < m1) & 
                 (x_shift >= 0) & (x_shift < n1))
    
    # Apply shifts where valid
    Im1_shift0[y_shift[valid_mask], x_shift[valid_mask]] = Im1[y_coords[valid_mask], 
                                                              x_coords[valid_mask]]
    
    # Where invalid, keep original values
    Im1_shift0[~valid_mask] = Im1[~valid_mask]

    # Refine the shifted image
    Im3 = Im1_shift0
    Im1_shift1 = Im3.copy()
    
    # Calculate fractional displacements
    duxI = uxI - np.round(uxI)
    duyI = uyI - np.round(uyI)

    # Apply Gaussian filter
    mask_size = 10
    std = 0.6 * mask_size
    kernel = ndimage.gaussian_filter(np.ones((mask_size, mask_size)), std)
    kernel = kernel / kernel.sum()  # Normalize
    
    duxI = ndimage.convolve(duxI, kernel, mode='reflect')
    duyI = ndimage.convolve(duyI, kernel, mode='reflect')

    # Set step sizes
    dx = dy = dt = 1

    # Calculate refined shift using first-order approximation
    for i in range(m1-1):
        for j in range(n1-1):
            term1 = (Im3[i, j+dx] * duxI[i, j+dx] - 
                    Im3[i, j] * duxI[i, j]) / dx
            term2 = (Im3[i+dy, j] * duyI[i+dy, j] - 
                    Im3[i, j] * duyI[i, j]) / dy
            Im1_shift1[i, j] = Im3[i, j] - (term1 + term2) * dt

    # Convert to uint8
    Im1_shift = np.clip(Im1_shift1, 0, 255).astype(np.uint8)

    return Im1_shift, uxI, uyI