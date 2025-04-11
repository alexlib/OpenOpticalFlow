import numpy as np
from scipy import ndimage
from skimage.transform import resize

def pre_processing(im1, im2, scale_im, size_filter):
    """
    Preprocess images by resizing and applying Gaussian filter
    
    Parameters:
        im1, im2: Input images
        scale_im: Scale factor for resizing
        size_filter: Size of Gaussian filter
    """
    # Convert to float if needed
    im1 = im1.astype(float)
    im2 = im2.astype(float)
    
    # Resize images
    if scale_im != 1:
        new_shape = tuple(int(s * scale_im) for s in im1.shape)
        i1 = resize(im1, new_shape, preserve_range=True)
        i2 = resize(im2, new_shape, preserve_range=True)
    else:
        i1, i2 = im1, im2
    
    # Apply Gaussian filter
    std = size_filter * 0.62
    i1 = ndimage.gaussian_filter(i1, std)
    i2 = ndimage.gaussian_filter(i2, std)
    
    return i1, i2

def correction_illumination(i1, i2, window_shifting, size_average):
    """
    Correct illumination differences between images
    
    Parameters:
        i1, i2: Input images
        window_shifting: [x1, x2, y1, y2] defining correction window
        size_average: Size of averaging window
    """
    if size_average <= 0:
        return i1, i2
        
    x1, x2, y1, y2 = window_shifting
    
    # Calculate local means
    kernel = np.ones((size_average, size_average)) / (size_average ** 2)
    i1_mean = ndimage.convolve(i1, kernel, mode='reflect')
    i2_mean = ndimage.convolve(i2, kernel, mode='reflect')
    
    # Correct intensities
    i1_corr = i1 - i1_mean + np.mean(i1_mean)
    i2_corr = i2 - i2_mean + np.mean(i2_mean)
    
    return i1_corr, i2_corr

def shift_image_refine(ux, uy, im1, im2):
    """
    Shift image based on velocity field
    
    Parameters:
        ux, uy: Velocity components (possibly downsampled)
        im1, im2: Original images
    Returns:
        im1_shift: Shifted image
        ux_interp: Interpolated x velocity
        uy_interp: Interpolated y velocity
    """
    m, n = im1.shape
    m_flow, n_flow = ux.shape
    
    # Resize velocity fields to match image dimensions
    ux_interp = ndimage.zoom(ux, (m/m_flow, n/n_flow))
    uy_interp = ndimage.zoom(uy, (m/m_flow, n/n_flow))
    
    # Create coordinate grids
    y, x = np.mgrid[0:m, 0:n]
    
    # Calculate shifted coordinates
    x_shifted = x + ux_interp
    y_shifted = y + uy_interp
    
    # Interpolate shifted image
    coords = np.array([y_shifted.flatten(), x_shifted.flatten()])
    im1_shift = ndimage.map_coordinates(im1, coords, order=1)
    im1_shift = im1_shift.reshape(m, n)
    
    return im1_shift, ux_interp, uy_interp
