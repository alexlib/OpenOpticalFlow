import numpy as np
from scipy import ndimage

def correction_illumination(im1, im2, window_shifting, size_average):
    """
    Correct illumination differences between two images.
    
    Parameters:
        im1, im2: Input images
        window_shifting: List/array [x3, x4, y3, y4] defining the window for mean intensity calculation
        size_average: Size of averaging window (if 0, no local illumination correction is applied)
    
    Returns:
        i1, i2: Corrected images
    """
    # Convert to float if needed
    im1 = im1.astype(float)
    im2 = im2.astype(float)
    
    i1 = im1.copy()
    i2 = im2.copy()
    
    # Extract window coordinates
    x3, x4, y3, y4 = window_shifting
    
    # Adjust overall illumination change
    i1_mean = np.mean(i1[y3:y4, x3:x4])
    i2_mean = np.mean(i2[y3:y4, x3:x4])
    r12 = i1_mean / i2_mean
    i2 = r12 * i2
    
    # Normalize intensity for i2 to eliminate local illumination changes
    if size_average > 0:
        # Create averaging kernel
        kernel = np.ones((size_average, size_average)) / (size_average * size_average)
        
        # Apply filtering and calculate difference
        i1_filtered = ndimage.convolve(i1, kernel, mode='reflect')
        i2_filtered = ndimage.convolve(i2, kernel, mode='reflect')
        i12f = i1_filtered - i2_filtered
        
        # Adjust i2
        i2 = i2 + i12f
    
    return i1, i2