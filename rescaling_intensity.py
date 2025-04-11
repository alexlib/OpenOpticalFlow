import numpy as np

def rescaling_intensity(Im1, Im2, max_intensity_value):
    """
    Rescale the intensity of two images to a specified maximum value.
    
    Parameters:
        Im1: First input image
        Im2: Second input image
        max_intensity_value: Target maximum intensity value
    
    Returns:
        I1, I2: Rescaled images
    """
    # First image
    Imax1 = np.max(Im1)
    Imin1 = np.min(Im1)
    Im1a = (Im1 - Imin1) / (Imax1 - Imin1)
    
    # Second image
    Imax2 = np.max(Im2)
    Imin2 = np.min(Im2)
    Im2a = (Im2 - Imin2) / (Imax2 - Imin2)
    
    # Scale to max_intensity_value
    Im1 = Im1a * max_intensity_value
    Im2 = Im2a * max_intensity_value
    
    # Convert to float (equivalent to MATLAB's double)
    I1 = Im1.astype(float)
    I2 = Im2.astype(float)
    
    return I1, I2