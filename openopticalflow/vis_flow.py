import numpy as np
import matplotlib.pyplot as plt

def vis_flow(vx, vy, gx=25, offset=0, mag=1, color='b'):
    """
    Visualize flow field using quiver plot with subsampling and normalized arrows.

    This function creates a quiver plot (vector field visualization) of the velocity field
    defined by vx and vy components. It subsamples the field to avoid overcrowding and
    normalizes the arrows for better visualization.

    Parameters:
        vx (numpy.ndarray): x-component of velocity field
        vy (numpy.ndarray): y-component of velocity field
        gx (int): Grid spacing parameter. Controls the density of arrows.
                 Higher values result in fewer arrows.
        offset (int): Offset for sampling grid points
        mag (float): Magnitude scaling factor. Higher values result in longer arrows.
        color (str): Color of the arrows

    Returns:
        matplotlib.figure.Figure: Figure object containing the quiver plot

    Notes:
        The function handles NaN values in the velocity field and automatically
        scales the arrows based on the maximum velocity magnitude.
    """
    sy, sx = vx.shape

    if gx == 0:
        jmp = 1
    else:
        jmp = max(1, sx // gx)

    indx = np.arange(offset, sx, jmp)
    indy = np.arange(offset, sy, jmp)

    X, Y = np.meshgrid(indx, indy)

    # Extract velocity components at the sampled points
    U = vx[indy][:, indx]
    V = vy[indy][:, indx]   # No sign reversal to match MATLAB behavior

    # Handle NaN values
    mask = ~(np.isnan(U) | np.isnan(V))
    if not mask.any():
        U[0,0] = 1
        V[0,0] = 0
        X[0,0] = 1
        Y[0,0] = 1
        mask[0,0] = True

    # Calculate the scale for arrow lengths
    # In matplotlib's quiver, smaller scale values make longer arrows
    # The mag parameter is used to adjust the arrow length (higher = longer arrows)

    # First, calculate the magnitudes of the velocity vectors
    vel_magnitude = np.sqrt(U[mask]**2 + V[mask]**2)

    # Calculate scale for arrow lengths
    if vel_magnitude.size > 0 and np.mean(vel_magnitude) > 0:
        # Calculate scale - smaller values make longer arrows
        # Divide by mag to make arrows longer when mag is larger
        scale_factor = 1.0 / (np.max(vel_magnitude) * mag)
    else:
        scale_factor = 1.0

    # Plot quiver with normalized arrows
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.quiver(X[mask], Y[mask], U[mask], V[mask],
              scale=scale_factor,    # Apply calculated scale
              scale_units='xy',      # Use xy coordinate system for scaling
              angles='xy',           # Use xy coordinate system for angles
              width=0.003,           # Normalized arrow width (smaller = thinner)
              headwidth=4,           # Head width relative to shaft (larger = wider)
              headlength=6,          # Head length relative to shaft (larger = longer)
              headaxislength=5,      # Head length at shaft intersection
              color=color,
              pivot='mid')

    ax.set_xlim(0, sx)
    ax.set_ylim(0, sy)

    return ax

