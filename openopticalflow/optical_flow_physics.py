import numpy as np
from scipy import ndimage
from openopticalflow.horn_schunck_estimator import horn_schunck_estimator
from openopticalflow.liu_shen_estimator import liu_shen_estimator

def optical_flow_physics(I1, I2, lambda_1, lambda_2):
    """
    Calculate optical flow using a physics-based approach.

    This function implements a two-step optical flow estimation:
    1. First, it uses the Horn-Schunck method to get an initial estimate
    2. Then, it refines the estimate using the Liu-Shen method which incorporates
       physical constraints

    Parameters:
        I1 (numpy.ndarray): First image
        I2 (numpy.ndarray): Second image
        lambda_1 (float): Regularization parameter for Horn-Schunck method
        lambda_2 (float): Regularization parameter for Liu-Shen method

    Returns:
        tuple: (u, v, vor, ux_horn, uy_horn, error1)
            - u: y-component of velocity field
            - v: x-component of velocity field
            - vor: vorticity field
            - ux_horn: x-component from Horn-Schunck
            - uy_horn: y-component from Horn-Schunck
            - error1: convergence error from Liu-Shen method

    Notes:
        The coordinate system follows image conventions where (0,0) is at the top-left.
        The velocity components (u,v) correspond to (y,x) directions respectively.
    """
    # Horn's solution as initial approximation of u and v
    # Matches MATLAB exactly:
    D1 = np.array([[0, 0, 0],
                   [0, -1, -1],
                   [0, 1, 1]]) / 2

    F1 = np.array([[0, 0, 0],
                   [0, 1, 1],
                   [0, 1, 1]]) / 4

    # Calculate derivatives exactly as in MATLAB
    Ix = ndimage.convolve((I1 + I2)/2, D1, mode='reflect')
    Iy = ndimage.convolve((I1 + I2)/2, D1.T, mode='reflect')
    It = ndimage.convolve(I2 - I1, F1, mode='reflect')

    # Horn-Schunck parameters matching MATLAB
    maxnum_1 = 500
    tol_1 = 1e-12

    # Get initial Horn-Schunck estimate
    uy_horn, ux_horn = horn_schunck_estimator(Ix, Iy, It, lambda_1, tol_1, maxnum_1)

    # Liu-Shen parameters matching MATLAB
    Dm = 0  # Physical diffusion term
    f = Dm * laplacian(I1, 1)

    maxnum = 60
    tol = 1e-8
    dx = 1
    dt = 1

    # Get Liu-Shen estimate
    v, u, error1 = liu_shen_estimator(I1, I2, f, dx, dt, lambda_2, tol, maxnum,
                                     uy_horn, ux_horn)

    # Match MATLAB output convention
    vor = vorticity(u, v)

    return u, v, vor, ux_horn, uy_horn, error1

def laplacian(img, dx):
    """
    Compute the Laplacian of an image.

    This function calculates the discrete Laplacian operator using a 3x3 kernel,
    which approximates the second spatial derivative.

    Parameters:
        img (numpy.ndarray): Input image
        dx (float): Spatial step size

    Returns:
        numpy.ndarray: Laplacian of the input image
    """
    kernel = np.array([[0, 1, 0],
                      [1, -4, 1],
                      [0, 1, 0]]) / (dx * dx)
    return ndimage.convolve(img, kernel, mode='reflect')

def vorticity(ux, uy):
    """
    Compute the vorticity field from velocity components.

    Vorticity is defined as the curl of the velocity field and represents
    the local rotation in the flow. For a 2D flow, it's calculated as
    the difference between the y-derivative of the x-velocity and
    the x-derivative of the y-velocity.

    Parameters:
        ux (numpy.ndarray): x-component of velocity field
        uy (numpy.ndarray): y-component of velocity field

    Returns:
        numpy.ndarray: Vorticity field
    """
    dx = 1
    dy = 1

    duydx = ndimage.convolve(uy, np.array([[-1, 0, 1]]) / (2 * dx), mode='reflect')
    duxdy = ndimage.convolve(ux, np.array([[-1], [0], [1]]) / (2 * dy), mode='reflect')

    return duydx - duxdy
