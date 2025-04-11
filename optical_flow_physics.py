import numpy as np
from scipy import ndimage
from horn_schunck_estimator import horn_schunck_estimator
from liu_shen_estimator import liu_shen_estimator

def optical_flow_physics(I1, I2, lambda_1, lambda_2):
    """
    Direct port of MATLAB OpticalFlowPhysics_fun.m
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
    Compute Laplacian matching MATLAB implementation
    """
    kernel = np.array([[0, 1, 0],
                      [1, -4, 1],
                      [0, 1, 0]]) / (dx * dx)
    return ndimage.convolve(img, kernel, mode='reflect')

def vorticity(ux, uy):
    """
    Compute vorticity matching MATLAB implementation
    """
    dx = 1
    dy = 1
    
    duydx = ndimage.convolve(uy, np.array([[-1, 0, 1]]) / (2 * dx), mode='reflect')
    duxdy = ndimage.convolve(ux, np.array([[-1], [0], [1]]) / (2 * dy), mode='reflect')
    
    return duydx - duxdy
