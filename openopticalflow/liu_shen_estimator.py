import numpy as np
from scipy import ndimage
from openopticalflow.generate_invmatrix import generate_invmatrix

def liu_shen_estimator(i0, i1, f, dx, dt, lambda_param, tol, maxnum, u0, v0):
    """
    Liu-Shen optical flow estimator

    Parameters:
        i0, i1: Input images
        f: Physical transport term
        dx, dt: Spatial and temporal steps
        lambda_param: Regularization parameter
        tol: Convergence tolerance
        maxnum: Maximum iterations
        u0, v0: Initial velocity estimates
    """
    # Define derivative kernels
    d = np.array([[0, -1, 0], [0, 0, 0], [0, 1, 0]]) / 2
    m = np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]]) / 4
    f_kernel = np.array([[0, 1, 0], [0, 0, 0], [0, 1, 0]])
    h = np.ones((3, 3))

    # Calculate derivatives and products
    iix = i0 * ndimage.convolve(i0, d/dx, mode='reflect')
    iiy = i0 * ndimage.convolve(i0, d.T/dx, mode='reflect')
    ii = i0 * i0
    ixt = i0 * ndimage.convolve((i1-i0)/dt-f, d/dx, mode='reflect')
    iyt = i0 * ndimage.convolve((i1-i0)/dt-f, d.T/dx, mode='reflect')

    # Initialize parameters
    k = 0
    total_error = 1e8
    u = u0.copy()
    v = v0.copy()
    r, c = i1.shape

    # Generate inverse matrix
    b11, b12, b22 = generate_invmatrix(i0, lambda_param, dx)
    error = []

    while total_error > tol and k < maxnum:
        bu = (2*iix * ndimage.convolve(u, d/dx, mode='reflect') +
              iix * ndimage.convolve(v, d.T/dx, mode='reflect') +
              iiy * ndimage.convolve(v, d/dx, mode='reflect') +
              ii * ndimage.convolve(u, f_kernel/(dx*dx), mode='reflect') +
              ii * ndimage.convolve(v, m/(dx*dx), mode='reflect') +
              lambda_param * ndimage.convolve(u, h/(dx*dx), mode='reflect') + ixt)

        bv = (iiy * ndimage.convolve(u, d/dx, mode='reflect') +
              iix * ndimage.convolve(u, d.T/dx, mode='reflect') +
              2*iiy * ndimage.convolve(v, d.T/dx, mode='reflect') +
              ii * ndimage.convolve(u, m/(dx*dx), mode='reflect') +
              ii * ndimage.convolve(v, f_kernel.T/(dx*dx), mode='reflect') +
              lambda_param * ndimage.convolve(v, h/(dx*dx), mode='reflect') + iyt)

        unew = -(b11*bu + b12*bv)
        vnew = -(b12*bu + b22*bv)

        total_error = (np.linalg.norm(unew-u, 'fro') +
                      np.linalg.norm(vnew-v, 'fro'))/(r*c)

        u = unew
        v = vnew
        error.append(total_error)
        k += 1

    return u, v, error
