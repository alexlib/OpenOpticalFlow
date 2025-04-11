import numpy as np
from scipy import ndimage

def horn_schunck_estimator(Ix, Iy, It, alpha, tol, maxiter):
    """
    Horn-Schunck optical flow estimation algorithm.

    This implementation follows the classic Horn-Schunck method for estimating optical flow
    between two images. It solves the optical flow constraint equation with a smoothness
    regularization term.

    Parameters:
        Ix (numpy.ndarray): Spatial derivative of the image in the x direction
        Iy (numpy.ndarray): Spatial derivative of the image in the y direction
        It (numpy.ndarray): Temporal derivative of the image
        alpha (float): Regularization parameter. Higher values produce smoother flow fields
        tol (float): Convergence tolerance. The algorithm stops when the maximum change
                    in the flow field is less than this value
        maxiter (int): Maximum number of iterations

    Returns:
        tuple: (u, v) - Optical flow components in x and y directions

    References:
        Horn, B. K., & Schunck, B. G. (1981). Determining optical flow.
        Artificial intelligence, 17(1-3), 185-203.
    """
    # Initialize velocities
    u = np.zeros_like(Ix)
    v = np.zeros_like(Ix)

    # Laplacian kernel for averaging
    kernel = np.array([[0, 0.25, 0],
                      [0.25, 0, 0.25],
                      [0, 0.25, 0]])

    for _ in range(maxiter):
        # Calculate averages
        uAvg = ndimage.convolve(u, kernel, mode='reflect')
        vAvg = ndimage.convolve(v, kernel, mode='reflect')

        # Calculate update according to Horn-Schunck equations
        den = alpha + Ix*Ix + Iy*Iy

        un = uAvg - Ix * (Ix*uAvg + Iy*vAvg + It) / den
        vn = vAvg - Iy * (Ix*uAvg + Iy*vAvg + It) / den

        # Check convergence
        du = np.abs(un - u).max()
        dv = np.abs(vn - v).max()

        u = un
        v = vn

        if max(du, dv) < tol:
            break

    return u, v
