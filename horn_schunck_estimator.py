import numpy as np
from scipy import ndimage

def horn_schunck_estimator(Ix, Iy, It, alpha, tol, maxiter):
    """
    Horn-Schunck optical flow estimation matching MATLAB implementation
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
