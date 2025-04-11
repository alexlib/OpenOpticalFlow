import numpy as np
from vorticity_factor import vorticity_factor
import matplotlib.pyplot as plt

# Create sample velocity fields
x, y = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
vx = -y  # Example: rigid body rotation
vy = x

# Set conversion factors (example values)
factor_x = 0.001  # 1 mm/pixel
factor_y = 0.001  # 1 mm/pixel

# Calculate vorticity
omega = vorticity_factor(vx, vy, factor_x, factor_y)

# Visualize results
plt.figure(figsize=(12, 4))

plt.subplot(131)
plt.quiver(x[::5, ::5], y[::5, ::5], vx[::5, ::5], vy[::5, ::5])
plt.title('Velocity Field')
plt.axis('equal')

plt.subplot(132)
plt.imshow(omega, cmap='RdBu_r')
plt.colorbar(label='Vorticity')
plt.title('Vorticity Field')

plt.tight_layout()
plt.show()
