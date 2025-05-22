from scipy.special import sph_harm
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Function to generate random coefficient for each harmonic
def coeff(num, l, m):
    val = num * (1 / (l + 1)**1.5) * np.random.normal(loc=0.0, scale=0.5)
    return val

# --- Planet geometry ---
radius = 5000e3       # radius of planet [m]
baseThickness = 10e3  # Base thickness number (core radius)

# --- Simulation Parameters ---
N = 35                            # max spherical harmonic degree
theta = np.linspace(0, np.pi, 100)
phi = np.linspace(0, 2 * np.pi, 100)
theta, phi = np.meshgrid(theta, phi)

Y_sum = np.zeros(theta.shape)

# --- Sum over all spherical harmonics ---
for l in range(N + 1):
    for m in range(-l, l + 1):
        Y = sph_harm(m, l, phi, theta)
        if m < 0:
            Y_real = np.sqrt(2) * (-1)**m * Y.imag
        elif m == 0:
            Y_real = Y.real
        else:
            Y_real = np.sqrt(2) * (-1)**m * Y.real
        Y_sum += coeff(baseThickness, l, m) * Y_real

# --- 2D Plot of the spherical harmonic field ---
plt.figure(figsize=(8, 4))
cmap = plt.get_cmap('terrain')
plt.pcolormesh(phi, theta, Y_sum.real, shading='auto', cmap=cmap)
plt.title(f"Real part of spherical harmonic sum up to l = {N}")
plt.xlabel('φ')
plt.ylabel('θ')
plt.colorbar(label='Σ Real Yₗᵐ')
plt.tight_layout()
plt.show()

# --- Convert to Cartesian coordinates ---
x = np.sin(theta) * np.cos(phi)
y = np.sin(theta) * np.sin(phi)
z = np.cos(theta)

# --- Apply harmonic deformation to the sphere radius ---
r = radius + Y_sum.real * 75  # scaled perturbation
x_s = r * x
y_s = r * y
z_s = r * z

# --- Create 3D plot ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Deformed terrain-colored surface
ax.plot_surface(
    x_s, y_s, z_s,
    facecolors=plt.cm.terrain((Y_sum.real - Y_sum.min()) / (Y_sum.max() - Y_sum.min())),
    rstride=1, cstride=1, antialiased=True, linewidth=0
)

# --- Transparent blue sphere (core) centered at origin ---
theta_core = np.linspace(0, np.pi, 50)
phi_core = np.linspace(0, 2 * np.pi, 50)
phi_core, theta_core = np.meshgrid(phi_core, theta_core)

x_core = radius * np.sin(theta_core) * np.cos(phi_core)
y_core = radius * np.sin(theta_core) * np.sin(phi_core)
z_core = radius * np.cos(theta_core)

ax.plot_surface(
    x_core, y_core, z_core,
    color='blue', alpha=0.7, edgecolor='none'
)

# --- Final plot settings ---
ax.set_title("Deformed Sphere with Transparent Core")
ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio
ax.axis('off')
plt.tight_layout()
plt.show()



# from scipy.special import sph_harm
# import scipy.stats
# import numpy as np
# import matplotlib.pyplot as plt


# def coeff(num, l, m):
#     val = num*(1/(l+1)) * np.random.normal(loc=0.0, scale=0.5/(np.abs(m)+1))
#     return val
# ''' Planet geometry '''
# radius = 5000e3 # radius of planet [m]
# baseThickness = 10e3    # Base thickness number


# ''' Simulation Parameters '''
# N = 10                 # max spherical harmonic
# gridNum = 1000         # num points in grid each dimension
# theta = np.linspace(0, np.pi, 100)
# phi = np.linspace(0, 2*np.pi, 100)
# theta, phi = np.meshgrid(theta, phi)

# Y_sum = np.zeros(theta.shape)

# # Sum over all l and m
# for l in range(N + 1):
#     for m in range(-l, l + 1):
#         Y = sph_harm(m, l, phi, theta)
#         if m < 0:
#             Y_real = np.sqrt(2) * (-1)**m * Y.imag
#         elif m == 0:
#             Y_real = Y.real
#         else:
#             Y_real = np.sqrt(2) * (-1)**m * Y.real
#         Y_sum += coeff(baseThickness, l, m) * Y_real


# # Optional: Plot real part
# plt.figure(figsize=(8, 4))
# plt.pcolormesh(phi, theta, Y_sum.real, shading='auto', cmap='viridis')
# plt.title("Real part of sum of spherical harmonics up to l = {}".format(N))
# plt.xlabel('φ')
# plt.ylabel('θ')
# plt.colorbar(label='Σ Real Yₗᵐ')
# plt.show()

# from scipy.special import sph_harm
# import scipy.stats
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# def coeff(num, l, m):
#     val = num * (1 / (l + 1)**(1.5)) * np.random.normal(loc=0.0, scale=0.5)
#     return val

# ''' Planet geometry '''
# radius = 5000e3       # radius of planet [m]
# baseThickness = 10e3  # Base thickness number

# ''' Simulation Parameters '''
# N = 35                 # max spherical harmonic
# gridNum = 5000         # number of grid points
# theta = np.linspace(0, np.pi, 100)
# phi = np.linspace(0, 2 * np.pi, 100)
# theta, phi = np.meshgrid(theta, phi)

# Y_sum = np.zeros(theta.shape)

# # Sum over all l and m
# for l in range(N + 1):
#     for m in range(-l, l + 1):
#         Y = sph_harm(m, l, phi, theta)
#         if m < 0:
#             Y_real = np.sqrt(2) * (-1)**m * Y.imag
#         elif m == 0:
#             Y_real = Y.real
#         else:
#             Y_real = np.sqrt(2) * (-1)**m * Y.real
#         Y_sum += coeff(baseThickness, l, m) * Y_real

# # Plot 2D colormap of spherical harmonic sum
# plt.figure(figsize=(8, 4))
# cmap = plt.get_cmap('terrain')
# plt.pcolormesh(phi, theta, Y_sum.real, shading='auto', cmap=cmap)
# plt.title("Real part of sum of spherical harmonics up to l = {}".format(N))
# plt.xlabel('φ')
# plt.ylabel('θ')
# plt.colorbar(label='Σ Real Yₗᵐ')
# plt.show()

# # --- 3D Sphere Plot with Harmonic Deformation ---

# # Convert to Cartesian coordinates
# x = np.sin(theta) * np.cos(phi)
# y = np.sin(theta) * np.sin(phi)
# z = np.cos(theta)

# # Normalize and scale radius by harmonics
# r = radius + Y_sum.real*75  # scale thickness perturbation to meters



# # Apply deformation
# x_s = r * x
# y_s = r * y
# z_s = r * z

# # Plot the deformed sphere
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')
# surface = ax.plot_surface(x_s, y_s, z_s, facecolors=plt.cm.terrain((Y_sum.real - Y_sum.min()) / (Y_sum.max() - Y_sum.min())),
#                           rstride=1, cstride=1, antialiased=True, linewidth=0)

# # Add transparent blue sphere centered at origin (same as main sphere)
# theta_overlay = np.linspace(0, np.pi, 50)
# phi_overlay = np.linspace(0, 2 * np.pi, 50)
# phi_overlay, theta_overlay = np.meshgrid(phi_overlay, theta_overlay)

# r_overlay = baseThickness
# x_overlay = r_overlay * np.sin(theta_overlay) * np.cos(phi_overlay)
# y_overlay = r_overlay * np.sin(theta_overlay) * np.sin(phi_overlay)
# z_overlay = r_overlay * np.cos(theta_overlay)

# # Plot transparent blue sphere centered at origin
# ax.plot_surface(
#     x_overlay, y_overlay, z_overlay,
#     color='black', alpha=0.5, edgecolor='none'
# )


# ax.set_title("Deformed Sphere from Real Spherical Harmonics")
# ax.set_box_aspect([1, 1, 1])
# ax.axis('off')
# plt.show()
