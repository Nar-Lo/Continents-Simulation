import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib
import sys

# Define and register the custom colormap
_terrain_exag_points = [
    (0.00, '#2e003e'),  # Deep, slightly purple ocean
    (0.25, '#0000ff'),  # Regular ocean blue
    (0.45, '#40e0d0'),  # Light turquoise
    (0.50, '#f2e394'),  # Sandy yellow (beaches)
    (0.60, "#3fb165"),  # Emerald green (grasslands)
    (0.70, "#1c801c"),  # Mature forest green
    (0.85, '#8b4513'),  # Taiga brown
    (0.95, '#eeeeee'),  # Snow line
    (1.00, '#eeeeee'),  # Snow cap
]
terrain_exag_cmap = LinearSegmentedColormap.from_list(
    name="terrain_exag", 
    colors=_terrain_exag_points
)
matplotlib.colormaps.register(name="terrain_exag", cmap=terrain_exag_cmap)

def main():
    filename = sys.argv[1]
    print(filename)
    result = np.loadtxt(filename, delimiter=',')
    nlat, nlon = result.shape

    # Latitude and longitude in degrees
    lat = np.linspace(90, -90, nlat)   # from north to south
    lon = np.linspace(0, 360, nlon, endpoint=False)

    # Convert to radians for spherical harmonic evaluation
    theta = np.radians(90 - lat)  # colatitude
    phi = np.radians(lon)

    # Create 2D meshgrid
    Theta, Phi = np.meshgrid(theta, phi, indexing='ij')  # shape (nlat, nlon)
    Lat, Long  = np.meshgrid(lat, lon, indexing='ij')

    # Shift longitude to [-180, 180)
    lon_shifted = (lon + 180) % 360 - 180

    # Roll data to shift longitude so 0° is centered
    # result_shifted = np.roll(result, shift=nlon // 2, axis=1)
    result_shifted = result
    
    plt.figure(figsize=(10, 5))
    plt.imshow(result_shifted, extent=[lon_shifted.min(), lon_shifted.max(), lat.min(), lat.max()],
            origin='upper', aspect='auto',cmap='terrain_exag')#, cmap='terrain_exag'
    plt.colorbar(label='Summed Harmonic Value')
    plt.xlabel('Longitude (degrees)')
    plt.ylabel('Latitude (degrees)')
    plt.title(filename)
    plt.savefig(filename.replace(".csv", ".png"))
    plt.show()

if __name__ == "__main__":
    main()