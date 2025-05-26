# from geoid_toolkit.read_topography_harmonics import read_topography_harmonics
# Ylms = read_topography_harmonics("data/Earth2014.BED2014.degree10800.bshc")

import numpy as np
from scipy.special import lpmv, factorial, gammaln, sph_harm
import matplotlib.pyplot as plt
import multiprocessing
import time

script_start_time = time.perf_counter()

def factorial_ratio(numerator, denominator):
    return np.exp(gammaln(numerator + 1) - gammaln(denominator + 1))


def schmidt_semi_normalization(n, m):
    delta = 1 if m == 0 else 0
    log_factor = (
        np.log(2 - delta)
        + np.log(2 * n + 1)
        + gammaln(n - m + 1)
        - gammaln(n + m + 1)
    ) / 2.0
    return np.exp(log_factor)


def read_bshc(filename):
    data = np.fromfile(filename, dtype='<f8')
    max_degree = int(data[1])

    # Cosine, Sine Arrays
    C = np.zeros((max_degree + 1, max_degree + 1))
    S = np.zeros((max_degree + 1, max_degree + 1))

    # Populate Cosines
    index = 2
    for n in range(max_degree + 1):
        for m in range(n + 1):
            C[n, m] = data[index]
            index += 1

    # Populate Sines
    for n in range(max_degree + 1):
        for m in range(n + 1):
            S[n, m] = data[index]
            index += 1

    return C, S, max_degree

def real_sph_harm(n, m, theta, phi):
    """
    Compute real spherical harmonics at (theta, phi).
    theta: colatitude in radians [0, pi]
    phi: longitude in radians [0, 2*pi]

    Returns:
        Real spherical harmonic value R_n^m(theta, phi)
    """
    if m < 0:
        return np.sqrt(2) * (-1)**m * sph_harm(abs(m), n, phi, theta).imag
    elif m == 0:
        return sph_harm(0, n, phi, theta).real
    else:
        return np.sqrt(2) * (-1)**m * sph_harm(m, n, phi, theta).real

def sum_harmonics(lat_grid, long_grid, S_Coeffs, C_Coeffs, max_degree):
    """
    Compute real spherical harmonic sum over a lat-lon grid.
    lat_grid, long_grid: 2D arrays in radians
    S_Coeffs, C_Coeffs: (max_degree+1, max_degree+1) coefficient arrays
    Returns: 2D array same shape as lat_grid
    """
    V = np.zeros_like(lat_grid)
    theta = np.radians(90-lat_grid)  # colatitude in radians
    cos_theta = np.cos(theta)
    phi = np.radians(long_grid)

    for n in range(max_degree + 1):
        print("===========================")
        print(f"Begun analysis of step {n}")
        start_time_analysis = time.perf_counter()
        # for m in range(n + 1):
        #     # Harmonics at each longitude
        #     cos_mphi = np.cos(m * phi)
        #     sin_mphi = np.sin(m * phi)

        #     # # Legendre Pnm evaluated at each latitude (1D)
        #     # Pnm = lpmv(m, n, cos_theta) * schmidt_semi_normalization(n, m)

        #     # # Accumulate sum
        #     # V += Pnm * (C_Coeffs[n, m] * cos_mphi + S_Coeffs[n, m] * sin_mphi)

        for m in range(-n, n + 1):
            # Compute the real spherical harmonic for this degree/order
            Ynm = real_sph_harm(n, m, theta, phi)

            # Get coefficients from C and S arrays
            if m < 0:
                # For negative m, coefficients relate to S and C with phase:
                # Real SH for negative m = sqrt(2)*(-1)^m * Im(Y_n^{|m|})
                # So contribution is: S[n, |m|] * Ynm (since Ynm already imaginary part)
                coeff = S_Coeffs[n, -m]
            elif m == 0:
                coeff = C_Coeffs[n, 0]
            else:  # m > 0
                coeff = C_Coeffs[n, m]

            V += coeff * Ynm       

        end_time_analysis = time.perf_counter()
        elapsed_time_analysis = end_time_analysis - start_time_analysis
        print(f"Time elapsed: {elapsed_time_analysis:.2} seconds")
    return V


filename = "data/Earth2014.TBI2014.degree10800.bshc"
C, S, max_degree = read_bshc(filename)

# Define grid resolution
nlat = 180  # number of latitude points
nlon = 360  # number of longitude points

# Latitude and longitude in degrees
lat = np.linspace(90, -90, nlat)   # from north to south
lon = np.linspace(0, 360, nlon, endpoint=False)

# Convert to radians for spherical harmonic evaluation
theta = np.radians(90 - lat)  # colatitude
phi = np.radians(lon)

# Create 2D meshgrid
Theta, Phi = np.meshgrid(theta, phi, indexing='ij')  # shape (nlat, nlon)
Lat, Long  = np.meshgrid(lat, lon, indexing='ij')

result = sum_harmonics(Lat, Long, S, C, 85)

# Shift longitude to [-180, 180)
lon_shifted = (lon + 180) % 360 - 180

# Roll data to shift longitude so 0° is centered
result_shifted = np.roll(result, shift=nlon // 2, axis=1)

plt.figure(figsize=(10, 5))
plt.imshow(result_shifted, extent=[lon_shifted.min(), lon_shifted.max(), lat.min(), lat.max()],
           origin='upper', aspect='auto', cmap='terrain')
plt.colorbar(label='Summed Harmonic Value')
plt.xlabel('Longitude (degrees)')
plt.ylabel('Latitude (degrees)')
plt.title('Spherical Harmonic Synthesis Result')

script_end_time = time.perf_counter()
total_runtime = script_end_time - script_start_time
print("=========================================")
print(f"Total runtime of script was {total_runtime:.2f} seconds")

plt.show()