from dataExtraction import read_bshc
import numpy as np
import matplotlib.pyplot as plt

filename = "data/Earth2014.TBI2014.degree10800.bshc"
C, S, max_degree = read_bshc(filename)

#np.savetxt("data/TBI_cos.csv", C, delimiter=",", fmt="%.8f")
#np.savetxt("data/TBI_sin.csv", S, delimiter=",", fmt="%.8f")

# x_C = np.repeat(np.arange(C.shape[0]), C.shape[1])
# #x_S = np.repeat(np.arange(S.shape[0]), S.shape[1])

# # Flatten y values (all elements of array)
# y_C = C.flatten()
# #y_S = S.flatten()

# plt.scatter(x_C, y_C)
# plt.boxplot(C.T)  # Transpose so each column is a group of samples
# plt.xlabel('Term Order')
# plt.ylabel('Coefficients')
# plt.title('Cosine Coefficients')
# plt.show()

# Compute average along the second index (mean of each row)
C_row_means = np.mean(C, axis=1)
C_row_stds  = np.std(C, axis=1)
S_row_means = np.mean(S, axis=1)
S_row_stds  = np.std(S, axis=1)

# X values = row indices
x = np.arange(C.shape[0])

# Plot
plt.plot(x, C_row_means, color='red', marker='o', linestyle='none', label='Cosine means')
plt.plot(x, C_row_stds, color='orange', label='Cosine stds')
plt.plot(x, S_row_means, color='blue', marker='o', linestyle='none', label='Sine means')
plt.plot(x, S_row_stds, color='purple', label='Sine stds')
plt.xlabel('Term Order')
plt.ylabel('Statistical Value')
plt.title('Spherical Harmonic Distributions')
plt.yscale('log')
plt.grid(True)
plt.legend()
plt.show()
