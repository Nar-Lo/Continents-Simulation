from dataExtraction import read_bshc
import numpy as np
import matplotlib.pyplot as plt

filename = "data/Earth2014.TBI2014.degree10800.bshc"
C, S, max_degree = read_bshc(filename)

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
row_means = np.mean(C, axis=1)

# X values = row indices
x = np.arange(C.shape[0])

# Plot
plt.plot(x, row_means, marker='o', linestyle='none')
plt.xlabel('Row index (first index)')
plt.ylabel('Average of second index')
plt.title('Mean values by row index')
plt.yscale('log')
plt.grid(True)
plt.show()
