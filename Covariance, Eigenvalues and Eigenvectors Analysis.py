
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Sample dataset
data = {
    "Hours": [1, 2, 3, 4, 5],
    "Marks": [10, 20, 30, 40, 50]
}

df = pd.DataFrame(data)
print("Dataset:")
print(df)

# Calculate covariance matrix
cov_matrix = df.cov()
print("\nCovariance Matrix:")
print(cov_matrix)

# Calculate eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(
    cov_matrix.to_numpy()
)

print("\nEigenvalues:")
print(eigenvalues)

print("\nEigenvectors:")
print(eigenvectors)

# Visualization
plt.scatter(df["Hours"], df["Marks"])
plt.plot(df["Hours"], df["Marks"])
plt.title("Hours Studied vs Marks")
plt.xlabel("Hours Studied")
plt.ylabel("Marks")
plt.grid(True)
plt.show()
