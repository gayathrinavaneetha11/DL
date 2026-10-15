
import numpy as np
import matplotlib.pyplot as plt

# Create two vectors
a = np.array([2, 4, 6])
b = np.array([1, 3, 5])

# Vector operations
print("Vector Addition:", a + b)
print("Vector Subtraction:", a - b)
print("Dot Product:", np.dot(a, b))

# Create two matrices
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("Matrix Addition:\n", A + B)
print("Matrix Multiplication:\n", np.dot(A, B))

# Visualization
plt.plot(a, label="Vector A", marker="o")
plt.plot(b, label="Vector B", marker="s")
plt.title("Vector Operations")
plt.xlabel("Index")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.show()
