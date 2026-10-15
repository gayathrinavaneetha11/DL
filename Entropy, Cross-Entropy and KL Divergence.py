
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import entropy

# Actual and predicted probability distributions
P = np.array([0.7, 0.3])
Q = np.array([0.5, 0.5])

# Entropy
H_P = entropy(P)

# Cross-entropy
cross_entropy = -np.sum(P * np.log(Q))

# KL divergence
kl_divergence = entropy(P, Q)

print("Entropy:", round(H_P, 4))
print("Cross-Entropy:", round(cross_entropy, 4))
print("KL Divergence:", round(kl_divergence, 4))

# Plot probability distributions
labels = ["Class 1", "Class 2"]
x = np.arange(len(labels))
width = 0.35

plt.bar(x - width / 2, P, width, label="Actual P")
plt.bar(x + width / 2, Q, width, label="Predicted Q")

plt.title("Actual vs Predicted Probabilities")
plt.xlabel("Classes")
plt.ylabel("Probability")
plt.xticks(x, labels)
plt.legend()
plt.grid(axis="y")

plt.savefig("experiment4.png")
plt.close()
print("Graph saved as experiment4.png")
