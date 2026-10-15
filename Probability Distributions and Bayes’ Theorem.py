
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

# Binomial probability distribution
n = 10
p = 0.5

x = np.arange(0, n + 1)
probabilities = binom.pmf(x, n, p)

print("Binomial Probability Distribution:")
for i in range(len(x)):
    print(f"P(X={x[i]}) = {probabilities[i]:.4f}")

# Bayes' theorem example
# P(Disease) = 0.01
# P(Positive | Disease) = 0.90
# P(Positive | No Disease) = 0.05

p_disease = 0.01
p_positive_given_disease = 0.90
p_positive_given_no_disease = 0.05

p_positive = (
    p_positive_given_disease * p_disease
    + p_positive_given_no_disease * (1 - p_disease)
)

p_disease_given_positive = (
    p_positive_given_disease * p_disease / p_positive
)

print("\nBayes' Theorem:")
print("P(Disease | Positive) =",
      round(p_disease_given_positive, 4))

# Plot the distribution
plt.bar(x, probabilities)
plt.title("Binomial Probability Distribution")
plt.xlabel("Number of Successes")
plt.ylabel("Probability")
plt.grid(axis="y")
plt.show()
