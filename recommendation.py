
import numpy as np

# Generate 500 products with 20 features each
np.random.seed(42)
products = np.random.rand(500, 20)

# Normalize product vectors
norms = np.linalg.norm(products, axis=1, keepdims=True)
products = products / norms

# Calculate similarity using matrix multiplication
similarity = np.dot(products, products.T)

# Select a product viewed by the customer
product_id = 0

# Get similarity scores for that product
scores = similarity[product_id].copy()

# Exclude the selected product itself
scores[product_id] = -1

# Find the top 5 similar products
top_products = np.argsort(scores)[-5:][::-1]

print("Customer viewed Product:", product_id + 1)
print("\nRecommended Products:")

for product in top_products:
    print(
        "Product:", product + 1,
        "| Similarity Score:", round(scores[product], 3)
    )
