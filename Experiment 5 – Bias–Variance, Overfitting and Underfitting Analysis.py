import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

np.random.seed(42)

X = np.linspace(0, 10, 100).reshape(-1, 1)
y = np.sin(X).ravel() + np.random.normal(0, 0.2, 100)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

degrees = [1, 4, 15]

plt.figure(figsize=(12, 4))

for i, degree in enumerate(degrees):
    model = make_pipeline(
        PolynomialFeatures(degree),
        LinearRegression()
    )

    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    print("Degree:", degree)
    print("Training Error:", mean_squared_error(y_train, train_pred))
    print("Testing Error:", mean_squared_error(y_test, test_pred))

    plt.subplot(1, 3, i + 1)
    plt.scatter(X_train, y_train, color="blue")
    plt.plot(X, model.predict(X), color="red")
    plt.title(f"Polynomial Degree {degree}")

plt.tight_layout()
plt.show()
