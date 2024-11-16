# Libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
import pickle

class LinearRegression:
    def __init__(self, learning_rate=0.001, num_iterations=1000):
        """
        Linear Regression using Gradient Descent.
        """
        self.lr = learning_rate
        self.it = num_iterations
        self.weights = None
        self.bias = None
        self.losses = []

    def fit(self, X, y):
        """
        Train the model using gradient descent.
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in range(self.it):
            y_pred = np.dot(X, self.weights) + self.bias

            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            # Calculate and store MSE loss
            loss = self.mse(y, y_pred)
            self.losses.append(loss)

            if i % 100 == 0:
                print(f"Iteration {i}, Loss: {loss:.4f}")

    def predict(self, X):
        """
        Make predictions using the trained model.
        """
        return np.dot(X, self.weights) + self.bias

    def mse(self, y, y_pred):
        """
        Mean Squared Error (MSE) metric.
        """
        return np.mean((y - y_pred) ** 2)

    def mae(self, y, y_pred):
        """
        Mean Absolute Error (MAE) metric.
        """
        return np.mean(np.abs(y - y_pred))

    def r_squared(self, y, y_pred):
        """
        R-squared metric.
        """
        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_residual = np.sum((y - y_pred) ** 2)
        return 1 - (ss_residual / ss_total)

    def plot_learning_curve(self):
        """
        Plot the learning curve (loss vs. iterations).
        """
        plt.figure(figsize=(8, 5))
        plt.plot(range(len(self.losses)), self.losses, label="Loss")
        plt.xlabel("Iterations")
        plt.ylabel("MSE Loss")
        plt.title("Learning Curve")
        plt.legend()
        plt.show()

    def save_model(self, filename):
        """
        Save the trained model to a file.
        """
        with open(filename, 'wb') as file:
            pickle.dump(self, file)
        print(f"Model saved to {filename}")

    @staticmethod
    def load_model(filename):
        """
        Load a trained model from a file.
        """
        with open(filename, 'rb') as file:
            model = pickle.load(file)
        print(f"Model loaded from {filename}")
        return model

# Generate synthetic dataset
X, y = make_regression(n_samples=1000, n_features=1, noise=20, random_state=5)

# Normalize features
X = (X - np.mean(X)) / np.std(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the regressor
regressor = LinearRegression(learning_rate=0.01, num_iterations=1000)
regressor.fit(X_train, y_train)

# Predictions
y_pred_train = regressor.predict(X_train)
y_pred_test = regressor.predict(X_test)

# Metrics
train_mse = regressor.mse(y_train, y_pred_train)
test_mse = regressor.mse(y_test, y_pred_test)
train_r2 = regressor.r_squared(y_train, y_pred_train)
test_r2 = regressor.r_squared(y_test, y_pred_test)

print(f"Training MSE: {train_mse:.2f}")
print(f"Testing MSE: {test_mse:.2f}")
print(f"Training R-squared: {train_r2:.2f}")
print(f"Testing R-squared: {test_r2:.2f}")

# Save and load the model
regressor.save_model("linear_regression_model.pkl")
loaded_model = LinearRegression.load_model("linear_regression_model.pkl")

# Visualization of predictions
plt.figure(figsize=(12, 6))

# Training data visualization
plt.scatter(X_train, y_train, color="blue", label="Training Data", alpha=0.6)
plt.plot(X_train, y_pred_train, color="black", linewidth=2, label="Model Prediction (Train)")

# Test data visualization
plt.scatter(X_test, y_test, color="red", label="Testing Data", alpha=0.6)
plt.plot(X_test, y_pred_test, color="green", linewidth=2, label="Model Prediction (Test)")

plt.title("Linear Regression Fit")
plt.xlabel("Feature (X)")
plt.ylabel("Target (y)")
plt.legend()
plt.show()

# Plot learning curve
regressor.plot_learning_curve()
