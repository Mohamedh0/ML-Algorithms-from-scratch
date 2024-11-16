import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

class LogisticRegression:
    def __init__(self, learning_rate=0.001, num_iterations=1000):
        """
        Logistic Regression Model using Gradient Descent
        """
        self.lr = learning_rate
        self.it = num_iterations
        self.weights = None
        self.bias = None
        self.losses = []

    def sigmoid(self, x):
        """
        Sigmoid activation function with numerical stability
        """
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def compute_loss(self, y_true, y_pred):
        """
        Binary Cross-Entropy Loss
        """
        n_samples = len(y_true)
        loss = -(1 / n_samples) * np.sum(
            y_true * np.log(y_pred + 1e-15) + (1 - y_true) * np.log(1 - y_pred + 1e-15)
        )
        return loss

    def fit(self, X, y):
        """
        Train the model using gradient descent
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient descent
        for i in range(self.it):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear_model)

            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            # Calculate and store loss
            loss = self.compute_loss(y, y_pred)
            self.losses.append(loss)

            if i % 100 == 0:
                print(f"Iteration {i}, Loss: {loss}")

    def predict(self, X):
        """
        Predict binary labels (0 or 1)
        """
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(linear_model)
        return [1 if i > 0.5 else 0 for i in y_pred]

    def accuracy(self, y_true, y_pred):
        """
        Calculate accuracy of predictions
        """
        return np.sum(y_true == y_pred) / len(y_true)

    def plot_loss(self):
        """
        Plot the loss curve
        """
        plt.plot(range(self.it), self.losses, label="Loss")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.title("Loss Curve")
        plt.legend()
        plt.show()

# Load Dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Normalize features for better convergence
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression(learning_rate=0.01, num_iterations=1000)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = model.accuracy(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.2f}")

# Plot loss curve
model.plot_loss()
