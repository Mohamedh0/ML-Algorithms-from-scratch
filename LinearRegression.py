# Libraries
import numpy as np    
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression      
import matplotlib.pyplot as plt       

class LinearRegression:
    def __init__(self, learning_rate=0.001, num_iterations=1000):
        self.lr = learning_rate
        self.it = num_iterations
        self.weights = None
        self.bias = None
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        # gradient descent
        for _ in range(self.it):
            y_pred = np.dot(X, self.weights) + self.bias
            # compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            # update parameters
            self.weights -= self.lr * dw     
            self.bias -= self.lr * db     
            
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
    
    def mse(self, y, y_pred):
        return np.mean((y - y_pred) ** 2)

# Generate synthetic dataset
X, y = make_regression(n_samples=1000, n_features=1, noise=20, random_state=5)
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

print(f"Training MSE: {train_mse:.2f}")
print(f"Testing MSE: {test_mse:.2f}")

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
