import numpy as np
import plotly.express as px
import time
from numba import jit

# Vectorized GD without augmented matrix: handle bias separately
def predict(theta, bias, X):
    return X.dot(theta) + bias

def J_squared_residual(theta, bias, X, y):
    h = predict(theta, bias, X)
    return ((h - y)**2).sum()

def gradient_J_squared_residual(theta, bias, X, y):
    h = predict(theta, bias, X)
    error = (h - y)                         # shape (m,1)
    grad_theta = X.T.dot(error)            # shape (n_features,1)
    grad_bias = np.sum(error)              # scalar
    return grad_theta, grad_bias

@jit(nopython=True)
def sgd(X, y, learning_rate=0.1, epochs=1000, batch_size=2, random_seed=None, bias=0.0):
    # X: shape (m, n_features)
    # y: shape (m, 1) or (m,)
    if random_seed is not None:
        np.random.seed(random_seed)
    m, n_features = X.shape
    # initialize weights and bias
    theta = np.random.randn(n_features, 1)  # shape (n_features,1)
    cost_history = []

    for epoch in range(epochs):
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for i in range(0, m, batch_size):
            X_batch = X_shuffled[i:i + batch_size]
            y_batch = y_shuffled[i:i + batch_size]

            # ensure shapes: X_batch (b, n), theta (n,1) -> preds (b,1)
            preds = X_batch.dot(theta) + bias
            error = preds - y_batch.reshape(-1, 1)  # make column vector

            grad_theta = (2.0 / X_batch.shape[0]) * (X_batch.T.dot(error))  # (n,1)
            grad_bias = (2.0 / X_batch.shape[0]) * error.sum()              # scalar

            theta = theta - learning_rate * grad_theta
            bias = bias - learning_rate * grad_bias

        # optionally compute cost on whole set (mean squared error)
        preds_full = X.dot(theta) + bias
        cost = np.mean((preds_full - y.reshape(-1, 1)) ** 2)
        cost_history.append(cost)

        if epoch % 100 == 0:
            print("Epoch: ", epoch , " Cost: ", float(cost))

    return theta, bias, cost_history

# the dataset: plain features (no column of ones) and targets
X = np.array([[0.5], [1.0], [2.0]])    # shape (m, n_features)
y = np.array([[1.0], [1.5], [2.5]])    # shape (m, 1)
m, n_features = X.shape

# variables we need
theta = np.zeros((n_features, 1))    # weight vector (n_features,1)
bias = 0.0                           # scalar bias
learning_rate = 0.1

# run SGD
n_epochs = 20
batch_size = 1
theta, bias, j_history = sgd(X, y, learning_rate, n_epochs, batch_size, bias=bias)

print("theta shape:", theta.shape)

# append the final result.
j = J_squared_residual(theta, bias, X, y)
j_history.append(j)
print("The L2 error (sum squared residuals) is: {:.6f}".format(j))

# find the L1 error.
y_pred = predict(theta, bias, X)
l1_error = np.abs(y_pred - y).sum()
print("The L1 error is: {:.6f}".format(l1_error))

# Find the R^2 
u = ((y - y_pred)**2).sum()
v = ((y - y.mean())**2).sum()
print("R^2: {:.6f}".format(1 - (u/v)))

# plot the result
fig = px.line(j_history, title="J(theta) - Loss History")
fig.write_image("Task4_GD_L2_Loss.png")