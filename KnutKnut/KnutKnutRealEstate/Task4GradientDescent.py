import numpy as np
import plotly.express as px

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

# the dataset: plain features (no column of ones) and targets
X = np.array([[0.5], [1.0], [2.0]])    # shape (m, n_features)
y = np.array([[1.0], [1.5], [2.5]])    # shape (m, 1)
m, n_features = X.shape

# variables we need
theta = np.zeros((n_features, 1))    # weight vector (n_features,1)
bias = 0.0                           # scalar bias
learning_rate = 0.1

# run GD
j_history = []
n_iters = 10
for it in range(n_iters):
    j = J_squared_residual(theta, bias, X, y)
    j_history.append(j)

    grad_theta, grad_bias = gradient_J_squared_residual(theta, bias, X, y)

    # mean gradients
    grad_theta = (1/m) * grad_theta
    grad_bias = (1/m) * grad_bias

    # parameter updates
    theta = theta - learning_rate * grad_theta
    bias = bias - learning_rate * grad_bias

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
fig = px.line(x=list(range(n_iters+1)), y=j_history, labels={'x':'Iteration', 'y':'L2 Loss (Sum of Squared Residuals)'}, title='Gradient Descent: L2 Loss vs Iteration')
fig.write_image("Task4_GD_L2_Loss.png")