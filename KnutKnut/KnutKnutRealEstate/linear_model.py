import math
from typing import Tuple

import numpy as np


class LinearRegressionSGD:
    """Simple linear regression trained with (mini-batch) SGD.

    This implements
        y_hat = X @ w + b
    and minimizes squared error with optional L2 regularization (ridge).
    """

    def __init__(
        self,
        n_features: int,
        learning_rate: float = 1e-3,
        l2: float = 0.0,
        batch_size: int = 32,
        shuffle: bool = True,
        random_state: int | None = 0,
    ) -> None:
        self.learning_rate = float(learning_rate)
        self.l2 = float(l2)
        self.batch_size = int(batch_size)
        self.shuffle = shuffle

        rng = np.random.default_rng(random_state)
        # Small random init for weights, zero bias
        self.w = rng.normal(loc=0.0, scale=0.01, size=(n_features,))
        self.b = 0.0

    def _iterate_minibatches(self, X: np.ndarray, y: np.ndarray):
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        if self.shuffle:
            np.random.shuffle(indices)
        for start in range(0, n_samples, self.batch_size):
            end = min(start + self.batch_size, n_samples)
            batch_idx = indices[start:end]
            yield X[batch_idx], y[batch_idx]

    def fit(self, X: np.ndarray, y: np.ndarray, n_epochs: int = 50) -> None:
        """Train the model using SGD.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
        y : array, shape (n_samples,)
        n_epochs : number of passes over the data
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        for _ in range(n_epochs):
            for X_batch, y_batch in self._iterate_minibatches(X, y):
                # Forward pass
                y_pred = X_batch @ self.w + self.b
                error = y_pred - y_batch

                # Gradients (MSE)
                grad_w = (2.0 / X_batch.shape[0]) * (X_batch.T @ error)
                grad_b = 2.0 * float(np.mean(error))

                # L2 regularization on weights only
                if self.l2 > 0.0:
                    grad_w += 2.0 * self.l2 * self.w

                # Parameter update
                self.w -= self.learning_rate * grad_w
                self.b -= self.learning_rate * grad_b

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return X @ self.w + self.b

    def parameters(self) -> Tuple[np.ndarray, float]:
        """Return (w, b)."""
        return self.w.copy(), float(self.b)
