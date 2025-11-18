import math
from typing import Tuple

import numpy as np


class LogisticRegressionSGD:
    """Binary logistic regression trained with mini-batch SGD.

    Model: p(y=1|x) = sigmoid(x @ w + b)
    Loss: binary cross-entropy with optional L2 on w.
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

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        # numerically stable sigmoid
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X: np.ndarray, y: np.ndarray, n_epochs: int = 50) -> None:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        for _ in range(n_epochs):
            for X_batch, y_batch in self._iterate_minibatches(X, y):
                logits = X_batch @ self.w + self.b
                probs = self._sigmoid(logits)

                # Gradient of BCE loss with respect to logits is (probs - y)
                error = probs - y_batch

                grad_w = (1.0 / X_batch.shape[0]) * (X_batch.T @ error)
                grad_b = float(np.mean(error))

                if self.l2 > 0.0:
                    grad_w += 2.0 * self.l2 * self.w

                self.w -= self.learning_rate * grad_w
                self.b -= self.learning_rate * grad_b

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        logits = X @ self.w + self.b
        return self._sigmoid(logits)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)

    def parameters(self) -> Tuple[np.ndarray, float]:
        return self.w.copy(), float(self.b)
