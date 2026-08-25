import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """
    Returns the trained weights and bias as (w, b).
    """
    n_samples, n_features = X.shape

    # Initialize parameters
    w = np.zeros(n_features)
    b = 0.0

    for _ in range(steps):
        # Linear prediction
        z = X @ w + b

        # Predicted probabilities
        y_pred = _sigmoid(z)

        # Gradients
        dw = (X.T @ (y_pred - y)) / n_samples
        db = np.mean(y_pred - y)

        # Gradient descent update
        w -= lr * dw
        b -= lr * db

    return w, float(b)
    pass