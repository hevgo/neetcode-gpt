import numpy as np
from numpy.typing import NDArray
from typing import Tuple

class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        n_samples, n_features = X.shape
        
        # Initialize weights as a 1D array to perfectly match y's 1D shape
        w = np.zeros(n_features)
        b = 0.0

        for _ in range(epochs):
            # 1. Forward Pass: y_hat shape will be (n_samples,)
            y_hat = X @ w + b
            
            # 2. Error Vector: shape (n_samples,)
            error = y_hat - y
            
            # 3. Backward Pass: Compute Gradients
            dw = (2 / n_samples) * (X.T @ error)
            db = (2 / n_samples) * np.sum(error)
            
            # 4. Update Parameters
            w -= lr * dw
            b -= lr * db

        # Return rounded values to 5 decimal places
        return np.round(w, 5), np.round(b, 5).item()