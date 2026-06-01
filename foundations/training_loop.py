import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        # X: (n_samples, n_features)
        # y: (n_samples,) targets
        # epochs: number of training iterations
        # lr: learning rate
        #
        # Model: y_hat = X @ w + b
        # Loss: MSE = (1/n) * sum((y_hat - y)^2)
        # Initialize w = zeros, b = 0
        # return (np.round(w, 5), round(b, 5))
        
        n_samples, n_features = X.shape
        n_y, = y.shape
        if n_samples != n_y:
            raise ValueError("format not match")

        # learn with batch
        # initial weight
        w = np.zeros((n_features,)) # n_features
        b = 0.0

        for epoch in range(epochs):
            y_hat = X @ w + b # n_samples,
            # loss = (1/n_samples) * sum((y_hat - y)^2)
            dy_hat = (2/n_samples) * (y_hat - y) # n_samples
            dw = X.T @ dy_hat # n_features, 
            db = np.sum(dy_hat)

            w -= lr * dw
            b -= lr * db
        
        return (np.round(w, 5), round(b, 5))


        
