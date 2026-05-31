import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # return round(your_answer, 4)
        n = len(y_true)
        output = -(1/n) * np.sum(y_true * np.log(y_pred) + (1-y_true) * np.log(1 - y_pred))
        return np.round(output, 4)


    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        n = len(y_true)
        y_pred = np.clip(y_pred, 1e-15, 1.0 - 1e-15)
        y_match = np.log(y_pred) * y_true
        output = - (1/n) * np.sum(y_match)
        return np.round(output, 4)