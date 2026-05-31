import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, activation: str) -> float:
        # x: 1D input array
        # w: 1D weight array (same length as x)
        # b: scalar bias
        # activation: "sigmoid" or "relu"

        z = x @ w + b
        
        if activation == 'sigmoid':
            a = 1 / (1 + np.exp(-z))
        elif activation =='relu':
            a = np.maximum(0, z)
        else:
            a = z
        
        return np.round(a, 5)