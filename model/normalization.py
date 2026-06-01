import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], gamma: NDArray[np.float64], beta: NDArray[np.float64]) -> NDArray[np.float64]:
        # x: 1D feature vector
        # gamma: 1D scale parameter (same length as x)
        # beta: 1D shift parameter (same length as x)
        # eps = 1e-5

        n = x.shape[-1]
        
        mu = (1/n) * np.sum(x, axis = -1, keepdims = True)
        var = (1/n) * np.sum((x-mu)**2, axis = -1, keepdims = True) 
        eps = 1e-5

        x = (x - mu) / np.sqrt(var + eps)
        x_hat = x * gamma + beta

        return np.round(x_hat, 5)
