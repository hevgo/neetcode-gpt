import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        # x: 1D input array
        # weights: list of 2D weight matrices
        # biases: list of 1D bias vectors
        # Apply ReLU after each hidden layer, no activation on output layer
        # return np.round(your_answer, 5)
        
        n_layers = len(weights)
        if n_layers != len(biases):
            raise ValueError('not match')
        
        x = x
        for i in range(n_layers):
            weight = weights[i]
            bias = biases[i]
            z1 = x @ weight + bias
            a1 = np.maximum(0, z1)
            x = a1
        
        return np.round(x, 5)
