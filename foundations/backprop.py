import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        # x: 1D input array
        # w: 1D weight array
        # b: scalar bias
        # y_true: true target value
        # Return: (dL_dw rounded to 5 decimals, dL_db rounded to 5 decimals)
        
        # forward
        z = x @ w + b
        y_pred = 1 / (1 + np.exp(-z))
        loss = 0.5 * (y_pred - y_true)**2

        # backward
        dl_dy_pred = y_pred - y_true
        dl_dz = (y_pred - y_true) * y_pred * (1 - y_pred)
        dl_dw = dl_dz * x
        dl_db = np.sum(dl_dz)

        return (np.round(dl_dw, 5), np.round(dl_db, 5))
