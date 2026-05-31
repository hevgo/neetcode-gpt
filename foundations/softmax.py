import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        z_max = np.max(z, axis = -1, keepdims = True)
        z_cal = z - z_max
        z_exp = np.exp(z_cal)
        output = z_exp / np.sum(z_exp)
        return np.round(output, 4)