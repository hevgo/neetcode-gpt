import numpy as np
from typing import Tuple, List


class Solution:
    def batch_norm(self, x: List[List[float]], gamma: List[float], beta: List[float],
                   running_mean: List[float], running_var: List[float],
                   momentum: float, eps: float, training: bool) -> Tuple[List[List[float]], List[float], List[float]]:
        # During training: normalize using batch statistics, then update running stats
        # During inference: normalize using running stats (no batch stats needed)
        # Apply affine transform: y = gamma * x_hat + beta
        # Return (y, running_mean, running_var), all rounded to 4 decimals as lists
        
        X = np.array(x)
        gamma = np.array(gamma)
        beta = np.array(beta)
        running_mean_np = np.array(running_mean)
        running_var_np = np.array(running_var)

        N = len(x)

        B, F = X.shape

        if training:
            mu_B = np.mean(X, axis = 0)
            var_B = np.var(X, axis = 0)
            x_hat = (X - mu_B) / np.sqrt(var_B + eps)

            running_mean_np = (1 - momentum) * running_mean_np + momentum * mu_B
            running_var_np = (1 - momentum) * running_var_np + momentum * var_B
        else:
            x_hat = (X - running_mean_np) / np.sqrt(running_var_np + eps)
        
        y = gamma * x_hat + beta

        y_rounded = np.round(y, 4).tolist()
        running_mean_rounded = np.round(running_mean_np, 4).tolist()
        running_var_rounded = np.round(running_var_np, 4).tolist()
        
        return y_rounded, running_mean_rounded, running_var_rounded
