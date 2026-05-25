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
        gamman = np.array(gamma)
        beta = np.array(beta)
        running_mean = np.array(running_mean)
        running_var = np.array(running_var)

        n_batch, n_feture = X.shape

        if training:
            mu = np.mean(X, axis = 0)
            var = np.var(X, axis = 0)
            x_hat = (X-mu)/np.sqrt(var+eps)
            
            running_mean = (1 - momentum) * running_mean + momentum * mu
            running_var = (1 - momentum) * running_var + momentum * var
        
        else:
            x_hat = (X - running_mean)/np.sqrt(running_var + eps)
        
        y = gamma * x_hat + beta

        return np.round(y, 4), np.round(running_mean, 4), np.round(running_var, 4)