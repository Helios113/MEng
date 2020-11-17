import numpy as np
from typing import Callable
from numpy.linalg import pinv


class GN:
    def __init__(self, fit_function: Callable = None,
                 max_iter: int = 1000,
                 tolerance_difference: float = 10 ** (-16),
                 tolerance: float = 10 ** (-9),
                 init_guess: np.ndarray = None,
                 ):
        if fit_function is None:
            raise ValueError("Fit function has be inputed")
        self.function = fit_function
        self.max_iter = max_iter
        self.tolerance_difference = tolerance_difference
        self.tolerance = tolerance
        self.theta = None
        self.x = None
        self.y = None
        self.init_guess = None
        if init_guess is not None:
            self.init_guess = init_guess

    def fit(self,
            x: np.ndarray = None,
            y: np.ndarray = None,
            init_guess: np.ndarray = None
            ):
        """
        Fit coefficients by minimizing RMSE.
        :param x: Independent variable.
        :param y: Response vector.
        :param init_guess: Initial guess for coefficients.
        :return: Fitted coefficients.
        """
        self.x = x
        self.y = y
        if init_guess is not None:
            self.init_guess = init_guess

        if self.init_guess is None:
            raise Exception("Initial guess needs to be provided")
        
        self.theta = self.init_guess
        rmse_prev = np.inf
        steps = 0
        for k in range(self.max_iter):
            steps += 1
            r = self.get_residual(self.theta)
            rJ = self.get_residual_der(self.theta)
            rJ = self.get_inverse(rJ)
            self.theta = self.theta - np.einsum("i,ji", r, rJ)
            rmse = np.sqrt(np.sum(r ** 2))
            if self.tolerance_difference is not None:
                diff = np.abs(rmse_prev - rmse)
                if diff < self.tolerance_difference:
                    return self.theta, steps
            if rmse < self.tolerance:
                return self.theta, steps
            rmse_prev = rmse

        return self.theta, steps
        
    def get_residual(self, t):
        return self.y-self.function(self.x, t).reshape(-1)

    def get_residual_der(self, t: np.ndarray = None, step: float = 10 ** (-6)) -> np.ndarray:
        a = None
        b = None
        for i in range(len(t)):
            t1 = t.copy()
            t1[i] += step
            c = self.get_residual(t1).reshape(-1,1)
            d = self.get_residual(t).reshape(-1,1)
            a = np.hstack((a, c)) if a is not None else c
            b = np.hstack((b, d)) if b is not None else d
        return (a-b)/step

    def get_inverse(self, rJ):
        return pinv(rJ)
