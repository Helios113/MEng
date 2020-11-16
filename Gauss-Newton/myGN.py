import numpy as np
from typing import Callable
tolerance

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
        self.coefficients = None
        self.x = None
        self.y = None
        self.init_guess = None
        if init_guess is not None:
            self.init_guess = init_guess

    def fit(x: np.ndarray = None,
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
        

    def residual():
