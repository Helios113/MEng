import numpy as np
from typing import Callable
import numpy as np
from numpy.linalg import pinv
import logging

logger = logging.getLogger(__name__)

    """
    2 cases that drive gauss newton
    I. one variable X and a bunch of params th 1XM complexity
    II. bunch of X and bunch of th result is NxM complexity
    """


def fun(x, th):
    """
    Univariate x(variable), multiple th(theta - parameters)
    """
    return (th[0]*x)/(th[1]+x)


def Fun(x, th):
    """
    Multivariate x(variables), multiple th(theta - parameters)
    """
    return 





class GNSolver():
    """
    Gauss-Newton solver class
    Usage:
            Set measurements at initialization
            Set initaial guesses for the paramenters @ theta
            Set a callable model function "model" used for minimization
            The class uses least quares minimization 0.5rTr = 0
    """
    def __init__(self,
                 measurments: np.ndarray = None,
                 theta: np.ndarray = None,
                 model: Callable = None):
        if measurments is None:
            raise Exception(
                "Measurments need to be provided <First function param>")
        if theta is None:
            raise Exception(
                "Theta parameters need to be provided <Second function param>")
        if model is None:
            raise Exception(
                "A model needs to be provided <Third function param>")
        self.y = measurments
        self.th = theta
        self.f = model

    def solve(self) -> np.ndarray:
        rmse_prev = np.inf
        for k in range(100):
            r = self.get_residual()
            j = self.get_jacobian()
            self.th = self.th - self.inv(j) @ r
            rmse = np.sqrt(np.sum(residual ** 2))
            logger.info(f"Iteration {k}: RMSE {rmse}")
            if self.tolerance_difference is not None:
                diff = np.abs(rmse_prev - rmse)
                if diff < self.tolerance_difference:
                    logger.info("RMSE difference between iterations smaller than tolerance. Fit terminated.")
                    return self.coefficients
            if rmse < self.tolerance:
                logger.info("RMSE error smaller than tolerance. Fit terminated.")
                return self.coefficients
            rmse_prev = rmse
        logger.info("Max number of iterations reached. Fit didn't converge.")

        return self.coefficients

    def get_residual(self) -> np.ndarray:
        y_fit = self.model(self.x, self.th)
        return y_fit - self.y
    def get_jacobian(self) -> np.ndarray:
        """
        Calculate Jacobian matrix numerically.
        J_ij = d(r_i)/d(x_j)
        """
        y0 = self.get_residual()
        jacobian = []
        for i, parameter in enumerate(self.th):
            x = self.th.copy()
            x[i] += step
            y = self._calculate_residual(x)
            derivative = (y - y0) / step
            jacobian.append(derivative)
        jacobian = np.array(jacobian).T

        return jacobian
    def _calculate_residual(self, x: )


import logging
from typing import Callable

import numpy as np
from numpy.linalg import pinv

logger = logging.getLogger(__name__)


class GNSolver:
    """
    Gauss-Newton solver.
    Given response vector y, dependent variable x and fit function f, 
    Minimize sum(residual^2) where residual = f(x, coefficients) - y.
    """

    def __init__(self,
                 model: Callable,
                 max_iter: int = 1000,
                 tolerance_difference: float = 10 ** (-16),
                 tolerance: float = 10 ** (-9),
                 theta: np.ndarray = None,
                 ):
        """
        :param fit_function: Function that needs be fitted; y_estimate = fit_function(x, coefficients).
        :param max_iter: Maximum number of iterations for optimization.
        :param tolerance_difference: Terminate iteration if RMSE difference between iterations smaller than tolerance.
        :param tolerance: Terminate iteration if RMSE is smaller than tolerance.
        :param init_guess: Initial guess for coefficients.
        """
        self.fit_function = fit_function
        self.max_iter = max_iter
        self.tolerance_difference = tolerance_difference
        self.tolerance = tolerance
        self.coefficients = None
        self.x = None
        self.y = None
        self.init_guess = None
        if init_guess is not None:
            self.init_guess = init_guess

    def fit(self,
            x: np.ndarray,
            y: np.ndarray,
            init_guess: np.ndarray = None) -> np.ndarray:
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

        if init_guess is None:
            raise Exception("Initial guess needs to be provided")

        self.coefficients = self.init_guess
        rmse_prev = np.inf
        for k in range(self.max_iter):
            residual = self.get_residual()
            jacobian = self._calculate_jacobian(self.coefficients, step=10 ** (-6))
            self.coefficients = self.coefficients - self._calculate_pseudoinverse(jacobian) @ residual
            rmse = np.sqrt(np.sum(residual ** 2))
            logger.info(f"Round {k}: RMSE {rmse}")
            if self.tolerance_difference is not None:
                diff = np.abs(rmse_prev - rmse)
                if diff < self.tolerance_difference:
                    logger.info("RMSE difference between iterations smaller than tolerance. Fit terminated.")
                    return self.coefficients
            if rmse < self.tolerance:
                logger.info("RMSE error smaller than tolerance. Fit terminated.")
                return self.coefficients
            rmse_prev = rmse
        logger.info("Max number of iterations reached. Fit didn't converge.")

        return self.coefficients

    def predict(self, x: np.ndarray):
        """
        Predict response for given x based on fitted coefficients.
        :param x: Independent variable.
        :return: Response vector.
        """
        return self.fit_function(x, self.coefficients)

    def get_residual(self) -> np.ndarray:
        """
        Get residual after fit.
        :return: Residual (y_fitted - y).
        """
        return self._calculate_residual(self.coefficients)

    def get_estimate(self) -> np.ndarray:
        """
        Get estimated response vector based on fit.
        :return: Response vector
        """
        return self.fit_function(self.x, self.coefficients)

    def _calculate_residual(self, coefficients: np.ndarray) -> np.ndarray:
        y_fit = self.fit_function(self.x, coefficients)
        return y_fit - self.y

    def _calculate_jacobian(self,
                            x0: np.ndarray,
                            step: float = 10 ** (-6)) -> np.ndarray:
        """
        Calculate Jacobian matrix numerically.
        J_ij = d(r_i)/d(x_j)
        """
        y0 = self._calculate_residual(x0)

        jacobian = []
        for i, parameter in enumerate(x0):
            x = x0.copy()
            x[i] += step
            y = self._calculate_residual(x)
            derivative = (y - y0) / step
            jacobian.append(derivative)
        jacobian = np.array(jacobian).T

        return jacobian

    @staticmethod
    def _calculate_pseudoinverse(x: np.ndarray) -> np.ndarray:
        """
        Moore-Penrose inverse.
        """
        return pinv(x.T @ x) @ x.T