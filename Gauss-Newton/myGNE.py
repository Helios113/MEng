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

        stepps1 = []
        stepps2 = []
        for k in range(self.max_iter):
            steps += 1
            ri = self.get_residual(self.theta)  #  get residual at thetha zero
            riJ = self.get_residual_der(self.theta)  # get derivative of first derivative residual 
            num = np.einsum("i,ij", ri, riJ)  #  get "numertor"

            inv = self.get_inverse(np.einsum("ik,ij", riJ, riJ))  #  get inverse of JT J

            step = np.einsum("j,kj", num, inv)  # calculate step delta theta K

            riJK = self.get_residual_der_der(self.theta)  # calcualte second derivative of residual
            
            Jhik = (0.5*np.einsum("ikl,l", riJK, step))+riJ  # calculate J hat iK

            Jhkj = np.einsum("ik,ij", Jhik, Jhik)  # calculate Jhat T Jhat
            Jhkj1 = self.get_inverse(Jhkj)  # get inverse of Jhat T Jhat
            num = np.einsum("i,ij", ri, Jhik)  # get "numertor"
            step2 = np.einsum("j,jk", num, Jhkj1)  # calculate step delta theta L

            """
            This is where the step used can be changed
            """
            self.theta = self.theta - step2  # applymethod using corrected step



            #Text out
            print("########################")
            print("Original",step)
            print("---------------")
            print("Corrected",step2)
            print("########################")

            #stuff for plotting
            stepps1.append(step)
            stepps2.append(step2)


            rmse = np.sqrt(np.sum(ri ** 2))
            if self.tolerance_difference is not None:
                diff = np.abs(rmse_prev - rmse)
                if diff < self.tolerance_difference:
                    return self.theta, steps, stepps1, stepps2
            if rmse < self.tolerance:
                return self.theta, steps, stepps1, stepps2
            rmse_prev = rmse

        return self.theta, steps, stepps1, stepps2
        
    def get_residual(self, t):
        return self.y-self.function(self.x, t).reshape(-1)

    def get_residual_der(self, t: np.ndarray = None, step: float = 10 ** (-6)) -> np.ndarray:
        """
        function to calculate the derivative of the residual
        it iterates through each param, adds a step size and stacks the results horizontaly
        yielding d(ri)/dO = [d(ri)/dO_1, d(ri)/dO_2, ... ,d(ri)/dO_m]
        """
        a = None
        b = None
        for i in range(len(t)):
            t1 = t.copy()
            t1[i] += step
            c = self.get_residual(t1).reshape(-1,1)  # residual with added step  
            d = self.get_residual(t).reshape(-1,1)   # residual without added step
            a = np.hstack((a, c)) if a is not None else c  # stacking
            b = np.hstack((b, d)) if b is not None else d  # stacking
        return (a-b)/step  # calculating the derivative

    def get_residual_der_der(self, t: np.ndarray = None, step: float = 10 ** (-6)) -> np.ndarray:
        """
        function to calculate the second derivative of the residual
        it iterates through each param, adds a step size and stacks the results along the third axis
        yielding d^2(ri)/dO^2 = [d(d(ri)/dO)/dO_1, d(d(ri)/dO)/dO_2, ... ,d(d(ri)/dO)/dO_m]
        """
        a = None
        b = None
        for i in range(len(t)):
            t1 = t.copy()
            t1[i] += step
            c = self.get_residual_der(t1)
            d = self.get_residual_der(t)
            a = np.dstack((a, c)) if a is not None else c
            b = np.dstack((b, d)) if b is not None else d
        return ((a-b)/step)

    def get_inverse(self, rJ):
        return pinv(rJ)
