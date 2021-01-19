import numpy as np
from typing import Callable
from numpy.linalg import pinv
from scipy.optimize import line_search


class GNSolver:
    def __init__(self, fit_function: Callable = None,
                 derivative: Callable = None,
                 derivative2: Callable = None,
                 max_iter: int = 1000,
                 original_root: np.ndarray = None,
                 tolerance_difference: float = 10 ** (-10),
                 tolerance: float = 10 ** (-9),
                 init_guess: np.ndarray = None,
                 ):
        if fit_function is None:
            raise ValueError("Fit function has be inputed")
        self.function = fit_function
        self.max_iter = max_iter
        self.original_root = original_root
        self.tolerance_difference = tolerance_difference
        self.tolerance = tolerance
        self.theta = None
        self.x = None
        self.y = None
        self.init_guess = None
        self.derivative = None
        self.derivative2 = None
        if init_guess is not None:
            self.init_guess = init_guess
        if derivative is not None:
            self.derivative = derivative
        if derivative2 is not None:
            self.derivative2 = derivative2

    def fit(self,
            x: np.ndarray = None,
            y: np.ndarray = None,
            init_guess: np.ndarray = None,
            mode: bool = True
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
        
        mse_prev = np.inf

        step_number = 0
        steps = []
        mse_steps = []
        q_rate = []
        q1_rate = []
        mu_rate = []
        for k in range(self.max_iter):
            step_number += 1
            #self.get_exp_derivative(self.theta)
            ri = self.get_residual(self.theta) #  get residual at thetha zero
            #print(self.theta)
            
            if self.derivative is not None:
                riJ = self.get_exp_derivative(self.theta)
            else:
                riJ = self.get_residual_der(self.theta)
            try:
                th1 = np.linalg.solve(riJ.T @ riJ, -ri @ riJ)
            except:
                return
                
            if not mode:
                self.theta+=th1
            else:
                if self.derivative2 is not None:
                    riJK = self.get_exp_derivative2(self.theta)
                else:
                    riJK = self.get_residual_der_der(self.theta)  # calcualte second derivative of residual
                JiK = riJ + 0.5*(riJK @ th1)
                try:
                    th2 = np.linalg.solve(JiK.T @ JiK, -ri @ JiK)
                except:
                    print("Thetha2:",self.theta)
                    print("LHS",JiK.T @ JiK)
                    print("RHS", -ri @ JiK)
                    return
                self.theta+=th2

            steps.append(self.theta.copy())
            
            
            #Convergence
            
            if len(steps) > 3:
               lnf = np.linalg.norm(steps[-1]-self.original_root)
               lns = np.linalg.norm(steps[-2]-self.original_root)
               lds = np.linalg.norm(steps[-3]-self.original_root)
               nl = np.log10(lnf/lns)
               dl = np.log10(lns/lds)
               q_rate.append(nl/dl)
            if len(q_rate) > 0:
               e1 = np.linalg.norm(steps[-1]-self.original_root)
               e2 = np.linalg.norm(steps[-2]-self.original_root)
               q1_rate.append(e1/(e2**q_rate[-1]))
               
            if self.original_root is not None:
                mu_rate.append(np.linalg.norm(self.theta - self.original_root))


            mse = self.get_mse(self.theta)
            mse_steps.append(mse)
            if self.tolerance_difference is not None:
                diff = np.abs(mse_prev - mse)
                if diff < self.tolerance_difference:
                    return self.theta, step_number, steps, mse_steps, q_rate, q1_rate, mu_rate
            if mse < self.tolerance:
                return self.theta, step_number, steps, mse_steps, q_rate, q1_rate, mu_rate
            mse_prev = mse

        return self.theta, step_number, steps, mse_steps, q_rate, q1_rate, mu_rate
        
        
    def get_residual(self, t):
        return self.y-self.function(self.x, t).reshape(-1)

    def get_mse(self, t):
        ri = self.get_residual(t)
        return np.einsum("i,i",ri,ri)*0.5
    def get_rmse_der(self, t) -> np.ndarray:
        ri = self.get_residual(t)
        riJ = self.get_residual_der(t)
        return np.einsum("i,ij",ri,riJ)
    def get_estimate(self) -> np.ndarray:
        """
        Get estimated response vector based on fit.
        :return: Response vector
        """
        return self.function(self.x, self.theta)

    def get_residual_der(self, t: np.ndarray = None, step: float = 10 ** (-5)) -> np.ndarray:
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
            t2 = t.copy()
            t2[i] -= step
            c = self.get_residual(t1).reshape(-1,1)  # residual with added step  
            d = self.get_residual(t2).reshape(-1,1)   # residual without added step
            a = np.hstack((a, c)) if a is not None else c  # stacking
            b = np.hstack((b, d)) if b is not None else d  # stacking
        aa = (a-b)/(2*step)
        aa[abs(aa) < step] = 0.0
        
        return aa  # calculating the derivative

    def get_residual_der_der(self, t: np.ndarray = None, step: float = 10 ** (-5)) -> np.ndarray:
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
            t2 = t.copy()
            t2[i] -= step
            c = self.get_residual_der(t1)
            d = self.get_residual_der(t2)
            #print("Diff",c-d)
            a = np.dstack((a, c)) if a is not None else c
            b = np.dstack((b, d)) if b is not None else d
        aa = (a-b)/(2*step)
        aa[abs(aa) < step] = 0.0
        
        return aa

    def get_inverse(self, rJ):
        return pinv(rJ)
    def get_exp_derivative(self, t):
        b = None
        for i in self.x:
            a =-self.derivative(i,self.theta)
            b = np.vstack((b, a)) if b is not None else a
        
        return b
    def get_exp_derivative2(self,t):
        b = None
        for i in self.x:
            a =-self.derivative2(i,self.theta)
            b = np.vstack((b, a)) if b is not None else a
        
        return b
        
