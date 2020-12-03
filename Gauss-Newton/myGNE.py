import numpy as np
from typing import Callable
from numpy.linalg import pinv
from scipy.optimize import line_search


class GNSolver:
    def __init__(self, fit_function: Callable = None,
                 derivative: Callable = None,
                 derivative2: Callable = None,
                 max_iter: int = 1000,
                 tolerance_difference: float = 10 ** (-10),
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
        self.derivative = None
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
        
        rmse_prev = np.inf
        steps = 0

        stepps1 = []
        stepps2 = []
        rmse_steps = []
        for k in range(self.max_iter):
            steps += 1
            self.get_exp_derivative(self.theta)
            ri = self.get_residual(self.theta) #  get residual at thetha zero
            #print(self.theta)
            
            if self.derivative is not None:
                riJ = self.get_exp_derivative(self.theta)
            else:
                continue
                #riJ = self.get_residual_der(self.theta)
            try:
                th1 = np.linalg.solve(riJ.T @ riJ, -ri @ riJ)
                #print(f"Round {k} --------------")
                #print("coeffs", np.round(self.theta,6))
                #print("Step",np.round(th1,6))
                #print("---------------")
            except:
                print(f"Round {k} --------------")
                print(riJ.T @ riJ)
                print(-ri @ riJ)
                return
                
            if not mode:
                self.theta+=th1
            else:
                #riJ = self.get_residual_der(self.theta)  # get derivative of first derivative residual 
                """
                num = np.einsum("i,ij", ri, riJ)  #  get "numertor"
                inv = self.get_inverse(np.einsum("ik,ij", riJ, riJ))  #  get inverse of JT J
                step = np.einsum("j,kj", -num, inv)  # calculate step delta theta K
                """
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
            #print(self.theta)
            """
            Jhik = (0.5*np.einsum("ikl,l", riJK, step))+riJ  # calculate J hat iK
            Jhkj = np.einsum("ik,ij", Jhik, Jhik)  # calculate Jhat T Jhat
            Jhkj1 = self.get_inverse(Jhkj)  # get inverse of Jhat T Jhat
            num1= np.einsum("i,ij", ri, Jhik)  # get "numertor"
            step2 = np.einsum("j,jk", -num1, Jhkj1)  # calculate step delta theta L

            var1 = np.einsum("i,i", step,num)
            var2 = np.einsum("i,i", step2,num1)  
            print("original", var1)
            print("modified", var2)
            print("ratio", var1/var2)
            if np.abs((var1/var2)-1) <= 0.001:
                self.theta = self.theta + step2
                print("STEP 2 Taken")
            else:
                self.theta = self.theta + step
                print("STEP 1 Taken")
            
            #print("Original direction", var1)
            #print("Modified direction", var2)
            #print("aditional factor original", var3)
            #print("first factor", var5)
            #print("Original step", a1[0])
            #print("modified step", a2[0])
            #print("Work vector", wv)
            #print("________________________________")
            #
            
            #Approach two
            #Line search
            #line search results
            #alpha = 1 for stepsize
            
            a = line_search(self.get_rmse, self.get_rmse_der, np.array(self.theta), np.array(step2))
            if a[0] == None:
                print("step dir",np.einsum("i,i", step,np.einsum("i,ij",ri ,riJ)))
                print("step dir2",np.einsum("i,i", step2,np.einsum("i,ij",ri ,Jhik)))
                print(self.get_rmse(self.theta))
                self.theta = self.theta + step2
            else:
                print(a)
                step2 *= a[0]
                self.theta = self.theta + step2
            
            
            #Text out
            #print("########################")
            #print("Original",step)
            #print("---------------")
            #print("Corrected",step2)
            #print("########################")

            #stuff for plotting
            stepps1.append(step)
            stepps2.append(step2)
            """
            rmse = self.get_rmse(self.theta)
            #print(rmse)
            rmse_steps.append(rmse)
            if self.tolerance_difference is not None:
                diff = np.abs(rmse_prev - rmse)
                if diff < self.tolerance_difference:
                    #print("HERE")
                    return self.theta, steps, stepps1, stepps2, rmse_steps
            if rmse < self.tolerance:
                #print("HERE2")
                return self.theta, steps, stepps1, stepps2, rmse_steps
            rmse_prev = rmse

        return self.theta, steps, stepps1, stepps2, rmse_steps
        
        
    def get_residual(self, t):
        return self.y-self.function(self.x, t).reshape(-1)

    def get_rmse(self, t):
        ri = self.get_residual(t)
        return np.einsum("i,i",ri,ri)**0.5
    def get_rmse_der(self, t) -> np.ndarray:
        ri = self.get_residual(t)
        riJ = self.get_residual_der(t)
        return np.einsum("i,ij",ri,riJ)/self.get_rmse(t)
    def get_estimate(self) -> np.ndarray:
        """
        Get estimated response vector based on fit.
        :return: Response vector
        """
        return self.function(self.x, self.theta)

    def get_residual_der(self, t: np.ndarray = None, step: float = 10 ** (-8)) -> np.ndarray:
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
        return (a-b)/(2*step)  # calculating the derivative

    def get_residual_der_der(self, t: np.ndarray = None, step: float = 10 ** (-8)) -> np.ndarray:
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
        return (a-b)/(2*step)

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
        
