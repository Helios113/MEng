from CGN import GNSolver
import numpy as np

def func(x, coeff):
    return coeff[0] * x ** 3 + coeff[1] * x ** 2 + coeff[2] * x[1] + coeff[3] + coeff[4] * np.sin(x[1])


COEFFICIENTS = [-0.001, 0.1, 0.1, 2, 15]

x = np.arange(1, 100)
y = func(x, COEFFICIENTS)
init_guess = [100000,1,1,1,1]

a = GNSolver(fit_function=func)
ANSWER = a.fit(x,y,init_guess)
print(ANSWER)