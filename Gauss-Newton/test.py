from model import func, der_func, der2_func
import numpy as np
import matplotlib.pyplot as plt
from myGNE import GNSolver

COEFFICIENTS = [-0.001, 0.1,0.1,2,15]
NOISE = 0
x = np.arange(1,11)
#x = np.mgrid[1:8:8j, 1:8:8j].reshape(2, -1).T
y = func(x,COEFFICIENTS)
np.random.seed(24)
yn = y + NOISE * np.random.random_sample(y.shape)

init_guess = [1,1,1,1,1]#359*np.random.random(len(COEFFICIENTS)) 


g = GNSolver(func,der_func,der2_func,100)

ans = g.fit(x,yn,init_guess, False)
print(ans[0], ans[1])
"""
fit = g.get_estimate()
residual = g.get_residual(ans[0])
plt.figure()
plt.plot(x, y, label="Original, noiseless signal", linewidth=2)
plt.plot(x, yn, label="Noisy signal", linewidth=2)
plt.plot(x, fit, label="Fit", linewidth=2)
plt.plot(x, residual, label="Residual", linewidth=2)
plt.title("Gauss-Newton: curve fitting example")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid()
plt.legend()
plt.show()
"""