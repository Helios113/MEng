from myGNE import GNSolver
import numpy as np
import matplotlib.pyplot as plt
from model import func
from datetime import datetime



np.random.seed(10)

NOISE = 2
COEFFICIENTS = [6,3,-0.1,12]

x = np.mgrid[1:8:200j, 1:8:200j].reshape(2, -1).T
#x = np.arange(1,20)
print(x)
y = func(x, COEFFICIENTS)
yn = y + NOISE * np.random.random_sample(y.shape)


solver = GNSolver(fit_function=func, max_iter=600, tolerance_difference=10 ** (-6))
init_guess =1000*np.random.random(len(COEFFICIENTS))
startTime = datetime.now()
ANSWER = solver.fit(x, yn, init_guess)
print("Result",datetime.now() - startTime)

fit = solver.get_estimate()
residual = solver.get_residual(solver.theta)
print(ANSWER[0], ANSWER[1])
print(COEFFICIENTS)
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