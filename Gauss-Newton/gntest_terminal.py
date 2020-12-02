import logging

import matplotlib.pyplot as plt
import numpy as np
from model import func
from GN_test import GNSolver

logging.basicConfig(level=logging.INFO)

NOISE = 0
COEFFICIENTS = [-0.001, 0.1, 0.1, 2, 15]





def main():
    x = np.arange(1, 11)

    y = func(x, COEFFICIENTS)
    yn = y + NOISE * np.random.randn(len(x))

    solver = GNSolver(fit_function=func, max_iter=30)
    init_guess = [1,1,1,1,1]#1000000 * np.random.random(len(COEFFICIENTS))
    a = solver.fit(x, yn, init_guess)
    fit = solver.get_estimate()
    residual = solver.get_residual()
    print(a)
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


if __name__ == "__main__":
    main()