import numpy as np


def f_mod(f, x, c):
    return f(x)*((x-c)/(f(x)-f(c)))


def secant_method(f, x0, x1, c, iterations):
    """Return the root calculated using the secant method."""
    for i in range(iterations):
        x2 = x1 - f_mod(f,x1,c) * (x1 - x0) / float(f_mod(f,x1,c) - f_mod(f,x0,c))
        x0, x1 = x1, x2
        print(x0, x1, x2)
    return x2


def f_example(x):
    return 1/np.exp(x) - 10 #  problem



root = secant_method(f_example, 20, 10, 10, 50)

print(f"Root: {root}") # Root: 24.738633748750722


"""
P(x) = (x-c)/(r(x)-r(c))

so func = r(x)((x-c)/(r(x)-r(c)))
"""