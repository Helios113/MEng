import numpy as np
def secant_method(f, x0, x1, iterations):
    """Return the root calculated using the secant method."""
    for i in range(iterations):
        x2 = x1 - f(x1) * (x1 - x0) / float(f(x1) - f(x0))
        x0, x1 = x1, x2
        print(x0, x1, x2)
    return x2

def f_example(x):
    return 1/np.exp(x) - 10 #problem

root = secant_method(f_example, 20, 10, 15)

print("Root: {}".format(root))  # Root: 24.738633748750722