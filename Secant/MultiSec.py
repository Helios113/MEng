from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

delta = 1e-10


def fun(x):
    #  x_{1}^{3}-3x_1x_2^2-1\\
    #1return np.array([np.exp(x[0])-x[1], x[0]*x[1]-np.exp(x[0])])
    #return np.array([x[0]**2-x[1]**2-9, 2*x[0]*x[1]])
    return np.array([x[0]**3-3*x[0]*x[1]**2-1, 3*x[0]**2*x[1]-x[1]**3])


x0 = np.array([10, 10])
x1 = np.array([1, 3])


def secant(x):
    global delta
    x1 = x[2:]
    x0 = x[:2]
    for i in range(100):
        print("ITERATION:", i, "////////////////")
        print("x1", x1)
        ff = fun(x1)
        print("fun(x1)", ff)
        print("x0", x0)
        print("fun(x0)", fun(x0))
        dH = fun(x1)-fun(x0)
        print("dH", dH)
        t = np.zeros((2, 1))
        t = (-ff/dH)#.reshape(2, -1)
        #  t = np.divide(-ff, dH, where=dH!=0).reshape(2,-1)
        print("t", t)
        #print("t_row", t[i % 2])
        dXY = (x1-x0)#.reshape(-1, 2)
        print("dXY", dXY)

        #delt = np.matmul(t, dX)
        #step = (dXY * t[i%2]).flatten()
        step = (dXY * t).flatten()
        #print("delt:",delt)
        #step = delt.sum(axis=0).flatten()
        print("Step:",step)
        if np.linalg.norm(step) <= delta:
            return x1
        x2 = x1 + step
        print("x2", x2)
        x0, x1 = x1, x2
        print("ITERATION END ////////////////")
    return x2


ans = secant(fun, x0, x1,100)
print(ans)
print(fun(ans))
