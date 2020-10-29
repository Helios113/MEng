import numpy as np
from functools import partial
delta = 1e-6


def fun(x):
    return np.array([np.exp(x[0])-x[1], x[0]*x[1]-np.exp(x[0])])


def P(c, x):
    a = (x-c)*fun(x)
    b = fun(x)-fun(c)
    return a/b

x0 = np.array([10, 10])
x1 = np.array([1, 3])
c = np.array([2, 4])

def secant(f, x00, x11, iter):
    global delta
    x0 = x00.copy()
    x1 = x11.copy()
    for i in range(iter):
        print("ITERATION:", i,"////////////////")
        print("x1", x1)
        ff = f(x1)
        print("f(x1)", ff)
        print("x0", x0)
        print("f(x0)", f(x0))
        dH = f(x1)-f(x0)
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

ans = secant(partial(P, c), x0, x1,25)
print(ans)
print(fun(ans))
