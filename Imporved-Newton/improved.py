import numpy as np
import keyboard


globals = {'__builtins__': None, 'np': np}
expression = ""
parameters = {}
xn = 0
step = 0  # non dynamic step for now
c = 0
rc = 0

iterations_until_convergence = 8000


def rf(x):  # r function
    global expression
    global globals
    global parameters
    parameters['x'] = x
    return eval(expression, globals, parameters)


def dr(x):
    return (rf(x+step)-rf(x-step))/(step*2)


def dEN():
    global c
    global xn
    global rc
    r = rf(xn)
    r1 = dr(xn)
    if r == rc:
        return False
    o = (xn-c)*r
    p1 = rc/(r-rc)
    p = r - ((xn-c)*r1*p1)
    return xn - (o/p)


dY = []


def iter():
    global xn
    global dY
    xn1 = dEN()
    if xn1 == c:
        return False
    dY.append(abs(rf(xn1)/rf(xn)))
    if len(dY) > iterations_until_convergence:
        return False
    if np.isclose(rf(xn1), 0) == False:
        xn = xn1
        return iter()
    else:
        return dY


def solve(expr, param):
    global dY
    global parameters
    global expression
    global xn
    global step
    global c
    global rc
    dY = []
    expression = expr
    parameters = param
    xn = param['x']
    step = param['step']
    c = parameters['c']
    rc = rf(c)
    if xn == c:
        return False
    return iter()
