import numpy as np
globals = {'__builtins__': None, 'np': np}
expression = ""
parameters = {}
xn = 0
step = 0  # non dynamic step for now
c = 0


def rf(x):  # r function
    global expression
    global globals
    global parameters
    parameters['x'] = x
    return eval(expression, globals, parameters)


def dr(x):
    return (rf(x+step)-rf(x-step))/step


def dEN():
    global c
    r = rf(xn)
    rc = rf(c)
    r1 = dr(xn)
    print("denom", (r-((xn-c)*r1*(r/(r-rc)))))
    return xn - (((xn-c)*r)/(r-((xn-c)*r1*(r/(r-rc)))))


def iter():
    global xn
    xn1 = dEN()
    if abs(xn - xn1) > 1:
        xn = xn1
        iter()
    else:
        return xn


def solve(expr, param):
    global parameters
    global expression
    global xn
    global step
    global c
    expression = expr
    parameters = param
    xn = param['x']
    step = param['step']
    c = parameters['c']
    return iter()
