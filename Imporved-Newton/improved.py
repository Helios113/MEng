import numpy as np
from extendedClasses import InputSeed, ResultObject, IterationSteps
delta = 1e-4


def testFunction(x): # function name where we add equation
    return np.exp(x)-500


def secant(a, func): # needs to know our test function
    global delta
    return (func(a+delta) - func(a-delta)) / (2*delta)




"""
the function should work for only scalar values and not have list function built in
so that means that InputSeed needs to be a singleton as well
however a method for solving ranges of data is needed
"""
def extended(ent : InputSeed): 
    MAXITER = 1000
    global delta
    if ent.rtol != 0:
        delta = ent.rtol / 2
    rtol = ent.rtol
    tol = ent.tol
    xc = ent.x0 # initial guess of X
    c = ent.c # value of c
    f = ent.f # function being examined
    fp = ent.fp # first derivative of function defaults to secant method
    fc = f(c) # result of f(c) call
    xf = float("inf") # setting initial xf value to inf
   

    ans = [xc] # answer list
    converged = False
    result = ""
    if ent.mode == 0: # mode zero - set number of iterations
        for i in range(ent.max_iter):
            xf = eq(xc, f, fp, fc, xc-c)
            #print(f(xf))
            ans.append(xf)
            if rtol !=0 and abs(xc-xf) < rtol:
                result = "Relative tolerance reached"
                converged = True
                break
            if abs(f(xf)) < tol:
                result = "Absolute tolerance reached"
                converged = True
                break
            xc = xf
        if result == "":
            result = "Max iterations exceeded"
            converged = False
    elif ent.mode == 1: #mode one - relative tolerance
        itr = 0
        while abs(xf-xc) > rtol:
            itr+=1
            xf = eq(xc, f, fp, fc, xc-c)
            ans.append(xf)
            if itr > MAXITER:
                result = "Internal max iterations exceeded"
                converged = False
                break
            if abs(f(xf)) < tol:
                result = "Absolute tolerance reached"
                converged = True
                break
            xc = xf
        if result == "":
            result = "Relative tolerance reached"
            converged = True
    elif ent.mode == 2: #mode two - absolute tolerance
        itr = 0
        while abs(f(xf)) > tol:
            itr+=1
            xf = eq(xc, f, fp, fc, xc-c)
            ans.append(xf)
            if itr > MAXITER:
                result = "Internal max iterations exceeded"
                converged = False
                break
            if rtol !=0 and abs(xc-xf) < rtol:
                result = "Relative tolerance reached"
                converged = True
                break
            xc = xf
        if result == "":
            result = "Absolute tolerance reached"
            converged = True
    else:
        raise ValueError("invalid mode entered")

    if result != "":
        s = IterationSteps(ans)
        r = ResultObject(s,f,converged,result)
        return r
        

def eq(xc, f, fp, fc, sub): # here to keep code dry
    return xc - ((sub*f(xc)) / (f(xc) - (sub*fp(xc,f) * (fc / (f(xc)-fc)))))

print(secant(5,testFunction))

i = InputSeed(5,3,testFunction,secant,None,0.01, 0,max_iter=50,mode = 0)
r = extended(i)
print(r.resultFlag)
print(r.list.setps)
r.generateCurve(10)
print(r.curve)
r.generatePath()
print(r.path)