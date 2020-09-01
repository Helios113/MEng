import numpy as np
from typing import List

class IterationSteps:
    min = -1
    max = -1
    size = -1
    def __init__(self, steps : List[float]):
        self.setps = steps
        self.min = min(self.setps)
        self.max = max(self.setps)
        self.size = len(self.setps)

class ResultObject:
    def __init__ (self, result : IterationSteps, func, converged : bool, function_calls : int):
        if callable(func) == False:
            raise ValueError("func has to be a vallable function")
        self.list = result
        self.steps = self.list.size
        self.func = func
        self.curve = self.generateCurve(100) #Default number of points for original funnction
        self.converged = converged
        self.function_calls = function_calls  
        
    def generateCurve (self, points):
        return self.func(np.linspace(self.list.min, self.list.max, points))

def f(x):
    return x/2

a = [1,2,3]

i = IterationSteps(a)
r = ResultObject(i, f,True,3)
print(r.curve)


def extended(func, x0, c, fprime = None):
    print("TODO")
    """
    Parameters
    ----------
    func : callable
        The function whose zero is wanted. It must be a function of a
        single variable of the form ``f(x,a,b,c...)``, where ``a,b,c...``
        are extra arguments that can be passed in the `args` parameter.
    x0 : float, sequence, or ndarray
        An initial estimate of the zero that should be somewhere near the
        actual zero. If not scalar, then `func` must be vectorized and return
        a sequence or array of the same shape as its first argument.
    fprime : callable, optional
        The derivative of the function when available and convenient. If it
        is None (default), then the secant method is used.
        
    """