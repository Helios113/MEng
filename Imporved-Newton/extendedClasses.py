from typing import List
import numpy as np

class IterationSteps:
    min = -1
    max = -1
    size = -1
    def __init__(self, steps : np.ndarray.astype):
        self.setps = steps
        self.min = min(self.setps)
        self.max = max(self.setps)
        self.size = len(self.setps)

class ResultObject:
    curve = None
    def __init__ (self, result : IterationSteps, func, converged : bool, resultFlag : str):
        if callable(func) == False:
            raise ValueError("func has to be a vallable function")
        self.list = result
        self.steps = self.list.size
        self.func = func
        #self.curve = self.generateCurve(100) #Default number of points for original funnction
        self.converged = converged
        self.resultFlag = resultFlag
        
    def generateCurve (self, points):
        self.curve = self.func(np.linspace(self.list.min, self.list.max, points))
    def generatePath (self):
        self.path = self.func(self.list.setps)


class InputSeed:
    def __init__(self, x0, c , f, fp , fpp, tol, rtol, max_iter : int = 50, mode : int = 0):
        self.x0 = x0
        self.c = c
        if callable(f) == False:
            raise ValueError("f needs to be a callable function")
        self.f = f
        if callable(fp) == False:
            raise ValueError("f' needs to be a callable function")
        self.fp = fp
        self.fpp = fpp
        self.tol = tol
        self.rtol = rtol
        self.max_iter = max_iter
        self.mode = mode

    def populate(self, min, max, f, fp , fpp, tol, rtol, max_iter : int = 50, mode : int = 0 , step = None, num = None, ): #single variate only currently
        if min > max:
            raise ValueError("minumum value has to be less than maximum")
        if min!=max and step == 0 and num == 0:
            raise ValueError("step/num size cannot be zero when a range is given")
        if step is not None:
            x0 = np.arange(min ,max, step)
            c = np.arange(min ,max, step)
        if num is not None:
            x0 = np.linspace(min ,max, num)
            c = np.linspace(min ,max, num)
        if callable(f) == False:
            raise ValueError("f needs to be a callable function")
        self.f = f
        if callable(fp) == False:
            raise ValueError("f' needs to be a callable function")
        self.fp = fp
        if callable(fpp) == False:
            raise ValueError("f'' needs to be a callable function")
        self.fpp = fpp
        self.tol = tol
        self.rtol = rtol
        self.max_iter = max_iter
        
