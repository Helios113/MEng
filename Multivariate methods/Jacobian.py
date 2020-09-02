import numpy as np
from typing import List
delta = 1e-4

def Jacobian(L, x):
    ans = np.empty([len(x),len(L)])
    for b in range(len(L)):
        ans[:,b] = JacobianColumn(L[b],x)
    return ans

def JacobianColumn(func, x : List[float]):
    size = len(x)
    ans = np.empty([size])
    for i in range(size):
        ans[i] = getPartial(func, i , x)
    return ans

def getPartial(func, i, x):
    global delta
    a = [j for j in x]
    a[i] -= delta
    b = [j for j in x]
    b[i] += delta
    fa = func(a)
    fb = func(b)
    return (fb-fa)/(2*delta)