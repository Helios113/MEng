import numpy as np
from numpy.linalg import pinv
from functools import partial 

delta = 1e-4
#functions
def f1(x):
    return np.exp(x[0])-x[1]

def f2(x):
    return x[0]*x[1] - np.exp(x[0])

F = [f1,f2]
n = np.arange(8).reshape((2,4))
#x = np.zeros((2,1))
#c = np.zeros((2,1))
x = np.array([36.0,2321.0]).reshape((2,1))
c = np.array([1,40]).reshape((2,1))

def P(c,g,x):
    a = x-c
    b = g(x)-g(c)
    d = g(x)
    return (d/b)*a

def getPartial(func, i, x):
    global delta
    a = x.copy()
    a[i] -= delta
    b = x.copy()
    b[i] += delta
    fa = func(a)
    fb = func(b)
    return (fb-fa)/(2*delta)

list = []
for i in F:
    list.append(partial(P,c,i))
def p(x, list):
    vs = None
    for i in list:
        hs = None
        for j in range(len(x)):
            ans = getPartial(i,j,x)
            hs = np.hstack((hs,ans)) if hs is not None else ans
        vs = np.vstack((vs,hs)) if vs is not None else hs
    return pinv(vs)
def q(x, list):
    vs = None
    for i in list:
        ans = i(x)
        vs = np.vstack((vs,ans)) if vs is not None else ans
    return vs
for i in range(80):
    x = x - np.matmul(p(x,list),q(x,list))
print(x)