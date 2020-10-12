import numpy as np
from numpy.linalg import pinv
from functools import partial
delta = 1e-4
import matplotlib.cm as cm
import matplotlib.pyplot as plt

class ENM:

    def __init__(self, x, c, f):
        self.__x = x
        self.c = c
        self.f = self.covert_to_partial_function(f)
        self.f1 = f
        self.roots, self.steps = self.solve()
        #  print(self.q())

    def covert_to_partial_function(self, f):
        list = []
        for i in f:
            list.append(partial(self.P, i))
        return list

    def P(self, g, x):
        a = x-self.c
        b = g(x)-g(self.c)
        d = g(x)
        #print((d/b)*a)
        return (d/b)*a

    def getPartial(self, j, i):
        global delta
        func = self.f[j]

        a = self.__x.copy()
        b = self.__x.copy()

        a[i] -= delta
        b[i] += delta

        fa = func(a)
        fb = func(b)
        return (fb-fa)/(2*delta)

    #  Assembiling Jacobian inverse (pij)
    def p(self):
        vs = None
        for i in range(len(self.f)):
            hs = None
            for j in range(len(self.__x)):
                ans = self.getPartial(i, j)
                hs = np.hstack((hs, ans)) if hs is not None else ans
            vs = np.vstack((vs, hs)) if vs is not None else hs
        return pinv(vs)

    #  Assembiling full F vector (qj)
    def q(self):
        vs = None
        for i in range(len(self.f)):
            ans = self.f[i](self.__x)
            vs = np.vstack((vs, ans)) if vs is not None else ans
        return vs

    #  Iterate method
    # TODO
    # convergance criteria
    def solve(self):
        steps = []
        for i in range(10):
            #  print(self.__x)
            steps.append(self.__x.flatten())
            self.__x = self.__x - np.matmul(self.p(), self.q())
        #  print(self.x)
        return self.__x, steps

    def mfunc(self, nos):
        if self.roots is None:
            raise ValueError("No solution to show func")
            return
        vals = np.asanyarray(self.steps).T
        vs = None
        for i in vals:
            ans = np.linspace(np.min(i), np.max(i), nos).flatten()
            vs = np.vstack((vs, ans)) if vs is not None else ans
        X, Y = np.asanyarray(np.meshgrid(*vs, sparse=False))
        
        xz = np.empty((len(X), len(Y)))
        for ii,i in enumerate(X):
            for jj,j in enumerate(i):
                xz[ii, jj] = self.f1[0](np.array([X[ii,jj],Y[ii,jj]]).reshape(-1, 1))
        print(X.shape)
        return X, Y, xz


def f1(x):
    return np.exp(x[0])-x[1]


def f2(x):

    return x[0]*x[1] - np.exp(x[0])


f = [f1, f2]

n = np.arange(8).reshape((2, 4))
#  x = np.zeros((2,1))
#  c = np.zeros((2,1))
x = np.array([20.0, 25.0]).reshape((2, 1))
c = np.array([1, 1]).reshape((2, 1))

t = ENM(x, c, f)
X, Y, Z = t.mfunc(100)

fig, ax = plt.subplots()
CS = ax.contour(X, Y, Z)
plt.show()