import numpy as np
from numpy.linalg import pinv
from functools import partial
from datetime import datetime
startTime = datetime.now()
delta = 1e-6


def f1(x):
    #  x_{1}^{3}-3x_1x_2^2-1\\
    #return np.exp(x[0])-x[1]
    return x[0]**2-x[1]**2-9
    #return x[0]**3-3*x[0]*x[1]**2-1
    


def f2(x):
    #  3x_1^2x_2-x_2^3\\
    #return x[0]*x[1]-np.exp(x[0])
    return 2*x[0]*x[1]
    #return 3*x[0]**2*x[1]-x[1]**3


class ENM:

    def __init__(self, x):
        self.__x = np.array(x[0]).reshape((2, 1))
        self.start = x[0]
        self.c = np.array(x[1]).reshape((2, 1))
        self.f = self.covert_to_partial_function([f1, f2])
        self.f1 = [f1, f2]
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
        #  print((d/b)*a)
        return (d/b)*a

    def getPartial(self, j, i):
        global delta
        func = self.f[j]

        a = self.__x.copy()
        b = self.__x.copy()

        a[i] -= delta
        b[i] += delta
        #print(a)
        #print(b)
        fa = func(a)
        fb = func(b)
        #print((fb-fa)/(2*delta))
        #print("????????????????????")
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
        global delta
        steps = []
        cnt = 0
        while (True):
            if cnt > 100:
                break
            if len(steps) > 1:
                if np.linalg.norm(steps[-1]-steps[-2]) < delta:
                    break
            #  print(self.__x)
            steps.append(self.__x.flatten())
            qq = self.q()
            pp = self.p()
            #print(pp)
            #print(qq)
            self.__x = self.__x - np.matmul(pp, qq)
            cnt += 1
        #self.__x = self.__x - np.matmul(self.p(), self.q())
        #print(self.__x)
        if self.check_root(self.__x):
            return np.around(self.__x, 5), steps
        return None, steps

    def check_root(self, x):
        #ans = np.array([1, 2.718])
        ans =0
        global delta
        for i in self.f1:   
            #print(np.linalg.norm(i(x)))
            ans += np.linalg.norm(i(x))
        #  print(ans)
        #if np.linalg.norm(ans-x.flatten()) <= delta*10**3: #something like this
        #    return True
        #print("diff", np.linalg.norm(ans-np.round(x,3).flatten()))
        if ans <= delta:
            return True
        return False

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
        pcg = []
        for i in self.f1:
            pcg.append((X, Y, i([X, Y])))
        return pcg
