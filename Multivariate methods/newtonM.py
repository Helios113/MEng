import numpy as np
import Jacobian as jb

delta = 1e-6


def f1(x):
    return np.exp(x[0])-x[1]
    #2return x[0]**2-x[1]**2-9
    #return x[0]**3-3*x[0]*x[1]**2-1


def f2(x):
    return x[0]*x[1]-np.exp(x[0])
    #2return 2*x[0]*x[1]
    #return 3*x[0]**2*x[1]-x[1]**3


L = [f1, f2]


def Newton(x):
    global L
    x = np.array(x)
    steps = []
    #print(x)

    for i in range(100):
        if len(steps) > 1:
            if np.linalg.norm(steps[-1]-steps[-2]) < delta/10**6:
                break
        steps.append(x.copy())
        j = jb.Jacobian(L, x)
        ff = f(L, x)
        if not is_invertible(j):
            return None, steps
        b = np.linalg.inv(j).dot(ff)
        x -= b
        
    if check_root(x):
        #  print("True:",x)
        return np.around(x, 5), steps
    #  print("False:",x, steps)
    return None, steps


def is_invertible(a):
    return a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0]


def check_root(x):
    ans = np.array([1, 2.718])
    if isinstance(ans, np.ndarray):
        if np.linalg.norm(ans-x.flatten()) <= 1e-3:
            return True
        else:
            print(x.flatten())
    else:
        for i in self.f1: 
            ans += np.linalg.norm(i(x))
        if ans <= 1e-3:
            return True
        return False


def f(L, x):
    ans = np.empty([len(L)])
    for j in range(len(L)):
        ans[j] = L[j](x)
    #  print(ans)
    return ans
    # for every fucntion evaluate it at vector x and return list of answers