import numpy as np
from numpy.linalg import pinv
from functools import partial
from datetime import datetime
startTime = datetime.now()
delta = 1e-6
x = np.array([1, 2],dtype=float)
c = np.array([2, 4],dtype=float)

def fun(x):
    #  x_{1}^{3}-3x_1x_2^2-1\\
    return np.array([np.exp(x[0])-x[1], x[0]*x[1]-np.exp(x[0])])
    #  2return x[0]**2-x[1]**2-9
    #  return x[0]**3-3*x[0]*x[1]**2-1


def P(c,x):
    a = x-c
    b = fun(x)-fun(c)
    d = fun(x)
    return np.matmul(a.reshape(2,-1),(d/b).reshape(-1,2)).T


def getPartial(f, x):
    global delta
    """
    dt = np.array([delta,-delta])
    xx = np.array(np.meshgrid(x, dt)).T.reshape(-1,2)
    xx = np.sum(xx, axis=1)
    xx = np.split(xx,2)
    xx = np.vstack((np.array(np.meshgrid(xx[0], x[1])).T.reshape(-1,2),np.array(np.meshgrid(x[0],xx[1])).T.reshape(-1,2)))
    vals = np.apply_along_axis(f,1,xx)
    print(xx)
    """
    vs = None
    for i in range(2):
        a = x.copy()
        b = x.copy()
        a[i] += delta
        b[i] -= delta
        ans = np.vstack((a,b))
        vs = np.vstack((vs, ans)) if vs is not None else ans
    vals = np.apply_along_axis(f,1,vs)
    return ((vals[::2]-vals[1::2])/(2*delta)).reshape(2,-1).T


def check_root(x):
    ans = np.array([1, 2.718])
    #ans =0
    global delta
    #for i in self.f1:   
    #print(np.linalg.norm(i(x)))
    #    ans += np.linalg.norm(i(x))
    #  print(ans)
    if np.linalg.norm(ans-x.flatten()) <= delta*10**3: #something like this
        return True
    #print("diff", np.linalg.norm(ans-np.round(x,3).flatten()))
    #if ans <= delta:
    #    return True
    return False


#  Iterate method
#
# convergance criteria
def solve(x):
    global delta
    global c
    cnt = 0
    for i in range(10):
        cnt += 1
        p = pinv(getPartial(partial(P, c), x)) 
        q = P(c, x).reshape(-1, 1)
        step = np.matmul(p, q)
        if np.linalg.norm(step) < delta:
            break
        x = x - step.flatten()

    if check_root(x):
        return x, cnt
    return None, cnt


print(solve(x))
print(datetime.now() - startTime)