import numpy as np
from numpy.linalg import pinv
from functools import partial
from datetime import datetime
startTime = datetime.now()
delta = 1e-6
x = np.array([1, 2],dtype=float)
c = np.array([2, 5],dtype=float)

def fun(x):
    #  x_{1}^{3}-3x_1x_2^2-1\\
    #1return np.array([np.exp(x[0])-x[1], x[0]*x[1]-np.exp(x[0])])
    return np.array([x[0]**2-x[1]**2-9, 2*x[0]*x[1]])
    #  return x[0]**3-3*x[0]*x[1]**2-1


def grad_fun(x):
    #return np.array([np.exp(x[0]), -1, x[1]-np.exp(x[0]), x[0]]).reshape(2, 2)
    return np.array([2*x[0], -2*x[1], 2*x[1] ,2*x[0]]).reshape(2, 2)


def P(c,x):
    a = x-c
    b = fun(x)-fun(c)
    d = fun(x)
    return np.matmul(a.reshape(2,-1),(d/b).reshape(-1,2)).T


def getPartial(x):
    global c
    el1 = x-c
    el2 = fun(x)
    el3 = el2-fun(c)
    gr = grad_fun(x).T

    num2 = el2*gr
    num1 = el3*gr
    den = el3**2
    ans = np.outer(el1, ((num1-num2)/den).T)
    ans[::, :2:] = ans[::, :2:].T
    ans[::, 2:4:] = ans[::, 2:4:].T

    ans1 = np.outer(np.identity(2), el2/el3).T    
    hold = ans1[1::, :2:].copy()
    ans1[1::, :2:] = ans1[:1:, 2:4:]
    ans1[:1:, 2:4:] = hold

    ans = ans + ans1
    return ans.T


def check_root(x):
    #ans = np.array([1, 2.718])
    ans =0
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
        p = pinv(getPartial(x)) 
        q = P(c, x).reshape(-1, 1)
        step = np.matmul(p, q)
        if np.linalg.norm(step) < delta:
            break
        x = x - step.flatten()
    #print(x)
    if check_root(x):
        return x, cnt
    return None, cnt


#print(getPartial(x))


print(solve(x))
print(datetime.now() - startTime)