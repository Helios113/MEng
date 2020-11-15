import numpy as np
from numpy.linalg import pinv
from functools import partial
from datetime import datetime
startTime = datetime.now()
delta = 1e-6
conDelta = 0.001414
def fun(x):
    #return np.array([np.exp(x[0])-x[1], x[0]*x[1]-np.exp(x[0])]) #  1
    #return np.array([x[0]**2-x[1]**2-9, 2*x[0]*x[1]])  #  2
    #return np.array([x[0]**2-x[1]**3-x[0]*x[1]**2-1, x[0]**3-x[1]*x[1]**3-4])  #  3
    return np.array([(x[0]**2)-(1/x[0])+x[1],(1/x[1])+x[0]])  #  4
    #return np.array([x[0]**3-3*x[0]*x[1]**2-1, 3*x[0]**2*x[1]-x[1]**3]) #  5


def grad_fun(x):
    #return np.array([np.exp(x[0]), -1, x[1]-np.exp(x[0]), x[0]]).reshape(2, 2)  #  1
    #return np.array([2*x[0], -2*x[1], 2*x[1], 2*x[0]]).reshape(2, 2)  #  2
    #return np.array([2*x[0]-x[1]**2, -3*x[1]**2-2*x[0]*x[1], 3*x[0]**2-x[1]**3, -3*x[0]*x[1]**2]).reshape(2, 2)  #  3
    return np.array([2*x[0]+(1/(x[0]**2)),1,1,-1/(x[1]**2)]).reshape(2, 2)  #  4
    #return np.array([3*x[0]**2-3*x[1]**2, -6*x[0]*x[1], 6*x[0]*x[1], 3*x[0]**2-3*x[1]**2]).reshape(2, 2) #  5

def P(c,x):
    a = x-c
    b = fun(x)-fun(c)
    d = fun(x)
    #c = np.divide(d, b, out=np.zeros_like(d), where=b!=0)
    return np.matmul(a.reshape(2,-1), (d/b).reshape(-1,2)).T


def getPartial(x):
    global c
    el1 = x-c
    el2 = fun(x)
    el3 = el2-fun(c)
    gr = grad_fun(x).T

    num2 = el2*gr
    num1 = el3*gr
    den = el3**2

    #d = np.divide((num1-num2), den, out=np.zeros_like(num1), where=den!=0)
    ans = np.outer(el1, ((num1-num2)/den).T)
    ans[::, :2:] = ans[::, :2:].T
    ans[::, 2:4:] = ans[::, 2:4:].T

    #a = np.divide(el2, el3, out=np.zeros_like(el2), where=el3!=0)
    ans1 = np.outer(np.identity(2), el2/el3).T    
    hold = ans1[1::, :2:].copy()
    ans1[1::, :2:] = ans1[:1:, 2:4:]
    ans1[:1:, 2:4:] = hold

    ans = ans + ans1
    return ans.T


def check_root(x):
    #ans = np.array([1, 2.718])
    ans = 0
    if isinstance(ans, np.ndarray):
        if np.linalg.norm(ans-x.flatten()) <= conDelta:
            return True
    else:
        ans += np.linalg.norm(fun(x))
        if ans <= conDelta:
            return True
    return False
#  Iterate method
#
# convergance criteria


def solve(x):
    global delta
    global c
    c = x[2:].copy()
    x = x[:2].copy()
    cnt = 0
    for i in range(100):
        cnt += 1
        q = P(c, x).reshape(-1, 1)
        p = pinv(getPartial(x)) 
        step = np.matmul(p, q)
        if np.linalg.norm(step) < delta:
            break
        x = x - step.flatten()
    #print(x)
    
    if check_root(x) and cnt<50:
        return np.round(x, 3).tolist(), cnt
    return None, cnt

