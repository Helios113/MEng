import numpy as np
from numpy.linalg import pinv
from functools import partial
from datetime import datetime
startTime = datetime.now()
delta = 1e-6
conDelta = 0.001414


def fun(x):
    return np.array([np.exp(x[0])-x[1], x[0]*x[1]-np.exp(x[0])])  # 1
    # return np.array([x[0]**2-x[1]**2-9, 2*x[0]*x[1]])  #  2
    # return np.array([(x[0]**2)-(1/x[0])+x[1],(1/x[1])+x[0]])  #  4
    # return np.array([x[0]**3-3*x[0]*x[1]**2-1, 3*x[0]**2*x[1]-x[1]**3]) #  5


def grad_fun(x):
    # 1
    return np.array([np.exp(x[0]), x[1]-np.exp(x[0]), -1, x[0]]).reshape(2, 2)
    # return np.array([2*x[0], -2*x[1], 2*x[1], 2*x[0]]).reshape(2, 2)  #  2
    # return np.array([2*x[0]+(1/(x[0]**2)),1,1,-1/(x[1]**2)]).reshape(2, 2)  #  4
    # return np.array([3*x[0]**2-3*x[1]**2, -6*x[0]*x[1], 6*x[0]*x[1], 3*x[0]**2-3*x[1]**2]).reshape(2, 2) #  5


def P(c, x):
    a = x-c
    b = fun(x)-fun(c)
    d = fun(x)
    #c = np.divide(d, b, out=np.zeros_like(d), where=b!=0)
    return np.matmul(a.reshape(2, -1), (d/b).reshape(-1, 2)).reshape(-1, 1)


def getPartial(x):
    global c
    dxc = x-c
    Fx = fun(x)
    Fc = fun(c)
    Fij = grad_fun(x)

    Hj = -Fc/(Fx-Fc)**2
    G = Fx/(Fx-Fc)
    #print(Fij, Hj)
    S = (Fij.T*Hj).T
    S = np.einsum("i, jk", dxc, S)
    # print(S)
    # print(G)
    F = np.einsum("ik, j", np.identity(2), G)
    # print(F)
    ans = F+S
    # print(ans)

    return ans.reshape(4, 2)


def check_root(x):
    #ans = np.array([1, 2.718])
    ans = np.array([1.260, -0.794])
    #ans = 0
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
    ans = np.array([1, 2.718])
    c = x[2:].copy()
    x = x[:2].copy()
    cnt = 0
    rate = []
    order = []
    error = []
    for i in range(100):
        error.append(np.linalg.norm(x-ans))
        if len(error) > 2:
            order.append(np.log10(error[-1]/error[-2]) /
                         np.log10(error[-2]/error[-3]))
        if len(order) > 0:
            rate.append(error[-1]/(error[-2]**order[-1]))
        cnt += 1
        """
        q = P(c, x)
        p = getPartial(x)
        p = pinv(p)
        """
        q = P(c, x).reshape(1, -1)
        p = pinv(getPartial(x).T)
        step = np.matmul(q, p)
        if np.linalg.norm(step) < delta:
            break
        x = x - step.flatten()
    # print(x)
    print(np.round(x, 3).tolist(), cnt)
    if check_root(x):
        return np.round(x, 2).tolist(), cnt, order, rate
    return None, cnt, order, rate


solve(np.array([1, 1, 1, 2]))
