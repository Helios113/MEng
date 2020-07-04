import numpy as np
delta = 1e-6
conDelta = 0.001414
c = np.empty(2)


def fun(x):
    return np.array([np.exp(x[0])-x[1], x[0]*x[1]-np.exp(x[0])]) #  1
    #return np.array([x[0]**2-x[1]**2-9, 2*x[0]*x[1]])  #  2
    #return np.array([(x[0]**2)-(1/x[0])+x[1],(1/x[1])+x[0]])  #  4
    #return np.array([x[0]**3-3*x[0]*x[1]**2-1, 3*x[0]**2*x[1]-x[1]**3]) #  5



def P(x):
    return fun(x)
    global c
    a = (x-c)*fun(x)
    b = fun(x)-fun(c)
    return a/b


def check_root(x):
    global conDelta
    ans = np.array([1, 2.718])
    #ans = 0
    if isinstance(ans, np.ndarray):
        if np.linalg.norm(ans-x.flatten()) <= conDelta:
            return True
    else:
        ans += np.linalg.norm(fun(x))
        if ans <= conDelta:
            return True
    return False


def solve(x):
    global delta
    global c
    x0 = x[:2].copy()
    x1 = x[2:4].copy()
    c = x[4:].copy()
    cnt = 0
    for i in range(100):

        ff = P(x1)

        dH = P(x1)-P(x0)

        #  t = np.zeros((2, 1))
        t = (-ff/dH)#  .reshape(2, -1)
        #  t = np.divide(-ff, dH, where=dH!=0).reshape(2,-1)
        dXY = (x1-x0)#  .reshape(-1, 2)
        #  delt = np.matmul(t, dX)
        #  step = (dXY * t[i%2]).flatten()
        step = (dXY * t).flatten()
        #  step = delt.sum(axis=0).flatten()
        x2 = x1 + step
        if np.linalg.norm(step) <= delta:
            break
        x0, x1 = x1, x2
        cnt += 1
    if check_root(x2):
        return np.round(x2, 2).tolist(), cnt
    return None, cnt
