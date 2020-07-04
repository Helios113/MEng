import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
delta = 1e-4
conDelta = 0.001414


def fun(x):
    #return np.array([np.exp(x[0])-x[1], x[0]*x[1]-np.exp(x[0])]) #  1
    #return np.array([x[0]**2-x[1]**2-9, 2*x[0]*x[1]])  #  2
    #return np.array([(x[0]**2)-(1/x[0])+x[1],(1/x[1])+x[0]])  #  4
    return np.array([x[0]**3-3*x[0]*x[1]**2-1, 3*x[0]**2*x[1]-x[1]**3]) #  5

def check_root(x):
    global conDelta
    #ans = np.array([1, 2.718])
    ans = 0
    if isinstance(ans, np.ndarray):
        if np.linalg.norm(ans-x.flatten()) <= conDelta:
            #print(x)
            return True
    else:
        #print(fun(x))
        ans += np.linalg.norm(fun(x))
        if ans <= conDelta:
            return True
    return False


def solve(x):
    xsteps = []
    ysteps = []
    global delta
    x0 = x[:2].copy()
    x1 = x[2:].copy()
    cnt = 0
    prev_delt = None
    for i in range(100):
        ff = fun(x1)
        dH = fun(x1)-fun(x0)
        dXY = (x1-x0).reshape(1,2)
        t = (-ff/dH).reshape(2,1)
        diag = np.matmul(t, dXY)
        step = np.diag(diag)
        """
        
        print(diag)
        print(np.linalg.matrix_rank(diag))
        print("////////")
        
        print(np.diag(step))
        print(np.linalg.matrix_rank(np.diag(step)))
        print("loc", x1)
        print("val",fun(x1))
        #input()
        """
        #step = np.diag(np.fliplr(diag))
        x2 = (x1 + step)
        xsteps.append(x2[0])
        ysteps.append(x2[1])
        #print("X", x2)
        cnt += 1
        if np.linalg.norm(step) <= delta:
           break
        
        x0 = x1.copy()
        x1 = x2.copy()
    #return xsteps, ysteps, x1
    if check_root(np.round(x2, 3)):
        return xsteps, ysteps, x1
        return np.round(x2, 2).tolist(), cnt
    return None,None, cnt

a,b,c= solve(np.array([3,2,3.5,1]))
print(c)
plt.plot(a,b,'k')
for k,(i,j) in enumerate(zip(a,b)):
    #plt.annotate('%s)' %j, xy=(i,j), xytext=(30,0), textcoords='offset points')
    plt.annotate('%s,' %k, xy=(i,j))
plt.scatter([-1,3],[2,1],c='red')
plt.scatter(c[0],c[1], c='orange')
print(repr(a))
print(repr(b))
plt.show()
