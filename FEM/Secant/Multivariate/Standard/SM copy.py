from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

delta = 1e-6
conDelta = 0.001414


def fun(x):
    #return np.array([np.exp(x[0])-x[1], x[0]*x[1]-np.exp(x[0])]) #  1
    #return np.array([x[0]**2-x[1]**2-9, 2*x[0]*x[1]])  #  2
    #return np.array([x[0]**2-x[1]**3-x[0]*x[1]**2-1, x[0]**3-x[1]*x[1]**3-4])  #  3
    #return np.array([np.sin(x[0])*np.exp(x[0])+np.sin(x[1])*np.exp(x[1])-10, x[0]+x[1]])  #  4
    return np.array([x[0]**3-3*x[0]*x[1]**2-1, 3*x[0]**2*x[1]-x[1]**3]) #  5



def check_root(x):
    global conDelta
    #ans = np.array([1, 2.718])
    ans = 0
    if isinstance(ans, np.ndarray):
        if np.linalg.norm(ans-x.flatten()) <= conDelta:
            return True
    else:
        print("ANS", fun(x))
        ans += np.linalg.norm(fun(x))
        if ans <= conDelta:
            return True
    return False


def solve(x):
    global delta
    x0 = x[:2].copy()
    x1 = x[2:4].copy()
    cnt = 0
    for i in range(50):
        #print("ITERATION:", i,"////////////////")
        print("x1, x0", x1 , x0)
        ff = fun(x1)
        print("fun(x1)", ff)
        #print("x0", x0)
        #print("fun(x0)", fun(x0))
        dH = fun(x1)-fun(x0)
        dXY = (x1-x0)#.reshape(-1, 2)
        A = np.append(x1, ff[0])
        B = np.append(x1, ff[1])
        dA = np.append(dXY,dH[0])
        dA = dA / np.linalg.norm(dA) 
        dB = np.append(dXY,dH[1])
        dB = dB / np.linalg.norm(dB) 
        u = A - B
        b = np.dot(dA, dB)
        if abs(b) == 1: # better check with some tolerance
            print("lines are parallel")
        d = np.dot(dA, u)
        e = np.dot(dB, u)
        t_intersect = ((b * e) - d) / (1 - (b * b))
        P = A + t_intersect * dA
        dP = ((dA - dB)/ np.linalg.norm(dA - dB))
        v = -P[2]/dP[2]
        print("point",P ,dP, v)
        bis1 = P + (v * dP)
        print("bis" ,bis1)
        """"dXY = (x1-x0)#.reshape(-1, 2)
        #print("dH", dH)
        t = np.zeros((2, 1))
        t = (-ff/dH).reshape(2, -1)
        #  t = np.divide(-ff, dH, where=dH!=0).reshape(2,-1)
        #print("t", t)
        ##print("t_row", t[i % 2])
        
        #print("dXY", dXY)

        #delt = np.matmul(t, dX)
        #step = (dXY * t[i%2]).flatten()
        step = (dXY * t).flatten()
        ##print("delt:",delt)
        #step = delt.sum(axis=0).flatten()
        #print("Step:",step)
        """




        x2 = bis1[0:2]
        cnt += 1
        #if np.linalg.norm(step) <= delta:
        #    break
        print("x2", x2)
        print("ans", fun(x2))
        x0, x1 = x1, x2
        #print("ITERATION END ////////////////")
        
    if check_root(x2):
        return np.round(x2, 2).tolist(), cnt
    return None, cnt

solve(np.array([3,1,5,4]))
