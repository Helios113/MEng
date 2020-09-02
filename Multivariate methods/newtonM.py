import numpy as np
import Jacobian as jb




def f1(x):
    return np.exp(x[0])-x[1]

def f2(x):
    return x[0]*x[1] - np.exp(x[0])


L = [f1,f2]
a = np.array([1,1], float)
J = jb.Jacobian(L,a)

def Newton(L, x):
    #iteration mode only 50 iterations
    #print(jb.Jacobian(L,x))
    

    for i in range(50):
        b = np.linalg.inv(jb.Jacobian(L,x)).dot(f(L,x))
        x -= b
        print(f(L,x))
        #np.linalg.inv(jb.Jacobian(L,x)) # dot product the inverse Jacobian with the evaluated function and subtract from original


def f(L, x):
    ans = np.empty([len(L)])
    for j in range(len(L)):
        ans[j] = L[j](x) 
    return ans
    # for every fucntion evaluate it at vector x and return list of answers
Newton(L,a)