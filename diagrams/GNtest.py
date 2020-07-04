
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import pinv
#I = 2.340 *10**7 #mm4
L = 2 #m
P = 10000 #N
E = 200*(10**9)
I0 = 2000 * (100**-4)
def func(x , I):
    global L
    global P
    global E
    return 1000*P*(x**2)*(3*L-x)/(6*E*I)

def der(x,I):
    global L
    global P
    global E
    return 1000*P*(x**2)*(3*L-x)/(6*E*I**2)

y = np.array([0.,0.49,1.781,3.606,5.698])
x = np.array([0.,0.5,1.,1.5,2])
for i in range(4):
    r = y - func(x, I0)
    print("Residual", r)
    ri = der(x, I0)
    rhs = np.einsum("i,i",-r, ri)
    lhs = np.einsum("i,i",ri,ri)
    step = rhs/lhs
    print("Step",step*100000000)
    print("Theta", I0 *100000000)
    if abs(step) < 10**-5:
        print("success")
        print(I0)
    I0 +=step
    
