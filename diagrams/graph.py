import numpy as np
import matplotlib.pyplot as plt
#203x133x25

L = 2 #m
P = 10000 #N
E = 200*(10**9) #Pa
I = 2340 * (100**-4) #m^4
"""
L = 2000 #mm
P = 10 #kN
E = 200 #GPa
I = 2.340 *10**7 #mm4
"""
def func(x):
    global L
    global P
    global E
    return P*(x**2)*(3*L-x)/(6*E*I)

x = np.linspace(0,L, 5)
y= func(x)*1000
yn = y + (np.random.rand(5)-0.5)
print(x)
print(np.round(y,3))
plt.plot(x,y)
plt.scatter(x,yn)
plt.show()