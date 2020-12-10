import numpy as np


def fun(x):
    return np.exp(x)-500
def grad_fun(x):
    return np.exp(x)

def mod_fun(x):
    return np.abs(x)/(fun(x)-fun(0))
def grad_mod(x):
    return -(x*(x*grad_fun(x)-fun(x)+fun(0)))/(np.abs(x)*(fun(x)-fun(0))**2)

def f(x):
    return mod_fun(x)*fun(x)

def grad_f(x):
    return grad_mod(x)*fun(x)+grad_fun(x)*mod_fun(x)

def a(x):
    global c
    #print((x-c)*fun(x))
    #print(fun(x)-((x-c)*grad_fun(x)*fun(c)/(fun(x)-fun(c))))
    return ((x-c)*fun(x))/(fun(x)-((x-c)*grad_fun(x)*fun(c)/(fun(x)-fun(c))))

c = 5


y = -20
print("std")
for i in range(1000):
    step1 = fun(y)/grad_fun(y)
    y1 = y - step1
    y = y1
    #print(y)
    if np.abs(step1) < 0.01:
        break
print(y)


z = -20
print("modified")
for i in range(1000):
    #print(z)
    step = f(z)/grad_f(z)
    z1 = z - step
    z = z1
    #print(z)
    if np.abs(step) < 0.01:
        break
print(z1)




q = -20
print("ankush")
for i in range(1000):
    #print(z)
    step2 = a(q)
    q1 = q - step2
    q = q1
    if np.abs(step2) < 0.01:
        break

print(q1)

