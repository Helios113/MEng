import numpy as np


def fun(x):
    return x**3-1

def grad_fun(x):
    return 3*x**2
def ggrad_fun(x):
    return 6*x

def mod_fun(x):
    return (1/(6*x))**0.5


def grad_mod(x):
    return -1/(2*6**0.5*x**(3/2))


def f(x):
    return mod_fun(x)*fun(x)


def grad_f(x):
    return grad_mod(x)*fun(x)+grad_fun(x)*mod_fun(x)
c = -6
def a(x):
    global c
    #print((x-c)*fun(x))
    #print(fun(x)-((x-c)*grad_fun(x)*fun(c)/(fun(x)-fun(c))))
    return ((x-c)*fun(x))/(fun(x)-((x-c)*grad_fun(x)*fun(c)/(fun(x)-fun(c))))




y = -5000
cnt=0
print("std")
for i in range(1000):
    cnt+=1
    step1 = fun(y)/grad_fun(y)
    y1 = y - step1
    y = y1
    #print(y)
    if np.abs(step1) < 0.01:
        break
print(y,cnt)


z = -5000
cnt=0
print("modified")
for i in range(1000):
    #print(z)
    cnt+=1
    step = (f(z)/grad_f(z)).real
    z1 = z - step
    z = z1
    #print(z)
    if np.abs(step) < 0.01:
        break
print(z1,cnt)

q = -5000
cnt = 0
print("ankush")
for i in range(3000):
    #print(z)
    cnt+=1
    step2 = a(q)
    q1 = q - step2
    q = q1
    if np.abs(step2) < 0.01:
        break

print(q1,cnt)