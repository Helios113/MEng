
import numpy as np
from numpy.linalg import pinv
from functools import partial 

delta = 1e-4


#functions
def f1(x):
    return np.exp(x[0])-x[1]

def f2(x):
    return x[0]*x[1] - np.exp(x[0])

x1 = np.array([2.44701926,0.14712591]).reshape((2,1))
x0 = np.array([-5.45099591,10.13973042]).reshape((2,1))
c = np.array([3,4]).reshape((2,1))
F = [f1,f2]

def P(c,g,x):
    a = x-c
    b = g(x)-g(c)
    d = g(x)
    return (d/b)*a




list = []
for i in F:
    list.append(partial(P,c,i))

def getFmatrix(x):
    global list
    hs = None
    for i in list:
        ans = i(x)
        hs = np.hstack((hs,ans)) if hs is not None else ans
    print('####### F matrix ########')
    p = hs.copy()
    return p
#den = inverce of F matrix

fx1 = getFmatrix(x1)
fx0 = getFmatrix(x0)
sub = fx1-fx0
den = np.linalg.inv(sub)

num = x1-x0
div0 = np.matmul(den, fx1)
div1 = np.matmul(fx1, den)
print('####### 0 ########')
print(div0)
print('####### 1 ########')
print(div1)
#print(num)
ans00 = np.matmul(div0,num)
ans01 = np.matmul(div0.T,num)
print('####### 00 step matrix ########')
print(x1-ans00)
print('####### 01 step matrix ########')
print(x1-ans01)

ans10 = np.matmul(div1,num)
ans11 = np.matmul(div1.T,num)
print('####### 10 step matrix ########')
print(x1-ans10)
print('####### 11 step matrix ########')
print(x1-ans11)