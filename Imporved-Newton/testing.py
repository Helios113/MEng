
import numpy as np
from numpy.linalg import pinv
from functools import partial 

delta = 1e-4



#functions
def f1(x):
    return np.exp(x[0])-x[1]

def f2(x):
    return x[0]*x[1] - np.exp(x[0])

F = [f1,f2]
F = np.asanyarray(F).reshape(2,1)
n = np.arange(8).reshape((2,4))
#x = np.zeros((2,1))
#c = np.zeros((2,1))
x = np.array([1.01252171,2.5314747]).reshape((2,1))
c = np.array([3,4]).reshape((2,1))

def P(c,g,x):
    a = x-c
    b = g(x)-g(c)
    d = g(x)
    return (d/b)*a



def getPartial(func, i, x):
    global delta
    a = x.copy()
    a[i] -= delta
    b = x.copy()
    b[i] += delta
    fa = func(a)
    fb = func(b)
    return (fb-fa)/(2*delta)


list = []
for i in F:
    list.append(partial(P,c,*i))



#Assembling Jacobian
vs = None
for i in list:
    hs = None
    for j in range(len(x)):
        ans = getPartial(i,j,x)
        hs = np.hstack((hs,ans)) if hs is not None else ans
    vs = np.vstack((vs,hs)) if vs is not None else hs

#print('####### Jacobian ########')
#print(vs)
print('####### Jacobian inverse ########')
ji = pinv(vs)
print(ji)

print('####### X ########')
print(x)

print('####### Jacobian inverse 3D ########')
#the right assembley
jii = np.transpose(np.reshape(ji,(2,2,2)),(0,2,1))
print(jii)
print(jii[0,0,1])
hs = None
for i in list:
    ans = i(x)
    hs = np.hstack((hs,ans)) if hs is not None else ans
print('####### F matrix ########')
p = hs.T.copy()
print(p)
print('####### New X ########')
ans = np.einsum(jii,[0,1,2],p,[2,1]).reshape((2,1))
print(x-ans)
