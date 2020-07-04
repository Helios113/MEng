import numpy as np
from numpy.linalg import pinv
from scipy.linalg import lu
delta = 1e-9
class solve:

    def __init__(self, force_stiffness, c):
        self.force_stiffness = force_stiffness
        self.c = c
    
    def P(self, x):
        a = x - self.c
        q = self.force_stiffness(self.c,1)
        d = self.force_stiffness(x, 1)
        d = d/(d-q)
        #print("This!!!!!",a, d)
        #print("Modified F")
        #print(-np.outer(a, d))
        return -np.outer(d, a)

    def getPartial(self,x):
        dcxk = x-self.c
        fi,fij = self.force_stiffness(x, 2)
        dcFi = fi-self.force_stiffness(self.c,1)
        
        #a = np.einsum("i,ij", fi,fij)
        dfij = ((dcFi -fi)/(dcFi**2))#*np.identity(2)
        print(np.linalg.matrix_rank(fij))
        #d = np.einsum("jl,jk",dfij,fij)
        #print(d)
        exit()
        #b = np.einsum("i,j", a,d)
        e = np.einsum("k, ij", dcxk, d)
        f = fi/dcFi
        g = f
        g = np.einsum("ik, j",np.identity(len(x)),g)
        g = g.reshape(len(x),-1, order = 'C')
        print(g)
        exit()
        e+=g
        #e = np.transpose(e, (1, 2, 0))
        #raise ValueError("stop")
        return e
        
    def execute(self, x ):
        global delta
        for i in range(300):
            q = self.P(x).reshape(-1)
            P = self.getPartial(x)
            print("Rank",np.linalg.matrix_rank(P))
            print("Rank",np.linalg.matrix_rank(q))
            exit()
            p = pinv(P) 
            print(q.shape)
            print(p.shape)
            step = np.einsum("l, li", q,p).flatten()
            x = x + step
            if np.linalg.norm(q) < delta:
                print("Disp", x)
                return x
        return x

def f(x, n):
    r = np.array([np.exp(x[0])-x[1], x[0]*x[1]-np.exp(x[0])])
    if n==2:
        k = np.array([np.exp(x[0]), -1, x[1]-np.exp(x[0]), x[0]]).reshape(2, 2)
        return r,k
    return r



np.random.seed(10)
x = np.array([2.,2.])
c = np.array([4.,9.])
while not np.all(c[:-1] <= c[1:]):
    c = x+(np.random.random(x.shape))
s = solve(f,c)
s.execute(x)