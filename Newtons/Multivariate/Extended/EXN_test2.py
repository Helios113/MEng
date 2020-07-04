import numpy as np
from numpy.linalg import pinv
delta = 1e-9
class solve:

    def __init__(self, f, c):
        self.force_stiffness = f
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
        dxk = x-self.c
        fi,fij = self.force_stiffness(x, 2)
        dFi = fi-self.force_stiffness(self.c,1)
        
        dfij = ((dFi -fi)/(dFi**2))*np.identity(2)
        d = np.einsum("ij,jk",dfij,fij)
        print(np.linalg.matrix_rank(d))
        e = np.einsum("i, jk", dxk, d)
        
        e = e.T.reshape(len(x),-1, order = 'C')
       
        f = fi/dFi
        g = np.einsum("jk, i",np.identity(len(x)),f)
        g = g.T.reshape(len(x),-1, order = 'F')
        e+=g
        #e = np.transpose(e, (1, 2, 0))
        #raise ValueError("stop")
        return e.T
        

    def execute(self, x ):
        global delta
        for i in range(200):
            q = self.P(x).reshape(-1,1)
            #print(q)
            p = pinv(self.getPartial(x)) 
            
            step = np.matmul(p, q).flatten()
            x = x + step
            if np.linalg.norm(step) < delta:
                #print("q", f(x,1))
                #print("step-1",step)
                return x
            print("Disp", x)
        return x

def f(x, n):
    r = np.array([np.exp(x[0])-x[1], x[0]*x[1]-np.exp(x[0])])
    if n==2:
        k = np.array([np.exp(x[0]), -1, x[1]-np.exp(x[0]), x[0]]).reshape(2, 2)
        return r,k
    return r



np.random.seed(10)
x = np.array([1.,1.])
c = np.array([1.,2.])
while not np.all(c[:-1] <= c[1:]):
    print("stuck", c)
    c = x+(np.random.random(x.shape))
s = solve(f,c)
s.execute(x)