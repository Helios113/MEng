import numpy as np
from numpy.linalg import pinv
delta = 1e-9
class solve:

    def __init__(self, force_stiffness, c):
        self.force_stiffness = force_stiffness
        self.c = c
    

    def P(self, x):
        a = x[1:]-self.c[1:]
        q = self.force_stiffness(self.c,1)
        d = self.force_stiffness(x, 1)
        #print("Check here",d)
        o = (d/(d-q)).copy()
        #d = np.divide(d, (d-q), out=np.zeros_like(d), where=(d-q)!=0)
        return -np.outer(o, a), a, o


    def getPartial(self,x):
        dxk = x[1:]-self.c[1:]
        try:
            fi,fij = self.force_stiffness(x, 2)
        except:
            return np.zeros((len(x)-1,(len(x)-1)**2)).T
        dFi = fi-self.force_stiffness(self.c,1)
        
        dfij = ((dFi -fi)/(dFi**2))*np.identity(len(x)-1)
        #dfij = np.divide((dFi -fi), (dFi**2), out=np.zeros_like(dFi), where=dFi!=0)*np.identity(len(x)-1)
        d = np.einsum("ij,jk",dfij,fij)
        e = np.einsum("i, jk", dxk, d)
        e = e.T.reshape(len(x)-1,-1, order = 'C')
        f = fi/dFi
        #f = np.divide(fi, dFi, out=np.zeros_like(fi), where=dFi!=0)
        g = np.einsum("ik, j",np.identity(len(x)-1),f)
        g = g.reshape(len(x)-1,-1, order = 'C')
        e+=g
        return e.T
        
    def check_root(self, x):
        #ans = np.array([1, 2.718])
        conDelta = 0.001
        ans = 0
        if isinstance(ans, np.ndarray):
            if np.linalg.norm(ans-x.flatten()) <= conDelta:
                return True
        else:
            ans += np.linalg.norm(self.force_stiffness(x,1))
            if ans <= conDelta:
                return True
        return False

    def execute(self, x ):
        global delta
        #np.set_printoptions(precision=4)
        for i in range(100):
            q,inf1, inf2 = self.P(x)
            #print("Q", q)
            q = q.reshape(-1, 1)
            p = pinv(self.getPartial(x)) 
            step = np.matmul(p, q).flatten()
            
            x = x + np.insert(step,0,0)
            fvec = np.linalg.norm(self.force_stiffness(x,1))
            
            print("x-c",inf1)
            print("fac",inf2)
            print("Step",step)
            print("|Step|", np.linalg.norm(step))
            print("X",x)
            print("q", q.flatten())
            print("|q|", np.linalg.norm(q))
            print("F", self.force_stiffness(x,1))
            print("|F|",fvec)
            input()
            
            if fvec < 0.005:
                print("Result", i)
                return x, i
            
            #print("Disp", x)
        #print(x)
        return None , 0

        