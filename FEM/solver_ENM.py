import numpy as np
from numpy.linalg import pinv
delta = 1e-6
class solve:

    def __init__(self, force_stiffness, c):
        self.force_stiffness = force_stiffness
        self.c = c
    

    def P(self, x):
        a = x[1:]-self.c[1:]
        q = self.force_stiffness(self.c,1)
        d = self.force_stiffness(x, 1)
        b= d-q
        #print("P function",x,d,b,q, a)
        #raise ValueError("stop")
        return np.matmul(a.reshape(-1,1), (d/b).reshape(1,-1)).T

    def getPartial(self,x):
        #global c
        el1 = x[1:]-self.c[1:]
        el2,gr = self.force_stiffness(x, 2)
        el3 = el2-self.force_stiffness(self.c,1)
        gr = gr.T
        #print("GR",gr)
        num2 = el2*gr
        num1 = el3*gr
        den = el3**2

        #d = np.divide((num1-num2), den, out=np.zeros_like(num1), where=den!=0)
        #print("Partial",el1,el2,el3,gr)
        #print(self.force_stiffness(self.c,1))
        ans = np.outer(el1, ((num1-num2)/den).T)
        ans=ans.T.reshape(-1,len(x)-1,len(x)-1)
       

        
        #a = np.divide(el2, el3, out=np.zeros_like(el2), where=el3!=0)
        ans1 = np.outer(np.identity(len(x)-1), el2/el3).T   
        ans1 = ans1.reshape(-1,len(x)-1,len(x)-1)

        ans = ans + ans1
        ansFin = None
        for i in range(len(x)-1):
            ansFin = np.hstack((ansFin,ans[i])) if ansFin is not None else ans[i]
        #print(ansFin)
        #input()
        
        return ansFin.T

    def execute(self, x ):
        global delta
        for i in range(100):
            q = self.P(x).reshape(-1, 1)
            p = pinv(self.getPartial(x)) 
            step = np.matmul(p, q)
            if np.linalg.norm(step) < delta:
                break
            x = x - np.insert(step.flatten(),0,0)
        return x