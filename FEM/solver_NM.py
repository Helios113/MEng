import numpy as np
from numpy.linalg import pinv
delta = 1e-9
class solve:

    def __init__(self, force_stiffness, c):
        self.force_stiffness = force_stiffness
        self.c = c
       

    def execute(self, x ):
        global delta
        for i in range(100):
            f,j = self.force_stiffness(x,2)
            b = np.linalg.inv(j).dot(f)
            x = x - np.insert(b,0,0)
            print("Step",b)
            print("|Step|", np.linalg.norm(b))
            print("X",x)
            print("F", f)
            print("|F|",np.linalg.norm(f))
            input()
            if np.linalg.norm(f) < 0.005:
                print("Result", i)
                return x
            
        return None