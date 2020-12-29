import numpy as np
from numpy.linalg import pinv
delta = 1e-9
class solve:

    def __init__(self, force_stiffness, c):
        self.force_stiffness = force_stiffness
        self.c = c
       

    def execute(self, x ):
        global delta
        for i in range(300):
            f,j = self.force_stiffness(x,2)
            b = np.linalg.inv(j).dot(f)
            x = x - np.insert(b,0,0)
            if np.linalg.norm(b) < delta:
                if check_root(x):
                    return x
            #print("Disp", x)
        return None