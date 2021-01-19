from model import func, der_func, der2_func
import numpy as np
from myGNE import GNSolver
from datetime import datetime
from scipy.interpolate import griddata
import lhsmdu #latin hypercube
import tools

COEFFICIENTS = [-0.001, 0.1, 0.1, 2, 15]

x = np.linspace(0,20,20)
#x = np.mgrid[1:20:20j, 1:20:20j].reshape(2, -1).T
y = func(x,COEFFICIENTS)
maxIterations = 100
Ans = None
cnt = 0
et = []
for NOISE in np.linspace(0,4,9):
    for DISTANCE in np.linspace(0,6,13):
        ans = []
        for r in range(3):
            cnt+=1
            tools.printProgressBar(cnt, 3*13*9, prefix="Progress", suffix="Complete", length = 50)
            yn = y + NOISE * (np.array(lhsmdu.sample(y.shape[0],1)).flatten())
            init_guess = ((np.array(lhsmdu.sample(len(COEFFICIENTS),1)).flatten()))*(10**DISTANCE)
            
            g = GNSolver(func,der_func,der2_func,maxIterations)
            startTime = datetime.now()
            a = g.fit(x,yn,init_guess, False)
            if a is not None:
                #execution time
                et.append((datetime.now() - startTime).total_seconds())
                r = g.get_mse(a[0])
            try:
                if a[1]!=maxIterations:
                    ans.append([NOISE,DISTANCE, a[1]])
            except:
                continue
        if len(ans) == 0:
            ans = [np.array([NOISE,DISTANCE,0])]
        ans = np.mean(np.array(ans),axis=0)
        Ans = np.vstack((Ans,ans)) if Ans is not None else ans
        

print(np.average(et))
with open('Gauss-Newton/f22gn.npy', 'wb') as f:
    np.save(f, Ans)

