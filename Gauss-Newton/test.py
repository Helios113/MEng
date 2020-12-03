from model import func, der_func, der2_func
import numpy as np
from myGNE import GNSolver
from datetime import datetime
from scipy.interpolate import griddata
 

COEFFICIENTS = [0.1,4,0.1,2]
np.random.seed(10)

#x = np.linspace(0,20,20)
x = np.mgrid[1:20:20j, 1:20:20j].reshape(2, -1).T
y = func(x,COEFFICIENTS)

Ans = []
for NOISE in np.linspace(0,4,8):
    for DISTANCE in np.linspace(0,6,12):
        for r in range(1):
            yn = y + NOISE * np.random.random_sample(y.shape)
            init_guess = (np.random.random(len(COEFFICIENTS))-0.5)*(10**DISTANCE)
            dist = np.linalg.norm(COEFFICIENTS - init_guess)
            ans = []
            g = GNSolver(func,der_func,der2_func,500)
            startTime = datetime.now()
            a = g.fit(x,yn,init_guess, True)
            r = g.get_rmse(a[0])
            ans.append([NOISE, dist,DISTANCE, r, a[1]])
        ans =np.mean(np.array(ans),axis=0)
        Ans.append(ans)
            #print(ans[0], ans[1], datetime.now() - startTime, a[0])


Ans = np.array(Ans)
print(Ans.shape)
with open('Gauss-Newton/test4.npy', 'wb') as f:
    np.save(f, Ans)

