from model import func, der_func, der2_func
import numpy as np
from myGNE import GNSolver #
from datetime import datetime
from scipy.interpolate import griddata
import lhsmdu #latin hypercube
import tools
COEFFICIENTS = [2.4, 0.16]

x = np.linspace(2,6,15)
#x = np.mgrid[0.1:10:20j, 0.1:10:20j].reshape(2, -1).T
y = func(x,COEFFICIENTS)
maxIterations = 100
Ans = None
cnt = 0;
coeff = 20
coefff = 1
for NOISE in np.linspace(0,4*coeff,9):
    SNR = 10*np.log10(np.abs(np.mean(y))/(np.abs(np.mean(y))+NOISE))
    #NOISE = np.linalg.norm(y)/SNR - np.linalg.norm(y)
    for DISTANCE in np.linspace(0,6/coefff,13):
        ans = []
        for r in range(3):
            cnt+=1
            tools.printProgressBar(cnt, 3*13*9, prefix="Progress", suffix="Complete", length = 50)
            yn = y + NOISE * (np.array(lhsmdu.sample(y.shape[0],1)).flatten())
            init_guess = ((np.array(lhsmdu.sample(len(COEFFICIENTS),1)).flatten()))*(10**DISTANCE)
            g = GNSolver(func, der_func, der2_func, original_root=COEFFICIENTS,max_iter = maxIterations)
            startTime = datetime.now()
            a = g.fit(x,yn,init_guess, False)
            if a is not None:
                #print(a[5])
                r = g.get_mse(a[0])
            try:
                if a[1]!=maxIterations:
                    ans.append([NOISE/coeff,DISTANCE*coefff, a[1],SNR])
            except:
                continue
        if len(ans) == 0:
            ans = [np.array([NOISE/coeff,DISTANCE*coefff,0,SNR])]
        ans = np.mean(np.array(ans),axis=0)
        Ans = np.vstack((Ans,ans)) if Ans is not None else ans
        

print(Ans)
with open('FEM_GN/f_VW3_gn.npy', 'wb') as f:
    np.save(f, Ans)

