from model import func, der_func, der2_func
import numpy as np
from myGNE import GNSolver
from datetime import datetime
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import lhsmdu #latin hypercube
import tools
COEFFICIENTS = [-0.001, 0.1, 0.1, 2, 15]
NOISE = 1
DISTANCE = 3
x = np.linspace(1,20,20)
#x = np.mgrid[1:20:20j, 1:20:20j].reshape(2, -1).T  
y = func(x,COEFFICIENTS)
maxIterations = 100
Ans = None
cnt = 0;
ans = []
yn = y + NOISE * (np.array(lhsmdu.sample(y.shape[0],1)).flatten())
init_guess = ((np.array(lhsmdu.sample(len(COEFFICIENTS),1)).flatten()))*(10**DISTANCE)
g = GNSolver(func,der_func,der2_func,maxIterations,COEFFICIENTS)
startTime = datetime.now()
a = g.fit(x,yn,init_guess, False)
#a[5] is order of convergence based on q approx
#a[6] is rate approx based on en paper
print(a[1], a[5])
plt.plot(a[4])
plt.plot(a[5])
#plt.plot(a[6])
#plt.plot(np.log10(a[6]))


plt.show()

