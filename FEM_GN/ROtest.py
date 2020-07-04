from model import func, der_func, der2_func
import numpy as np
from myGNE import GNSolver #
from datetime import datetime
from scipy.interpolate import griddata
import lhsmdu #latin hypercube
import matplotlib.pyplot as plt
COEFFICIENTS = [-0.001, 0.1, 0.1, 2, 15]
x = np.linspace(1,10,10)
y = func(x,COEFFICIENTS)
maxIterations = 100
Ans = None
cnt = 0;
coeff = 1         
SNR = -0.01
NOISE= np.abs(np.mean(y))/(10**(SNR/10))-np.abs(np.mean(y))
DISTANCE = 10
yn = y + NOISE * (np.array(lhsmdu.sample(y.shape[0],1)).flatten())
init_guess = [10,10,10,10,10]
g = GNSolver(func, der_func, der2_func, original_root=COEFFICIENTS,max_iter = maxIterations)
a = g.fit(x,yn,init_guess, True)
#print(a[6])
print(repr(a[4]))
print(repr(a[5]))
plt.plot(a[4])
plt.plot(a[5])
plt.show()