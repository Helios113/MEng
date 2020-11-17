from myGNE import GN
import numpy as np
import matplotlib.pyplot as plt
def func(x, coeff):
    return coeff[0] * x ** 3 + coeff[1] * x ** 2 + coeff[2] * x + coeff[3] + coeff[4] * np.sin(x)


COEFFICIENTS = [-0.001, 3, 3, 2, 15]

x = np.arange(1, 7)
y = func(x, COEFFICIENTS)
init_guess = [1,1,1,1,1]

a = GN(fit_function=func)
ANSWER = a.fit(x,y,init_guess)
#print(ANSWER[2])

fig, axs = plt.subplots(2)
axs[0].plot(ANSWER[2])
axs[0].set_title('Original step')
axs[1].plot(ANSWER[3])
axs[1].set_title('Corrected step')
plt.show()