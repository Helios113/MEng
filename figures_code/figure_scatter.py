import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

def f(x):
    return 33*x-216


x = [14.2, 16.4,11.9,15.2,18.5]
y = [215, 325, 185, 332, 406]
x1 = [12 , 20]
y1 = [f(12),f(20)]

fig, ax = plt.subplots()
plt.plot([15.2, 15.2], [332, f(15.2)], '--', c = 'r')
plt.scatter(x,y, c = 'orange')
plt.plot(x1,y1, c = 'k')

a = 3/8
b = 125/280
"""plt.annotate(r"$\{$",fontsize=50,
            xy=(a, b), xycoords='figure fraction'
            )"""
"""plt.annotate("residual",fontsize=24,
            xy=(a*0.45, b*1.05), xycoords='figure fraction'
            )"""


plt.savefig('cubic.png', transparent=True)
plt.show()