import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

L = 2 #m
P = 10000 #N
E = 200*(10**9)
I0 = 2339.921 * (100**-4)
def func(x , I):
    global L
    global P
    global E
    return 1000*P*(x**2)*(3*L-x)/(6*E*I)


y = np.array([0.,0.49,1.781,3.606,5.698])
x = np.array([0.,0.5,1.,1.5,2])
y1 = func(x,I0)

fig, ax = plt.subplots()
plt.scatter(x,y, c = 'orange')
plt.plot(x,y1, c = 'k')

ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')

# Eliminate upper and right axes
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

# Show ticks in the left and lower axes only
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
plt.gca().invert_yaxis()
yticks = ax.yaxis.get_major_ticks() 
yticks[1].label1.set_visible(False)
xticks = ax.xaxis.get_major_ticks() 
xticks[1].label1.set_visible(False)
ax.set_xlabel("Length [m]")
ax.xaxis.set_label_position('top') 
ax.set_ylabel("Deflection [mm]")
"""plt.annotate(r"$\{$",fontsize=50,
            xy=(a, b), xycoords='figure fraction'
            )"""
"""plt.annotate("residual",fontsize=24,
            xy=(a*0.45, b*1.05), xycoords='figure fraction'
            )"""


#plt.savefig('cubic.png', transparent=True)
plt.show()