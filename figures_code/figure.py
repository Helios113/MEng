import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

def f(x):
    return (x+2)*x*(x-1)


x = np.linspace(-2.5,2.5,100,endpoint=True)
y = f(x)


fig, ax = plt.subplots()

ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position('zero')

# Eliminate upper and right axes
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

# Show ticks in the left and lower axes only
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')



xticks = ax.xaxis.get_major_ticks() 
xticks[3].label1.set_visible(False)

yticks = ax.yaxis.get_major_ticks() 
yticks[2].label1.set_visible(False)

plt.plot(x,y, 'k')

plt.plot(0,0,'ro') 
plt.plot(-2,0,'ro') 
plt.plot(1,0,'ro') 



plt.savefig('cubic.png', transparent=True)
plt.show()