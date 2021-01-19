import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np




#y = np.array([0, 0, 0, 0, 0, 0, 0, 4, 4, 6, 11, 5, 7, 7, 14, 9, 3, 3, 3])
#x = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0])

x = np.array([3.120845921450151, 3.3014869592198317, 3.2035088843665522, 1.8331714370648011, 1.5655437379826576, 1.238738086068174, 1.0866929788070212, 1.0169457109089464, 1.001373196806672, 1.000022974025897, 1.0000000313405157])
y = np.array([-2.4878048780487805, -0.3401109088992009, 0.17493063716116553, 0.00660273225546229, 0.004494080527794139, 0.002415478217104313, 0.0013612728514751035, 0.0005538859859745458, 0.00010691609764047598, 3.9975242388089346e-06, 1.1190279910096234e-08])


fig, ax = plt.subplots()
plt.plot(x,y, c = 'k')
for k,(i,j) in enumerate(zip(x[:6],y[:6])):
    plt.annotate('%s,' %k, xy=(i,j+0.03), textcoords='offset points')
#plt.plot(x,y1, c = 'k')
plt.scatter(1,0, c='orange')
#ax.spines['left'].set_position('zero')
#ax.spines['bottom'].set_position('zero')

# Eliminate upper and right axes
#ax.spines['right'].set_color('none')
#ax.spines['top'].set_color('none')

# Show ticks in the left and lower axes only
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
yticks = ax.yaxis.get_major_ticks() 
#yticks[1].label1.set_visible(False)
xticks = ax.xaxis.get_major_ticks() 
#xticks[1].label1.set_visible(False)
ax.set_xlabel(r"$x_0$",fontsize = 15)
#ax.xaxis.set_label_position('top') 
ax.set_ylabel(r"$x_1$",fontsize = 15)
"""plt.annotate(r"$\{$",fontsize=50,
            xy=(a, b), xycoords='figure fraction'
            )"""
"""plt.annotate("residual",fontsize=24,
            xy=(a*0.45, b*1.05), xycoords='figure fraction'
            )"""


#plt.savefig('cubic.png', transparent=True)
plt.show()