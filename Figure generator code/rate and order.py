import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 25}
mpl.rc('font', **font)
y = np.array([0.8951067538564371, 0.950299304069867, 0.9889447188157281, 0.995392766238411, 0.9909477002352775, 0.9822584645966014, 0.9659766839990267, 0.9378626300373993, 0.9002002122098787])
#y = np.array([0.49378077808636583, 0.48642439787288844, 0.4951915109339076, 0.4989090772763069, 0.49482336461232984, 0.4840356894041302, 0.45933278807429323, 0.41179613722989583, 0.3471847924699639])
x = np.arange(len(y))+3
y1 = np.array([0.9413899913375077, 0.8999577332053864, 0.9291282917922126, 0.9661783080653855, 0.9847395845136822, 0.9969857593703533, 0.9984041314114857, 0.9976469512325291, 0.9964834316238094, 0.9947544839675736, 0.9921980109275387, 0.9884471001518005, 0.9830115489691906, 0.9752953538963604, 0.9647318403732544, 0.9512588702288035, 0.9367620878517912])
#y1 = np.array([0.6472682381072913, 0.6823203230483272, 0.6672960320853797, 0.6598059690235575, 0.6612949703597202, 0.6656267022349144, 0.6665133990620659, 0.6658360897783437, 0.6644845689226604, 0.6620203448704954, 0.6577184893228667, 0.6504828098413606, 0.6387621109441761, 0.620608313224649, 0.594169590524802, 0.5592424463917514, 0.5212083536566388])
x1 = np.arange(len(y1))+3
fig, ax = plt.subplots()
img1 = plt.plot(x,y, c = 'k' ,label='CGN', linewidth=3.0)
img2 = plt.plot(x1,y1,"--", c = 'k', label='GN',linewidth=3.0)
#plt.plot(x,y1, c = 'k')
plt.legend(fontsize='x-large')



#ax.spines['left'].set_position('zero')
#ax.spines['bottom'].set_position('zero')

# Eliminate upper and right axes
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

# Show ticks in the left and lower axes only
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
yticks = ax.yaxis.get_major_ticks() 
#yticks[1].label1.set_visible(False)
xticks = ax.xaxis.get_major_ticks() 
#xticks[1].label1.set_visible(False)
ax.set_xlabel("Iteration")
#ax.xaxis.set_label_position('top') 
ax.set_ylabel("Order")
"""plt.annotate(r"$\{$",fontsize=50,
            xy=(a, b), xycoords='figure fraction'
            )"""
"""plt.annotate("residual",fontsize=24,
            xy=(a*0.45, b*1.05), xycoords='figure fraction'
            )"""


#plt.savefig('cubic.png', transparent=True)
plt.show()