
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.interpolate import griddata
from matplotlib.colors import Normalize
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib import cm
import matplotlib.ticker as mtick

with open('FEM_GN/f_MR2_gn.npy', 'rb') as f:
    Ans = np.load(f) 
with open('FEM_GN/f_MR2_cgn.npy', 'rb') as f:
    Ans1 = np.load(f) #corrected



#Ans[np.where(Ans[:,4,:]!=0)] +=[0,0,0,0,1]

"""
#Bar Chart Plotting
print(Ans[:, 0])
noise = 0
workList = Ans[np.where(Ans[:, 0] == noise)]
dist = workList[:,1]
steps = workList[:,2]

width = (dist[1]-dist[0])
fig, ax = plt.subplots()
ax.bar(dist, steps, width,edgecolor = 'k' ,linewidth=0.1)

ax.set_ylabel('Steps')
ax.set_xlabel(r'Distance 10^n')
ax.set_xticks(dist)
plt.show()
"""


#image showing all values

viridis = cm.get_cmap('viridis_r', 100)
newcolors = viridis(np.linspace(0, 1, 256))
newcmp = ListedColormap(newcolors)
stepLimit = 100
vmax = np.max([Ans[:,2],Ans1[:,2]])
cmap = newcmp#mpl.cm.get_cmap("YlOrBr")
cmap.set_bad(color='white')
fig = plt.figure(figsize=(17,6))


font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 15}

mpl.rc('font', **font)
grid = ImageGrid(fig, 111,
                nrows_ncols = (1,2),
                axes_pad = 1,
                cbar_location = "right",
                cbar_mode="single",
                cbar_size="5%",
                cbar_pad=0.05
                )
def pl(Ans):
    noise = Ans[:, 0]
    dist = Ans[:, 1]
    steps = Ans[:, 2]
    print(steps)
    grid_x, grid_y = np.mgrid[0:np.max(dist):1000j, 0:np.max(noise):1000j]
    points = (dist, noise)
    grid_z0 = griddata(points, steps, (grid_x, grid_y), method='linear').T
    masked_array = np.ma.masked_where(grid_z0 ==0, grid_z0)
    return masked_array


mk1 = pl(Ans)
mk2 = pl(Ans1)
noise = Ans[:, 0]
dist = Ans[:, 1]
snr =np.flip(np.unique(np.round(Ans[:, 3],6)))
print(vmax)
#grid[0].imshow(mk1 ,extent=(0,np.max(dist),0,np.max(noise)), origin='lower', cmap=cmap,vmin = 1, vmax = vmax)
img = grid[0].contourf(mk1,extent=(0,np.max(dist),0,np.max(noise)),cmap=cmap,vmin = 1, vmax = vmax)
#img = grid[1].imshow(mk2 ,extent=(0,np.max(dist),0,np.max(noise)), cmap=cmap,vmin = 1, vmax = vmax)
img =  grid[1].contourf(mk2 ,extent=(0,np.max(dist),0,np.max(noise)), cmap=cmap,vmin = 1, vmax = vmax)

cbar = fig.colorbar(img, cax=grid.cbar_axes[0])
cbar.ax.set_ylabel("Number of steps",fontsize = 30)

labelsx = [f'$10^{ {int(i)} }$' for i in np.arange(0,np.max(dist)+1, step=1 )]
grid[0].set_xticks(np.arange(0,np.max(dist)+1, step=1 ))
grid[0].set_xticklabels(labelsx)
grid[0].set_xlabel("Distance",fontsize = 30)
grid[0].set_ylabel("SNR [dB]",fontsize = 30)
"""
labelsy = [f'{"%.2E" % i}dB' for i in snr]
ll = []
for i in labelsy:
    print(i)
    if i[0] == '-':
        a = float(i[0:5])
        b = int(i[6:9])
    else:
        a = float(i[0:4])
        b = int(i[5:8])
    if a == 0:
        ll.append(r'$ {a}E^{{{c}}}$'.format(a=a,c=b))
    else:
        ll.append(r'$ {a}E^{{{c}}}$'.format(a=a,c=b))
"""
ll = [f'{np.round(i,3)}' for i in snr]
grid[0].set_yticklabels(ll)

grid[0].set_title('GN', fontsize = 40)


grid[1].set_xticks(np.arange(0,np.max(dist)+1, step=1 ))
grid[1].set_xticklabels(labelsx)
grid[1].set_xlabel("Distance",fontsize = 30)
grid[1].set_title('CGN', fontsize = 40)
fig.tight_layout()
plt.show()


