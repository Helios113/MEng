
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from scipy.interpolate import griddata
from matplotlib.colors import Normalize

with open('Gauss-Newton/test4.npy', 'rb') as f:
    Ans = np.load(f)

xAxis = Ans[:, 1]
xAxisLog = np.log10(Ans[:,1])
yAxis = Ans[:, 0]
alphas = Ans[:, 4]
print(alphas)

values = Ans[:, 3]
print(values)
points = (xAxisLog,yAxis)
#values/=max(values)
#values+=0.5
alphas/=np.max(alphas)
values/=np.max(values)
alphas+=0.1
grid_x, grid_y = np.mgrid[1:max(xAxisLog):200j, 0:max(yAxis):100j]

grid_z1 = griddata(points, values, (grid_x, grid_y), method='cubic')
grid_z2 = griddata(points, alphas, (grid_x, grid_y), method='cubic')


plt.imshow(grid_z1.T,alpha=grid_z2.T, extent=(1,max(xAxisLog),0,max(yAxis)), origin='lower')

plt.show()