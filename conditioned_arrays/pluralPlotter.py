import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from mpl_toolkits.axes_grid1 import ImageGrid

colors = plt.get_cmap("tab20c")
outer_colors = colors(np.arange(5)*4)
outer_colors = np.vstack(([0, 0, 0, 0], outer_colors))




ans = []
FILE_PATH = "conditioned_arrays/FN1-['None'].npy"
with open(FILE_PATH, "r") as file:
    ans.append(np.load(FILE_PATH, allow_pickle=False))
FILE_PATH = "conditioned_arrays/FN1-0.5x.npy"
with open(FILE_PATH, "r") as file:
    ans.append(np.load(FILE_PATH, allow_pickle=False))
FILE_PATH = "conditioned_arrays/FN1-['2x,3x'].npy"
with open(FILE_PATH, "r") as file:
    ans.append(np.load(FILE_PATH, allow_pickle=False))
FILE_PATH = "conditioned_arrays/FN1-['1'].npy"
with open(FILE_PATH, "r") as file:
    ans.append(np.load(FILE_PATH, allow_pickle=False))

plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 2


fig = plt.figure(figsize=(8, 7), dpi=100,)
grid = ImageGrid(fig, 111,
                nrows_ncols = (2,2),
                axes_pad =0.4
                )

img = grid[0].imshow(ans[0], origin='lower', extent=[-50, 50, -50, 50])
grid[1].imshow(ans[1], origin='lower', extent=[-50, 50, -50, 50])
grid[2].imshow(ans[2], origin='lower', extent=[-50, 50, -50, 50])
grid[3].imshow(ans[3], origin='lower', extent=[-50, 50, -50, 50])
grid[2].set_xlabel(r'$x_0$')
grid[3].set_xlabel(r'$x_0$')
grid[0].set_ylabel(r'$x_1$')
grid[2].set_ylabel(r'$x_1$')
grid[0].set_title(r'$A$')
grid[1].set_title(r'$B$')
grid[2].set_title(r'$C$')
grid[3].set_title(r'$D$')

num = 1
maxx = 49.0
minn = 2
bbox_ax = grid[0].get_position()
print(bbox_ax)
bbox_ax1 = grid[1].get_position()
#cbar_im1a_ax = fig.add_axes([1.01, bbox_ax.y0, 0.02, bbox_ax1.y1-bbox_ax.y0])

print(bbox_ax)

for i in range( 1,num+1):
    change1 = outer_colors[i].copy()
    col1 = np.array([change1, ]*12)
    col1[..., -1] = np.linspace(0.33, 1, num=12)
    newcmp = ListedColormap(col1)
    norm = mpl.colors.Normalize(vmin=minn, vmax=maxx)
                       
    cbar_im1a_ax = fig.add_axes([0.85+(i*0.02), bbox_ax.y0, 0.02, bbox_ax1.y1-bbox_ax.y0])
    print(cbar_im1a_ax)
    if i == int(num):
        cb = fig.colorbar(cm.ScalarMappable(cmap=newcmp, norm=norm),cax = cbar_im1a_ax)
        cb.ax.set_ylabel('Iterations')
    else:
        cb = fig.colorbar(cm.ScalarMappable(cmap=newcmp, norm=norm), ticks=[], cax = cbar_im1a_ax)

    cb.ax.set_xlabel(r'$r_{{{}}}$'.format(i))
    #artists.append(cb)
plt.show()