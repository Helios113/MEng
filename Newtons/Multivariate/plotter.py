import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


c = ["None"]
n = 512
m = 512
root = [0,0]
start = -100
stop = 100
f_index = 3

colors = plt.get_cmap("tab20c")
outer_colors = colors(np.arange(5)*4)
outer_colors = np.vstack(([0, 0, 0, 0], outer_colors))



def Transform(a):
    global outer_colors
    alpha = int(a[1])
    n = np.array(outer_colors[int(a[0])])
    #  n = np.array(outer_colors[3])
    n[..., -1] = alpha
    return n


FILE_PATH = (f'results/Ans FN-{f_index} X ({start}, {stop}, {n}x{m})' +
             f' C ({c}).npy')
with open(FILE_PATH, "r") as file:
    ans = np.load(FILE_PATH, allow_pickle=False)
#  print(ans[:,:])
num = np.max(np.max(ans, axis=0)[:, 0])
ans = np.apply_along_axis(Transform, -1, ans)


#ans[np.where(ans[:,:,3] == 100)] = [1,1,1,100]
rangeM = ans[np.where(ans[:,:,0] != 0)]
#rangeM = rangeM[np.where(rangeM[0,:] != 1)]
maxx = np.max(rangeM[..., -1])
minn = np.min(rangeM[..., -1])

ans[..., -1] /= maxx
ans[..., -1] += 0.5
ans[..., -1] /= 1.5
ans[np.where(ans[:,:,3] == ((100/maxx)+0.5)/1.5 )] = [1,1,1,0]
ans[np.where(ans[:,:,0] == 0)] = [0,0,0,0.8]


#  Plotting

plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 2
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1)
imgplot = plt.imshow(ans, origin='lower', extent=[start+root[0], stop+root[0], start+root[1], stop+root[1]])
ax.set_xlabel(r'$x_0$', labelpad=10)
ax.set_ylabel(r'$x_1$', labelpad=10)

artists = []
print("Length:",num)
for i in range(1, int(num)+1):
    change1 = outer_colors[i].copy()
    col1 = np.array([change1, ]*12)
    col1[..., -1] = np.linspace(0.33, 1, num=12)
    newcmp = ListedColormap(col1)
    norm = mpl.colors.Normalize(vmin=minn, vmax=maxx)
    axins = inset_axes(ax,
                       width="5%",  # width = 5% of parent_bbox width
                       height="100%",  # height : 50%
                       loc='lower left',
                       bbox_to_anchor=(1.05+(i*0.05), 0., 1, 1),
                       bbox_transform=ax.transAxes,
                       borderpad=0
                       )
    if i == int(num):
        cb = fig.colorbar(cm.ScalarMappable(cmap=newcmp, norm=norm),
                          cax=axins)
        cb.ax.set_ylabel('Iterations',)
    else:
        cb = fig.colorbar(cm.ScalarMappable(cmap=newcmp, norm=norm), ticks=[],
                          cax=axins)

    cb.ax.set_xlabel(r'$r_{{{}}}$'.format(i))
    artists.append(cb)

plt.savefig(f'graphics/FN-{f_index} X ({start}, {stop}, {n}x{m})' +
            f' C ({c}).png', bbox_inches='tight', pad_inches=0.1)
#plt.show()


