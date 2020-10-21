import numpy as np
import matplotlib.pyplot as plt


c = [2, 5]
n = 100
m = 100
start = -20
stop = 20
f_index = 5

colors = plt.get_cmap("tab20c")
outer_colors = colors(np.arange(4)*4)


def Transform(a):
    global outer_colors
    alpha = int(a[1])
    
    n = np.array(outer_colors[int(a[0])])
    #n = np.array(outer_colors[3])
    n[..., -1] = alpha
    return n


FILE_PATH = (f'results/Ans F-{f_index} X ({start}, {stop}, {n}x{m})' +
             f' C ({c}).npy')
with open(FILE_PATH, "r") as file:
    ans = np.load(FILE_PATH, allow_pickle=False)
#  print(ans[:,:])
ans = np.apply_along_axis(Transform, -1, ans)
ans[..., -1] /= np.max(ans[..., -1])
#  print(ans[:,:])
"""
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
imgplot = plt.imshow(ans)
ax.set_title('Multivariate Newton, c=(2,5), res = 25')
plt.show()
"""


plt.rcParams['font.size'] = 18
plt.rcParams['axes.linewidth'] = 2


fig = plt.figure(figsize=(3, 3))
ax = fig.add_subplot(1, 1, 1)
#  plt.xticks(np.arange(n), np.round(np.linspace(start,stop,n),2))
#  plt.yticks(np.arange(m), np.round(np.linspace(start,stop,m),2))
imgplot = plt.imshow(ans, origin='lower', extent=[start, stop, start, stop])
ax.set_xlabel(r'$x_0$', labelpad=10)
ax.set_ylabel(r'$x_1$', labelpad=10)
ax.legend(bbox_to_anchor=(1, 1), loc=1, frameon=False, fontsize=16)
plt.show()
