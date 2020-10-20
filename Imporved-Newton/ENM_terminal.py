
import ENM
import numpy as np
import tools
from datetime import datetime
startTime = datetime.now()

FILE_PATH = 'results/Ans '


def f1(x):
    return x[0]**3-3*x[0]*x[1]**2-1


def f2(x):
    return 3*x[0]**2*x[1]-x[1]**3


f = [f1, f2]
c = np.array([2, 5]).reshape((2, 1))

n = 10
m = 10
start = -20
stop = 20
f_index = 5

ans = np.zeros((m, n, 3))
ansSet = {}

for i, ii in enumerate(np.linspace(start, stop, n)):
    for j, jj in enumerate(np.linspace(start, stop, m)):
        #  print('this is i' ,i)
        #  print('this is j', j)
        tools.printProgressBar(i*n+j, n*m-1, prefix="Progress",
                               suffix="Complete", length=50)
        x = np.array([ii, jj], dtype='float64').reshape((2, 1))
        t = ENM.ENM(x, c, f)
        if t.roots is not None:
            if tuple(t.roots.flatten().tolist()) not in ansSet:
                #  print("yay")
                ansSet[tuple(t.roots.flatten().tolist())] = np.array(
                    [len(ansSet)+1, len(t.steps), 0], dtype='int32')
                #  print(ansSet[tuple(t.roots.flatten().tolist())])
            ans[j, i, :] = ansSet[tuple(t.roots.flatten().tolist())]
            #
name = (f"F-{f_index} X ({start}, {stop}, {n}x{m})" +
        f" C ({c.flatten().tolist()}).npy")
with open(FILE_PATH + name, "w+") as file:
    np.save(FILE_PATH + name, ans, allow_pickle=False)
