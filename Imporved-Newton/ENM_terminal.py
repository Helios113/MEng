
import ENM
import numpy as np
import tools
from datetime import datetime
import multiprocessing as mp
FILE_PATH = 'results/Ans '
if __name__ == '__main__':
    startTime = datetime.now()
    pool = mp.Pool(processes=8)

    n = 300j
    m = 300j
    start = -10
    stop = 10
    f_index = 1
    c1 = ["3x"]

    ans = np.zeros((int(n.imag), int(m.imag), 3))
    ansSet = {}
    x = np.mgrid[start:stop:n, start:stop:m].reshape(2, -1).T
    #c = np.repeat(np.array(c1), -(n*m).real, axis=0).reshape(2, -1).T
    #workvec = np.array([2,5])
    #c = np.multiply(x, workvec)
    #c = 3*x
    c = x+(1e-5)
    #c = np.random.rand(int(-(n*m).real), 2) *5
    list = [a for a in zip(x.tolist(), c.tolist())]

    for i, ii in enumerate(pool.imap(ENM.ENM, list),start=0):
        tools.printProgressBar(i,-(n*m).real, prefix="Progress",suffix="Complete",length = 50)
        if ii.roots is not None:
            if tuple(ii.roots.flatten().tolist()) not in ansSet:
                ansSet[tuple(ii.roots.flatten().tolist())] = np.array(
                        [len(ansSet)+1, 0, 0], dtype='int32')
                    #  print(ansSet[tuple(t.roots.flatten().tolist())])
            work = ansSet[tuple(ii.roots.flatten().tolist())]
            work[1] = len(ii.steps)
            ans[i % int(n.imag), i // int(n.imag), :] = work

    pool.close()
    pool.join()
    print("done")
    print(datetime.now()-startTime)
    name = (f"F-{f_index} X ({start}, {stop}, {int(n.imag)}x{int(m.imag)})" +
            f" C ({c1})")
    with open(FILE_PATH + name+'.npy', "w+") as file:
        np.save(FILE_PATH + name+'.npy', ans, allow_pickle=False)
        np.save(FILE_PATH + name +'_ansSet'+'.npy', ansSet)
    print(ansSet)

