
import ENM
import numpy as np
import tools
from datetime import datetime
import multiprocessing as mp
FILE_PATH = 'results/Ans '
if __name__ == '__main__':
    startTime = datetime.now()
    
    pool = mp.Pool(processes=8)

    n = 100j
    m = 100j

    start = -20
    stop = 20
    f_index = 5
    c1 = [2,5]

    ans = np.zeros((int(n.imag), int(m.imag), 3))
    ansSet = {}
    x = np.mgrid[start:stop:n, start:stop:m].reshape(2, -1).T
    c = np.repeat(np.array(c1), -(n*m).real, axis=0).reshape(2, -1).T
    list = [a for a in zip(x.tolist(), c.tolist())]
    #t = pool.imap(ENM.ENM, list)

    for i, ii in enumerate(pool.imap(ENM.ENM, list),start=0):
        tools.printProgressBar(i,-(n*m).real, prefix="Progress",suffix="Complete",length = 50)
        if ii.roots is not None:
            if tuple(ii.roots.flatten().tolist()) not in ansSet:
                ansSet[tuple(ii.roots.flatten().tolist())] = np.array(
                        [len(ansSet)+1, len(ii.steps), 0], dtype='int32')
                    #  print(ansSet[tuple(t.roots.flatten().tolist())])
            ans[i//int(n.imag), i % int(m.imag), :] = ansSet[tuple(ii.roots.flatten().tolist())]  

    pool.close()
    pool.join()
    print("done")
    print(datetime.now()-startTime)
    name = (f"F1-{f_index} X ({start}, {stop}, {int(n.imag)}x{int(m.imag)})" +
            f" C ({c1}).npy")
    with open(FILE_PATH + name, "w+") as file:
        np.save(FILE_PATH + name, ans, allow_pickle=False)
    print(len(ansSet))

    """if t.roots is not None:
                if tuple(t.roots.flatten().tolist()) not in ansSet:
                    #  print("yay")
                    ansSet[tuple(t.roots.flatten().tolist())] = np.array(
                        [len(ansSet)+1, len(t.steps), 0], dtype='int32')
                    #  print(ansSet[tuple(t.roots.flatten().tolist())])
                ans[j, i, :] = ansSet[tuple(t.roots.flatten().tolist())]            
                #
    
    print(datetime.now()-startTime)
    name = (f"F-{f_index} X ({start}, {stop}, {n}x{m})" +
            f" C ({c.flatten().tolist()}).npy")
    with open(FILE_PATH + name, "w+") as file:
        np.save(FILE_PATH + name, ans, allow_pickle=False)
"""