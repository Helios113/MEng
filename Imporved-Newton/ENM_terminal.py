
import ENM
import numpy as np
import tools
from datetime import datetime
import multiprocessing as mp





if __name__ == '__main__':
    startTime = datetime.now()
    FILE_PATH = 'results/Ans '
    pool = mp.Pool(processes=8)

    n = 10j
    m = 10j
    start = -20
    stop = 20
    f_index = 5

    #ans = np.zeros((m, n, 3))
    ansSet = {}
    x = np.mgrid[start:stop:n, start:stop:m].reshape(2, -1).T
    c = np.repeat(np.array([2, 5]), -(n*m).real, axis=0).reshape(2, -1).T
    list = [a for a in zip(x.tolist(), c.tolist())]
    t = pool.map(ENM.ENM, list)

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