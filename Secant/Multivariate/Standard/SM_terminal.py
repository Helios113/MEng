import SM as enm
import numpy as np
import tools
from datetime import datetime
import multiprocessing as mp
FILE_PATH = 'results/Ans '
if __name__ == '__main__':
    startTime = datetime.now()
    pool = mp.Pool(processes=8)
    n = 200j
    m = 200j
    root = [1, 2.718]
    #Range
    start = -1
    stop = 1
    f_index = 5
    c1 = ["None"]


    ans = np.zeros((int(n.imag), int(m.imag), 3))
    ansSet = {}
    x = np.mgrid[start+root[0]:stop+root[0]:n, start+root[1]:stop+root[1]:m].reshape(2, -1).T
    x1 = np.random.rand(int(-(n*m).real), 2)


    for i, ii in enumerate(pool.imap(enm.solve, np.hstack((x, x1)))):
        tools.printProgressBar(i+1, (-(n*m).real), prefix="Progress", suffix="Complete",length = 50)
        if ii[0] is not None:
            if tuple(ii[0]) not in ansSet:
                ansSet[tuple(ii[0])] = np.array(
                        [len(ansSet)+1, 0, 0], dtype='int32')
                    #  print(ansSet[tuple(t[0])])
            work = ansSet[tuple(ii[0])]
            work[1] = ii[1]
            ans[i % int(n.imag), i // int(n.imag), :] = work

    pool.close()
    pool.join()
    print("done")
    print(datetime.now()-startTime)
    name = (f"F-{f_index} X ({start}, {stop}, {int(n.imag)}x{int(m.imag)})" +
            f" C ({c1}) SECANT")
    with open(FILE_PATH + name+'.npy', "w+") as file:
        np.save(FILE_PATH + name+'.npy', ans, allow_pickle=False)
        np.save(FILE_PATH + name +'_ansSet'+'.npy', ansSet)
    print(ansSet)

    