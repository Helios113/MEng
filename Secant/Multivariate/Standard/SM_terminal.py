import SM as enm
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
    root = [0.5,1]
    #Range
    start = -50
    stop = 50
    f_index = 5
    c1 = ["None"]


    ans = np.zeros((int(n.imag), int(m.imag), 3))
    ansSet = {}
    x = np.mgrid[start:stop:n, start:stop:m].reshape(2, -1).T
    x1 = np.mgrid[start+root[0]:stop+root[0]:n, start+root[1]:stop+root[1]:m].reshape(2, -1).T


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
    name = (f"FS-{f_index} X ({start}, {stop}, {int(n.imag)}x{int(m.imag)})" +
            f" C ({c1}) SECANT")
    with open(FILE_PATH + name+'.npy', "w+") as file:
        np.save(FILE_PATH + name+'.npy', ans, allow_pickle=False)
        np.save(FILE_PATH + name +'_ansSet'+'.npy', ansSet)
    print(ansSet)

    