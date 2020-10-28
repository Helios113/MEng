import numpy as np
import newtonM as nm
import tools
import multiprocessing as mp
from datetime import datetime
FILE_PATH = 'results/Ans '

if __name__ == '__main__':
    startTime = datetime.now()
    
    pool = mp.Pool(processes=8)
    c1 = ["None"]
    n = 300j
    m = 300j

    start = -10
    stop = 10
    f_index = 1
    x = np.mgrid[start:stop:n, start:stop:m].reshape(2, -1).T
    ans = np.zeros((int(n.imag), int(m.imag), 3))
    ansSet = {}
    #q = nm.Newton(x.tolist()[0])
    #print("Return answer:", q[0])
    
    for i, ii in enumerate(map(nm.Newton, x.tolist()), start=0):
        tools.printProgressBar(i, -(n*m).real, prefix="Progress", suffix="Complete", length = 50)
        if ii[0] is not None:
            if tuple(ii[0].flatten().tolist()) not in ansSet:
                ansSet[tuple(ii[0].flatten().tolist())] = np.array(
                        [len(ansSet)+1, len(ii[1]), 0], dtype='int32')
                #  print(ansSet[tuple(t.roots.flatten().tolist())])
            work = ansSet[tuple(ii[0].flatten().tolist())]
            work[1] = len(ii[1])
            ans[i % int(n.imag), i // int(n.imag), :] = work
   
    pool.close()
    pool.join()
    print("Done in:", datetime.now()-startTime)
    print("Ans set length:", len(ansSet))
    name = (f"F-{f_index} X ({start}, {stop}, {int(n.imag)}x{int(m.imag)})" +
            f" C ({c1})")
    with open(FILE_PATH + name+'.npy', "w+") as file:
        np.save(FILE_PATH + name +'.npy', ans, allow_pickle=False)
        np.save(FILE_PATH + name +'_ansSet'+'.npy', ansSet)
