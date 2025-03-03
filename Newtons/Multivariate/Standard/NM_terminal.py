import numpy as np
import NM as nm
import tools
import multiprocessing as mp
from datetime import datetime
import subprocess


FILE_PATH = 'results/Ans '


def sendmessage(message):
    subprocess.Popen(['notify-send', message])
    return


if __name__ == '__main__':
    startTime = datetime.now()
    
    pool = mp.Pool(processes=8)
    c1 = ["None"]
    n = 512j
    m = 512j
    root = [0, -0]
    start = -2
    stop = 2
    f_index = 6
    x = np.mgrid[start+root[0]:stop+root[0]:n, start+root[1]:stop+root[1]:m].reshape(2, -1).T
    ans = np.zeros((int(n.imag), int(m.imag), 3))
    ansSet = {}
    #q = nm.Newton(x.tolist()[0])
    #print("Return answer:", q[0])
    
    for i, ii in enumerate(pool.imap(nm.solve, x.tolist()), start=0):
        tools.printProgressBar(i, -(n*m).real, prefix="Progress", suffix="Complete", length = 50)
        if ii[0] is not None:
            if tuple(ii[0].flatten().tolist()) not in ansSet:
                ansSet[tuple(ii[0].flatten().tolist())] = np.array(
                        [len(ansSet)+1, ii[1], 0], dtype='int32')
                #  print(ansSet[tuple(t.roots.flatten().tolist())])
            work = ansSet[tuple(ii[0].flatten().tolist())]
            work[1] = ii[1]
            ans[i % int(n.imag), i // int(n.imag), :] = work
   
    pool.close()
    pool.join()
    print("Done in:", datetime.now()-startTime)
    print("Ans set length:", len(ansSet))
    name = (f"FN-{f_index} X ({start}, {stop}, {int(n.imag)}x{int(m.imag)})" +
            f" C ({c1})")
    with open(FILE_PATH + name+'.npy', "w+") as file:
        np.save(FILE_PATH + name +'.npy', ans, allow_pickle=False)
        np.save(FILE_PATH + name +'_ansSet'+'.npy', ansSet)
    sendmessage("Task Finished")