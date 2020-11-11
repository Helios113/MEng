import ENM_mod as enm
import numpy as np
import tools
from datetime import datetime
import multiprocessing as mp
import subprocess
FILE_PATH = 'results/Ans '

def sendmessage(message):
    subprocess.Popen(['notify-send', message])
    return


if __name__ == '__main__':
    startTime = datetime.now()
    pool = mp.Pool(processes=8)
    n = 200j
    m = 200j
    start = -5
    stop = 5
    f_index = 5
    c1 = [0, 0]
    ans = np.zeros((int(n.imag), int(m.imag), 3))
    ansSet = {}
    x = np.mgrid[start:stop:n, start:stop:m].reshape(2, -1).T
    c = np.repeat(np.array(c1), -(n*m).real, axis=0).reshape(2, -1).T
    #workvec = np.array([2,5])
    #c = np.multiply(x, workvec)
    #c = 3*x
    #c = x+(1e-5)
    #c = -x
    #c = np.random.rand(int(-(n*m).real), 2)
    #list = [a for a in zip(x.tolist(), c.tolist())]
    for i, ii in enumerate(pool.imap(enm.solve, np.hstack((x, c)))):
        tools.printProgressBar(i+1, -(n*m).real, prefix="Progress", suffix="Complete",length = 50)
        if ii[0] is not None:
            if tuple(ii[0]) not in ansSet:
                ansSet[tuple(ii[0])] = np.array(
                        [len(ansSet)+1, 0, 0], dtype='int32')
                    #  print(ansSet[tuple(t[0])])
            work = ansSet[tuple(ii[0])]
            work[1] = ii[1]
        if ii[0] is None:
            work = [0,ii[1],0]
        ans[i % int(n.imag), i // int(n.imag), :] = work

    pool.close()
    pool.join()
    print("done")
    print(datetime.now()-startTime)
    name = (f"FN-{f_index} X ({start}, {stop}, {int(n.imag)}x{int(m.imag)})" +
            f" C ({c1})")
    with open(FILE_PATH + name+'.npy', "w+") as file:
        np.save(FILE_PATH + name+'.npy', ans, allow_pickle=False)
        np.save(FILE_PATH + name +'_ansSet'+'.npy', ansSet)
    print(ansSet)
    sendmessage("Task Finished")

