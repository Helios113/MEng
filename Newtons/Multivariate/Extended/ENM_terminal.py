import ENM as enm
import numpy as np
import tools
from datetime import datetime
import multiprocessing as mp
import subprocess
FILE_PATH = 'results/Ans '

def sendmessage(message):
    subprocess.Popen(['notify-send', message])
    return


def solve(x,c,c1,n,m,root,start,stop,f_index):
    print(__name__)
    if __name__ == 'ENM_terminal':
        startTime = datetime.now()
        pool = mp.Pool(processes=8)
        #n = 512j
        #m = 512j
        #root = [1.9619, 0.5525]
        #start = -50
        #stop = 50
        #f_index = 3
        #c1 = [2,1]
        ans = np.zeros((int(n.imag), int(m.imag), 3))
        ansSet = {}
       
        #c = np.repeat(np.array(c1), -(n*m).real, axis=0).reshape(2, -1).T
        #workvec = np.array([2,5])
        #c = np.multiply(x, workvec)
        #c = x**2
        #c = x-(1e-5)
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
    return 


f = [f1, f2]
c = np.array([2, 5]).reshape((2, 1))

n = 100
m = 100
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
print(datetime.now()-startTime)
name = (f"F-{f_index} X ({start}, {stop}, {n}x{m})" +
        f" C ({c.flatten().tolist()}).npy")
with open(FILE_PATH + name, "w+") as file:
    np.save(FILE_PATH + name, ans, allow_pickle=False)
