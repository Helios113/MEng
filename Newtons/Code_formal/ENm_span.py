import ENM_terminal as term
import numpy as np
n = 512j
m = 512j
root = [0, 0]
start = -50
stop = 50
f_index = 4
x = np.mgrid[start+root[0]:stop+root[0]:n,
             start+root[1]:stop+root[1]:m].reshape(2, -1).T


c1 = [["+1e-5"], ["-1e-5"], ["2x"], ["0.5x"],
      [1, 2], [-2, 1], [10, 20], ["3x,2x"], ["^2"]]
#c2 = np.repeat(np.array([1,2]), -(n*m).real, axis=0).reshape(2, -1).T
#c3 = np.repeat(np.array([2,1]), -(n*m).real, axis=0).reshape(2, -1).T
#c4 = np.repeat(np.array([10,20]), -(n*m).real, axis=0).reshape(2, -1).T
c5 = np.multiply(x, np.array([3, 2]))
c = [x+1e-5, x-1e-5, 2*x, 0.5*x, c2, c3, c4, c5, x**2]
#c5 = x+1
# for i in range(len(c1)):
#    term.solve(x,c[i],c1[i],n,m,root,start,stop,f_index)

term.solve(x, c5, c1[7], n, m, root, start, stop, f_index)
