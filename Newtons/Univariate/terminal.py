import numpy as np
import tester as tst
import tools
from math import sqrt, exp, sin, cos

def f(x):
    return x*(1-x)
def f1(x):
    return 1-(2*x)
def f2(x):
    return -2+0*x

fList = [f,f1,f2]


RANGE_X = [-5,6]
RANGE_C = [-5,6]
FUNCTION = "x*(1-x)"
dataPath = r'results'
fileName = "/info.db"
iter = 0

for i in range(*RANGE_X):
    for j in range(*RANGE_C):
        iter+=1
        if j == i:
            continue
        tools.printProgressBar(iter+1, (RANGE_X[1]-RANGE_X[0])*(RANGE_C[1]-RANGE_C[0]), prefix="Progress",suffix="Complete",length = 50)
        data = [tst.test(i,j,FUNCTION,fList)]
        tst.writeToDB(dataPath,fileName,data)