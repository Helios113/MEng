import numpy as np
from decimal import *
import math

inputList = []
outputList = []
delta = 0.00000001
consts = [90, 90]
np.seterr(all='ignore')
path = "data"
# Data reader function
# takes data from location set @ variable and puts it into memory


def dataReader(location):
    with open(location, 'r') as reader:
        s = reader.read()
        a = ""
        first = True
        for i in s:
            if (i.isnumeric() or i == '.'):
                a += i
            if ((i == ' ' or i == '\n') and a != ""):
                if first:
                    inputList.append(float(a))
                    a = ""
                    first = False
                else:
                    outputList.append(float(a))
                    a = ""
                    first = True
    reader.close()


def dataWriter(location, A, B):
    with open(location, 'w+') as writer:
        writer.write("J\n")
        for i in A:
            writer.write(np.array_str(i)[1:-1] + '\n')
        writer.write("R\n")
        for i in B:
            writer.write(str(i) + '\n')
    writer.close()

# returns the square sum of a list


def sSquares(list):
    return sum([n**2 for n in list])

# returns the oputput of the specified function


def f(x, params):
    return params[0]*np.exp(params[1]*x)


def partial_derivative(x, index):
    n_consts = consts.copy()
    n_consts[index] = n_consts[index] - (delta/2)
    f1 = f(x, n_consts)
    n_consts[index] = n_consts[index] + delta
    f2 = f(x, n_consts)
    return (f2-f1)/delta


def Jacobian():
    J = []
    for i in inputList:
        j1 = []
        for j in range(len(consts)):
            j1.append(partial_derivative(i, j))
        J.append(j1)
    return np.array(J, dtype='d')


def Residual():
    R = []
    for i in np.arange(len(outputList)):
        R.append(outputList[i]-f(inputList[i], consts))
    return np.array(R, dtype='d')


dataReader('Gauss-Newton/'+path+'.raw')
lastConsts = [0, 0]
while not all(np.isclose(lastConsts, consts)):
    lastConsts = consts.copy()
    J = Jacobian()
    R = Residual()
    dataWriter('Gauss-Newton/'+path+'.out', J, R)
    c = np.matmul(np.matmul(np.linalg.inv(np.matmul(J.transpose(), J)), J.transpose()), R)
    consts = consts + c
    if math.isnan(consts[0]):
        break
    print(consts)
