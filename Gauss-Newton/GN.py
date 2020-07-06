import numpy as np
import math

inputList = []
outputList = []
delta = 0.00000000001  # finite step size
consts = [1, 90]  # initial guesses
np.seterr(all='ignore')
name = "data"  # data set name, imput files must have *.raw file type, outputs will be saved with *.out file type


def dataReader(location):
    # load data from name file
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


def dataWriter(location, t, A):
    # export data from name file
    # only shows last iteration of J and R matrix
    with open(location, 'w+') as writer:
        if t:
            writer.write("J\n")
            for i in A[0]:
                writer.write(np.array_str(i)[1:-1] + '\n')
            writer.write("R\n")
            for i in A[1]:
                writer.write(str(i) + '\n')
        else:
            writer.write("parameters: ")
            writer.write(np.array_str(A)[1:-1] + '\n')

    writer.close()


def sSquares(list):
    # calculate sum of square error
    return sum([n**2 for n in list])/2


def f(x, params):
    # Assumed model equation
    return params[0]*np.exp(params[1]*x)


def partial_derivative(x, index):
    # partial derivative using Finite differences method
    n_consts = consts.copy()
    n_consts[index] = n_consts[index] - (delta)
    f1 = f(x, n_consts)
    n_consts[index] = n_consts[index] + (delta)
    f2 = f(x, n_consts)
    n_consts[index] = n_consts[index] + (delta)
    f3 = f(x, n_consts)
    return (((f2-f1)/delta) + ((f3-f2)/delta))/2


def Jacobian():
    # assembley of Jacobian matrix
    J = []
    for i in inputList:
        j1 = []
        for j in range(len(consts)):
            j1.append(partial_derivative(i, j))
        J.append(j1)
    return np.array(J, dtype='d')


def Residual():
    # calculate the residual
    R = []
    for i in np.arange(len(outputList)):
        R.append(outputList[i]-f(inputList[i], consts))
    return np.array(R, dtype='d')


dataReader('Gauss-Newton/'+name+'.raw')
lastConsts = [0, 0]  # used for comparing convergance
fin = True
while not all(np.isclose(lastConsts, consts)):
    lastConsts = consts.copy()
    J = Jacobian()
    R = Residual()
    dataWriter('Gauss-Newton/'+name+'.out', True, [J, R])
    cccc = np.matmul(J.transpose(), J)  # J^t x J
    ccc = np.linalg.inv(cccc)  # (J^t x J) ^ -1
    cc = np.matmul(ccc, J.transpose())  # ((J^t x J) ^ -1) x J
    dc = np.matmul(cc, R)  # ((J^t x J) ^ -1) x J x R
    consts = consts + dc
    if math.isnan(consts[0]):
        print("failed")
        fin = False
        break
if fin:
    dataWriter('Gauss-Newton/'+name+'.out', False, consts)
    print(consts)
