import numpy as np
import matplotlib.pyplot as plt

inputList = []
outputList = []

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

# returns the square sum of a list


def sSquares(list):
    return sum([n**2 for n in list])

# returns the oputput of the specified function


def f(a, b, x):
    return a*np.exp(b*x)


dataReader('Gauss-Newton/data.raw')
