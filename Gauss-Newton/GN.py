import numpy as np
import matplotlib.pyplot as plt

inputList = []
outputList = []


def dataReader(location):
    with open(location, 'r') as reader:
        print(reader.read())


def sSquares(list):
    return sum([n**2 for n in list])


def f(a, b, x):
    return a*np.exp(b*x)


dataReader('data.raw')
