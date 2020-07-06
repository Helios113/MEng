#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 13:51:38 2020

@author: preslavaleksandrov
"""
# imports
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.misc import derivative
import numpy as np
import matplotlib.patches as mpatches

# array of all solution
r_sol = []
x_sol = []

# Functiones used for calculations


def r(x):  # r function
    return np.exp(x)-500
    # return (x**3)-(2*x)+2#only integer powers for some reason


def r1(x):  # r' function (derivative)
    return derivative(r, x)  # gradient method for attaining derivatives


def dxN(x):  # delta x Newton's method
    return -(r(x)/r1(r, x))


def x_iter(x):  # iteration method Newton's method
    return x+dxN(x)


def x_EN_iter(x, c):  # iteration method Imporved Newton's
    return x+dxEN(x, c)


def rEN(x, c):  # modified r function for Imporved Newton's
    return P(x, c)*r(x)


def P(x, c):  # P function
    if c == x:  # zero division protection
        x = x+0.01
    return (x-c)/(r(x)-r(c))


def dxEN(x, c):  # delta x improved newton's
    if c == x:
        x = x+0.01
    return -((x-c)*r(x))/(r(x)-((x-c)*r1(x)*r(c)/(r(x)-r(c))))


def iterate(x_var, x_tolerance):
    # iteration function (Newton's implementation)
    while abs(r(x_var)) > x_tolerance:
        r_sol.append(r(x_var))
        x_sol.append(x_var)
        x_var = x_iter(x_var)
    # inclusion of the final point
    r_sol.append(r(x_var))
    x_sol.append(x_var)
    x_var = x_iter(x_var)
    print("Solution is "+str(len(x_sol))+" long")


def iterateEN(x_var, c_var, x_tolerance):
    # iteration function (Improved Newton's implementation)
    r_EN_sol = []
    x_EN_sol = []
    while not np.isclose(r(x_var), 0, x_tolerance, x_tolerance):
        if len(r_EN_sol) > 3000:
            return 0, 0
        r_EN_sol.append(P(x_var, c_var)*r(x_var))
        x_EN_sol.append(x_var)
        x_var = x_EN_iter(x_var, c_var)

    # inclusion of the final point
    r_EN_sol.append(r(x_var))
    x_EN_sol.append(x_var)
    x_var = x_EN_iter(x_var, c_var)
    return x_EN_sol, r_EN_sol

# Plotting interface


# fig, ax1 = plt.subplots(constrained_layout=True, dpi=300)
fig = plt.figure(dpi=300)
ax = fig.gca(projection='3d')
fig.tight_layout()
X = []
Y = []
Z = []
for i in range(0, 200):  # iteration through different c values
    # xn = random.randrange(-20,20,1) #random selection of an X init value
    xn = 4
    # solution vectors (steps taken to reach solution)
    x, y = iterateEN(xn, i, 1e-4)
    if x != 0 and y != 0:
        # generating Modified function (P*r)
        # plot for a sepecif c (over entire domain)
        x_exp = np.linspace(min(x), max(x), 10)
        X.extend(x_exp)
        Y.extend([i]*len(x_exp))
        r_exp = []
        for j in range(10):
            r_exp.append(P(x_exp[j], i)*r(x_exp[j]))
        Z.extend(r_exp)  # shape of function not exact steps taken
        """
        #2D Plot labeling
        ax1.set_xlabel(r'$x_n$')
        ax1.set_ylabel(r'$r(x_n)$')
        ax1.plot(x, y,label='c='+str(i)+" len="+str(len(x)),linewidth=0.2)
        ax1.plot(x_exp,r_exp,color = 'black',linewidth=0.2)
        """
    # Solution compilation
    # print(x[-1],"@ ",len(x)," iterations @ c=",i,"and initial guess x =", xn)
    else:
        print("No convergence for c=", i, "and initial guess x =", xn)


# 3D plotting


# X,Y = np.meshgrid(X,Y)
# print(Y)5
# Z = [Z]*len(X[0])
# generating original shape
X1 = np.linspace(min(X), max(X), 20)
Z1 = []
for j in range(len(X1)):
    Z1.append(r(X1[j]))
X1 = X1.tolist()
X1 = X1*20
Z1 = Z1*20
Y1 = np.linspace(min(Y), max(Y), 20)
Y1 = np.repeat(Y1, 20)
Z1 = np.asarray(Z1)
Z1 = (Z1/(max(Z1)-min(Z1)))*(max(Z)-min(Z))


col1 = "c"
col2 = "y"

ax.plot_trisurf(X, Y, Z,
                linewidth=0.2, antialiased=True, alpha=0.5, color=col1)
ax.plot_trisurf(X1, Y1, Z1,
                linewidth=0.2, antialiased=True, alpha=0.5, color=col2)
# ax.contour3D(X, Y, Z, 50, cmap='binary')
# ax.tricontourf(X, Y, Z,alpha=0.5)
ax.set_xlabel(r'$x_n$')
ax.set_ylabel(r'$c$')
ax.set_zlabel(r'$P(x_n,c)*r(x_n)$')
# ax.legend()
# plt.ion()


col1_patch = mpatches.Patch(color=col1, label='orginal function')
col2_patch = mpatches.Patch(color=col2, label='modified function')
plt.legend(handles=[col1_patch, col2_patch])
plt.show()
