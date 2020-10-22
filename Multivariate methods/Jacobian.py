import numpy as np
delta = 1e-6


def Jacobian(L, x):
    vs = None
    for i in range(len(L)):
        hs = None
        for j in range(len(x)):
            ans = getPartial(L[i], j, x)
            hs = np.hstack((hs, ans)) if hs is not None else ans
        vs = np.vstack((vs, hs)) if vs is not None else hs
    return vs


def getPartial(func, i, x):
    global delta
    a = x.copy()
    b = x.copy()

    a[i] -= delta
    b[i] += delta

    fa = func(a)
    fb = func(b)
    return (fb-fa)/(2*delta)
