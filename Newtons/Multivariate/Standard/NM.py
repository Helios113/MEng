import numpy as np
delta = 1e-6


def fun(x):
    #  x_{1}^{3}-3x_1x_2^2-1\\
    #1return np.array([np.exp(x[0])-x[1], x[0]*x[1]-np.exp(x[0])])
    #return np.array([x[0]**2-x[1]**2-9, 2*x[0]*x[1]])
    return np.array([x[0]**3-3*x[0]*x[1]**2-1, 3*x[0]**2*x[1]-x[1]**3])


def grad_fun(x):
    #return np.array([np.exp(x[0]), -1, x[1]-np.exp(x[0]), x[0]]).reshape(2, 2)
    #return np.array([2*x[0], -2*x[1], 2*x[1] ,2*x[0]]).reshape(2, 2)
    return np.array([3*x[0]**2-3*x[1]**2, -6*x[0]*x[1], 6*x[0]*x[1], 3*x[0]**2-3*x[1]**2]).reshape(2, 2)


def Newton(x):
    x = np.array(x)
    cnt = 0
    for i in range(100):
        j = grad_fun(x)
        f = fun(x)
        if not is_invertible(j):
            return None, cnt
        b = np.linalg.inv(j).dot(f)
        if np.linalg.norm(b) < 1e-6:
            break
        x -= b
        cnt+=1
        
    if check_root(x):
        #  print("True:",x)
        return np.around(x, 3), cnt
    #  print("False:",x, steps)
    return None, cnt


def is_invertible(a):
    return a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0]


def check_root(x):
    #ans = np.array([1, 2.718])
    ans = 0
    if isinstance(ans, np.ndarray):
        if np.linalg.norm(ans-x.flatten()) <= 1e-3:
            return True
    else:
        ans += np.linalg.norm(fun(x))
        if ans <= 1e-3:
            return True
    return False

x = [1.0,2.0]
Newton(x)