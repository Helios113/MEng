import numpy as np
c = 0
conDelta = 0.001414
delta = 1e-4


def fun(x):
    return 1/np.exp(x) - 10 #  problem


def f_mod(x):
    global c
    return fun(x)*((x-c)/(fun(x)-fun(c)))


def check_root(x):
    global conDelta
    ans = np.array([-2.303])
    #  ans = 0
    if isinstance(ans, np.ndarray):
        if np.linalg.norm(ans-x.flatten()) <= conDelta:
            return True
    else:
        ans += np.linalg.norm(fun(x))
        if ans <= conDelta:
            return True
    return False

def solve(x):
    global c
    global delta
    x0 = x[0]
    x1 = x[1]
    c = x[2]
    cnt = 0
    """Return the root calculated using the secant method."""
    for i in range(100):
        cnt+=1
        step = f_mod(x1) * (x1 - x0) / float(f_mod(x1) - f_mod(x0))
        x2 = x1 - step
        if step < delta:
            break
        x0, x1 = x1, x2

    if check_root(x2):
        return np.round(x2, 3).tolist(), cnt
    return None, cnt    
    return x2






root = solve([1,0,1.2])
print(root)

