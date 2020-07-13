import numpy as np
globals = {'__builtins__': None}


def solve(expression, parameters):
    print(eval(expression, globals, parameters))
