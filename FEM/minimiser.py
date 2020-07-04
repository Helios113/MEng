import numpy as np
from numpy.linalg import pinv
import FEM as fem


fem.main()
fem.force_stiff(x,1)