import numpy as np
arg0 = np.random.normal(...)
arg1 = np.random.normal(...)

dis = np.einsum("", arg0, arg1)
