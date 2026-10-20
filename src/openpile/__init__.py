import numpy as np

# Use the numpy 1.x scalar repr, only relevant for numpy 2.x
if int(np.__version__.split(".")[0]) >= 2:
    np.set_printoptions(legacy="1.25")

__version__ = "1.1.0"
