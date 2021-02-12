# kernels.py

import numpy as np

def gaussian_kernel(X, Y, sigma = 1):
    dist = np.sum((X[:,np.newaxis] - X[np.newaxis])**2, axis = 2)
    return np.exp(-0.5*dist/sigma**2)
