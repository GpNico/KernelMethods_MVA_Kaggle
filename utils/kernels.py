# kernels.py

import numpy as np

def linear_kernel(X, Y):
    """
    Return the Gram matrix K(X,Y) with K being the linear kernel
    """
    return np.sum(X[:,np.newaxis] * Y[np.newaxis], axis = 2)

def gaussian_kernel(X, Y, sigma = 1):
    """
    Return the Gram matrix K(X,Y) with K being the gaussian kernel
    """
    dist = np.sum((X[:,np.newaxis] - Y[np.newaxis])**2, axis = 2)
    return np.exp(-0.5*dist/sigma**2)
