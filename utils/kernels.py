# kernels.py
# List of possible kernel functions

from collections import Counter
import numpy as np
import pandas as pd
from tqdm import tqdm

def linear_kernel(X, Y):
    """
    Return the Gram matrix K(X,Y) with K being the linear kernel
    """
    return np.sum(X[:,np.newaxis] * Y[np.newaxis], axis = 2)

def polynomial_kernel(X, Y, exponent = 2):
    """
    Return the Gram matrix K(X,Y) with K being the polynomial kernel
    """
    return np.sum((X[:,np.newaxis]*Y[np.newaxis] + 1)**exponent, axis = 2)

def gaussian_kernel(X, Y, gamma = 1):
    """
    Return the Gram matrix K(X,Y) with K being the gaussian kernel
    """
    dist = np.sum((X[:,np.newaxis] - Y[np.newaxis])**2, axis = 2)
    return np.exp(-gamma*dist)

def spectrum_kernel(X, Y, ids = "seq", size = 6):
    """
    Return the Gram matrix K(X,Y) with K being the spectrum kernel
    X and Y are expected to be pandas dataframe with as elements the full sequence.
    """
    def Kmers(seq):
        nonlocal size
        counter = Counter()
        for i in range(len(seq) - size + 1):
            counter[seq[i:i+size]] += 1
        return counter
    
    X_kmers = pd.DataFrame(X[ids].map(Kmers).tolist()).fillna(0)
    Y_kmers = pd.DataFrame(Y[ids].map(Kmers).tolist()).fillna(0)
    merged_columns = list(set().union(X_kmers.columns, Y_kmers.columns))

    X_kmers = X_kmers.reindex(columns = merged_columns, fill_value = 0)
    Y_kmers = Y_kmers.reindex(columns = merged_columns, fill_value = 0)

    kernel = np.dot(X_kmers, Y_kmers.T)
    return kernel

def sum_kernel(X, Y, kernels = None):
    _sum = 0
    for kernel in kernels:
        _sum = _sum + globals()[kernel["class"]](X, Y, **kernel["parameters"])
    return _sum