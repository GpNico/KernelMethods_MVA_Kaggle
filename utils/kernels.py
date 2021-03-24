# kernels.py
# List of possible kernel functions

from collections import Counter
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.mismatchtree import preprocess, MismatchKernel

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

    kernel = np.einsum('ij, kj -> ik', X_kmers, Y_kmers)
    return kernel

def sum_kernel(X, Y, kernels = None):
    _sum = 0
    for kernel in kernels:
        _sum = _sum + globals()[kernel["class"]](X, Y, **kernel["parameters"])
    return _sum
    
    
def mismatch_kernel(X, Y, size = 6, m = 1, normalize = True):
    
    int_seq_X = np.array(preprocess(X['seq'].tolist())) # n_samples, size_seq
    int_seq_Y = np.array(preprocess(Y['seq'].tolist())) 
    
    n_samples_X, n_samples_Y = int_seq_X.shape[0], int_seq_Y.shape[0]
    
    concat_XY = np.concatenate((int_seq_X, int_seq_Y), axis = 0) # n_samples_X + n_samples_Y, size_seq
    
    mismatch_kernel = MismatchKernel(k = size, m = m).get_kernel(concat_XY, normalize = True).kernel # n_samples_X + n_samples_Y, n_samples_X + n_samples_Y
    
    return mismatch_kernel[:n_samples_X, n_samples_X:] 
