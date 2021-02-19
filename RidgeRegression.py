# RidgeRegression.py

import numpy as np
import scipy.linalg

class RidgeRegression:
    """
    Classical ridge regression 
    """

    def __init__(self, alpha):
        self.alpha = alpha
        self.wRR = 0
        self.X = np.zeros((1,1))
    
    def fit(self, X, y):
        n, d = X.shape
        M = np.matmul(np.transpose(X), X) + self.alpha * n * np.eye(d)
        self.wRR = scipy.linalg.solve(M, np.matmul(np.transpose(X), y))
        self.X = X
        return

    def predict(self, X):
        return np.matmul(np.transpose(X), X) @ self.wRR