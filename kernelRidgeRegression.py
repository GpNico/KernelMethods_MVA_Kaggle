# kernelRidgeRegression.py

import numpy as np
import scipy.linalg

class KernelRidgeRegression:
    """
    Ridge regression using a kernel
    """

    def __init__(self, kernel, alpha):
        self.kernel = kernel
        self.alpha = alpha
        self.coef = 0
        self.X = np.zeros((1,1))
    
    def fit(self, X, y):
        n = X.shape[0]
        K = self.kernel(X, X)
        K_reg = K + self.alpha * n * np.eye(n)
        self.coef = scipy.linalg.solve(K_reg, y)
        self.X = X
        return

    def predict(self, X):
        return np.dot(self.kernel(X, self.X), self.coef)