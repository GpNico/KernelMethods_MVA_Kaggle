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
        self.coef: np.array
        self.X: np.array
    
    def fit(self, X, y):
        n = X.shape[0]
        K = self.kernel(X, X)
        K_reg = K + self.alpha * n * np.eye(n)
        self.coef = scipy.linalg.solve(K_reg, y).flatten()
        self.X = X
        return

    def predict(self, X):
        return np.dot(self.kernel(X, self.X), self.coef)


class KernelRidgeClassification(KernelRidgeRegression):
    """
    Ridge Classification using a kernel
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def predict(self, X):
        return np.sign(super().predict(X))