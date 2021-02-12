# kernelRidgeRegression.py

import numpy as np

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
        K_inv = np.linalg.inv(K + self.alpha * n * np.eye(n))
        self.coef = K_inv @ y
        self.X = X
        return

    def predict(self, X):
        return self.kernel(self.X, X) @ self.coef