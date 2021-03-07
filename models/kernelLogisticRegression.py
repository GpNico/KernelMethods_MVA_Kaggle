# kernelLogisticRegression.py

import numpy as np
import scipy.linalg
import scipy.optimize

class KernelLogisticRegression:
    """
    Kernel logistic regression 
    """

    def __init__(self, kernel, alpha):
        self.kernel = kernel
        self.alpha = alpha
        self.coef = 0
        self.X = np.zeros((1,1))
        
    def _J(self, w, X, y):
        K = self.kernel(X, X)
        val = np.mean(np.log(1 + np.exp(- y * np.dot(K, w)))) + self.alpha/2 * np.dot(w,np.dot(K, w))
        return val
    
    def _gradJ(self, w, X, y):
        K = self.kernel(X, X)
        val = np.mean( -(y[:, None])*K*(np.exp(- y * np.dot(K, w))[:,None]) / ((1 + np.exp(- y * np.dot(K, w)))[:,None]), axis = 0 ) + self.alpha*np.dot(K,w)
        return val
        
    def fit(self, X, y):
        n, d = X.shape
        opt = scipy.optimize.fmin_l_bfgs_b(lambda w: self._J(w, X, y), x0 = np.zeros(n), fprime = lambda w: self._gradJ(w, X, y), factr = 10.)
        #opt = scipy.optimize.minimize(lambda w: self._J(w, X, y), x0 = np.zeros(d), jac =  lambda w: self._gradJ(w, X, y))
        self.coef = opt[0]
        self.X = X
        return

    def predict(self, X):
        return np.dot(self.kernel(X, self.X), self.coef)

class KernelLogisticClassification(KernelLogisticRegression):
    """
    Kernel logistic classification
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def predict(self, X):
        return np.sign(super().predict(X))