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
    
    def func(self, w):
        f = lambda w: self._J(w, self.X, self.y)
        g = lambda w: self._gradJ(w, self.X, self.y)
        return (f(w), g(w))
        
    def fit(self, X, y):
        n, d = X.shape
        self.X = X
        self.y = y
        print("Go")
        #opt = scipy.optimize.fmin_l_bfgs_b(lambda w: self._J(w, X, y), x0 = np.zeros(n), fprime = lambda w: self._gradJ(w, X, y), maxiter = 100, pgtol = 1e-4)
        #opt = scipy.optimize.minimize(lambda w: self._J(w, X, y), x0 = np.zeros(d), jac =  lambda w: self._gradJ(w, X, y), options = {'maxiter': 100})
        opt = scipy.optimize.minimize(self.func , x0 = np.zeros(n), method="L-BFGS-B", jac =  True, options = {'maxiter': 100, "gtol": 1e-4})
        print("Done")
        self.coef = opt.x
        
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