# logisticRegression.py

import numpy as np
import scipy.linalg
import scipy.optimize

class LogisticRegression:
    """
    Classical logistic regression 
    """

    def __init__(self, alpha):
        self.alpha = alpha
        self.coef = 0
        self.X = np.zeros((1,1))
        
    def _J(self, w, X, y):
        val = np.mean(np.log(1 + np.exp(- y * np.dot(X, w)))) + self.alpha/2 * np.dot(w, w)
        return val
    
    def _gradJ(self, w, X, y):
        val = np.mean( -(y[:, None])*X*(np.exp(- y * np.dot(X, w))[:,None]) / ((1 + np.exp(- y * np.dot(X, w)))[:,None]), axis = 0 ) + self.alpha*w
        return val
        
    def fit(self, X, y):
        n, d = X.shape
        opt = scipy.optimize.fmin_l_bfgs_b(lambda w: self._J(w, X, y), x0 = np.zeros(d), fprime = lambda w: self._gradJ(w, X, y), factr = 10.)
        #opt = scipy.optimize.minimize(lambda w: self._J(w, X, y), x0 = np.zeros(d), jac =  lambda w: self._gradJ(w, X, y))
        self.coef = opt[0]
        self.X = X
        return

    def predict(self, X):
        return X @ self.coef
