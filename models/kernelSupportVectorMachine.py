# kernelSupportVectorMachine.py

import numpy as np
import scipy.linalg
from cvxopt import matrix, solvers

class KernelSVM:
    """
        Kernel Support Vector Machine 
    """

    def __init__(self, kernel, alpha):
        self.alpha = alpha
        self.coef = 0
        self.X = np.zeros((1,1))
        self.kernel = kernel
    
    def fit(self, X, y):

        n, d = X.shape
        K = self.kernel(X, X)
        
        Q = matrix(K)
        p = matrix(y.astype('float'))
        
        G = matrix(np.concatenate((np.diag(y).astype('float'), np.diag(-y).astype('float')),axis=0))
        C = 1./(2*n*self.alpha)
        h = matrix(np.concatenate((C*np.ones(n), np.zeros(n)),axis =0))
        
        solvers.options['show_progress'] = False
        sol = solvers.qp(Q,p,G,h)
        
        self.coef = np.array(sol['x']).flatten()

        self.X = X
        
        return

    def predict(self, X):
        return np.dot(self.kernel(X, self.X), self.coef)
        
class KernelSVMClassification(KernelSVM):
    """
    Kernel SVM classification
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def predict(self, X):
        return np.sign(super().predict(X))
        