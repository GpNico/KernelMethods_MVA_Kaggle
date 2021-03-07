# test_linear_regressions.py
# Test the implementations of the kernel regression with linear kernel

import numpy as np

#add project directory to the path
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
pparentdir = os.path.dirname(parentdir)
sys.path.append(pparentdir)

import kernels
import models.logisticRegression as logisticRegression
import models.kernelLogisticRegression as kernelLogisticRegression
import models.ridgeRegression as ridgeRegression
import models.kernelRidgeRegression as kernelRidgeRegression

def test_linear_ridge_regression():
    
    #Data
    n_samples, n_features = 2*300, 2
    X1 = np.random.multivariate_normal(np.array([1,-2]),np.eye(n_features),n_samples//2)
    X2 = np.random.multivariate_normal(2*np.ones(n_features),np.eye(n_features),n_samples//2)
    X = np.concatenate((X1,X2)).reshape(n_samples,n_features)
    y = np.array([1 for k in range(n_samples//2)] + [-1 for k in range(n_samples//2)])
    
    #Std Reg
    Myclf = ridgeRegression.RidgeRegression(alpha = 1.0)
    Myclf.fit(X,y)
    
    std_preds = np.sign(Myclf.predict(X))
    
    #Ker Reg
    Kerclf = kernelRidgeRegression.KernelRidgeRegression(kernel = kernels.linear_kernel, alpha = 1.0)
    Kerclf.fit(X,y)
    
    ker_preds = np.sign(Kerclf.predict(X))
    
    assert (ker_preds == std_preds).all()
    
def test_linear_logistic_regression():
    
    #Data
    n_samples, n_features = 2*300, 2
    X1 = np.random.multivariate_normal(np.array([1,-2]),np.eye(n_features),n_samples//2)
    X2 = np.random.multivariate_normal(2*np.ones(n_features),np.eye(n_features),n_samples//2)
    X = np.concatenate((X1,X2)).reshape(n_samples,n_features)
    y = np.array([1 for k in range(n_samples//2)] + [-1 for k in range(n_samples//2)])
    
    #Std Reg
    Myclf = logisticRegression.LogisticRegression(alpha = 1.0)
    Myclf.fit(X,y)
    
    std_preds = np.sign(Myclf.predict(X))
    
    #Ker Reg
    Kerclf = kernelLogisticRegression.KernelLogisticRegression(kernel = kernels.linear_kernel, alpha = 1.0)
    Kerclf.fit(X,y)
    
    ker_preds = np.sign(Kerclf.predict(X))
    
    assert (ker_preds == std_preds).all()
    






