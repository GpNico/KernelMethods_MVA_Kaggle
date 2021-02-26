# test_grad.py
# Test the implementations of grad

import numpy as np

#add project directory to the path
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
pparentdir = os.path.dirname(parentdir)
sys.path.append(pparentdir)

import kernels
import logisticRegression
import kernelLogisticRegression

def test_check_grad_logistic_regression():

    #Parameters
    n = 100
    N = 50
    d = 2
    tol = 10**-4
    
    #Data
    X = np.random.randn(n,d)
    y = np.random.randn(n)
    
    #Functions
    clf = logisticRegression.LogisticRegression(alpha = 1.0)

    ws = np.random.randn(N, d)
    eps = (10**-8)*np.random.randn(N)

    bool_list = []

    for k in range(N):

        val1 = np.array([(clf._J(ws[k] + eps[k]*np.eye(d)[j], X, y) - clf._J(ws[k], X, y))/eps[k] for j in range(d)])
        val2 = clf._gradJ(ws[k], X, y)
    
        if np.linalg.norm(val1 - val2) < tol:
            bool_list.append(True)
        else:
            bool_list.append(False)
            
    assert all(bool_list) == True
    
def test_check_grad_kernel_logistic_regression():

    #Parameters
    N = 50
    n = 100
    d = 2
    tol = 10**-3
    
    #Data
    X = np.random.randn(n,d)
    y = np.random.randn(n)
    
    #Functions
    clf = kernelLogisticRegression.KernelLogisticRegression(kernel = kernels.gaussian_kernel, alpha = 1.0)

    ws = np.random.randn(N, n)
    eps = (10**-8)*np.random.randn(N)

    bool_list = []

    for k in range(N):

        val1 = np.array([(clf._J(ws[k] + eps[k]*np.eye(n)[j], X, y) - clf._J(ws[k], X, y))/eps[k] for j in range(n)])
        val2 = clf._gradJ(ws[k], X, y)
    
        if np.linalg.norm(val1 - val2) < tol:
            bool_list.append(True)
        else:
            bool_list.append(False)

    assert all(bool_list) == True




