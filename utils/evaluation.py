# evaluation.py

import numpy as np

def accuracy(y_pred, y_test):
    return np.sum(y_test == y_pred)/len(y_test)
	
def precision(y_pred, y_test):
    sum_pred = np.sum(y_pred == 1)
    if sum_pred == 0:
        return np.NaN
    return np.sum((y_test == 1)*(y_pred == 1))/sum_pred

def recall(y_pred, y_test):
    return np.sum((y_test == 1)*(y_pred == 1))/np.sum(y_test == 1)

def f1_score(y_pred, y_test):
    prec = precision(y_pred, y_test)
    rec = recall(y_pred, y_test)
    return 2*prec*rec/(prec + rec)