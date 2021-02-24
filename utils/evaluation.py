# evaluation.py

import numpy as np

def accuracy(y_pred, y_test):
    return np.sum(y_test == y_pred)/len(y_test)