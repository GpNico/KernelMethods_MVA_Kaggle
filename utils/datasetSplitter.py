# datasetSplitter.py

import numpy as np


class TrainTestSplitter:
    """
    Classical dataset splitting
    """
    def __init__(self, seed = 42, percent = 0.7):
        self.seed = seed
        self.percent = percent
        self.train_idx: np.array
        self.test_idx: np.array
    
    def generate_idx(self, X, percent = None):
        if percent == None:
            percent = self.percent

        np.random.seed(self.seed)
        nrow = X.shape[0]
        train_size = int(percent * nrow)

        shuffled_index = np.arange(nrow)
        np.random.shuffle(shuffled_index)

        self.train_idx = shuffled_index[:train_size]
        self.test_idx = shuffled_index[train_size:]
    
    def split(self, X):
        try:
            return X[self.train_idx], X[self.test_idx]
        except KeyError:
            return X.loc[self.train_idx], X.loc[self.test_idx]


class BalancedTrainTestSplitter(TrainTestSplitter):
    """
    Dataset splitting by respecting labels proportion
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def generate_idx(self, y, percent = None):
        if percent == None:
            percent = self.percent

        np.random.seed(self.seed)
        idx0, idx1 = np.where(y == 0)[0], np.where(y == 1)[0]

        np.random.shuffle(idx0)
        np.random.shuffle(idx1)

        train_size0, train_size1 = int(percent*len(idx0)), int(percent*len(idx1))

        self.train_idx = np.hstack([idx0[:train_size0], idx1[:train_size1]])
        self.test_idx = np.hstack([idx0[train_size0:], idx1[train_size1:]])