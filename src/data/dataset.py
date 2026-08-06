import numpy as np


class Dataset:

    def __init__(self, X, y):

        self.X = np.asarray(X)
        self.y = np.asarray(y)

    def add(self, X_new, y_new):

        X_new = np.atleast_2d(X_new)
        y_new = np.atleast_1d(y_new)

        self.X = np.vstack([self.X, X_new])
        self.y = np.concatenate([self.y, y_new])