import numpy as np

from .base import Problem


class Park(Problem):

    def __init__(self):
        self.name = "Park"
        self.dimension = 2
        self.bounds = np.array([[0, 1], [0, 1]])

    def evaluate_high(self, X):
        x1 = X[:, 0]
        x2 = X[:, 1]

        term1 = (x1 / 2) * (np.sqrt(1 + (x2 + x1**2) * x2**2) - 1)
        term2 = (x1 + 3 * x2) * np.exp(-np.sin(x1 - x2))
        return term1 + term2

    def evaluate_low(self, X):
        x1 = X[:, 0]
        return 0.7 * self.evaluate_high(X) + 2 * np.sin(0.3 * x1)