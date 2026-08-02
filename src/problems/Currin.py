import numpy as np

from .base import Problem


class Currin(Problem):

    def __init__(self):
        self.name = "Currin"
        self.dimension = 2
        self.bounds = np.array([[0, 1], [0, 1]])

    def evaluate_high(self, X):
        x1 = X[:, 0]
        x2 = X[:, 1]

        term1 = 1 - np.exp(-1 / (2 * x2))
        num = 2300 * x1**3 + 1900 * x1**2 + 2092 * x1 + 60
        den = 100 * x1**3 + 500 * x1**2 + 4 * x1 + 20
        term2 = num / den
        return term1 * term2

    def evaluate_low(self, X):
        x1 = X[:, 0]
        return 0.7 * self.evaluate_high(X) + 2 * np.sin(0.3 * x1)