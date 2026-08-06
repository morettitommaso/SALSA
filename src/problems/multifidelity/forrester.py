import numpy as np

from .base import MultiFidelityProblem


class Forrester(MultiFidelityProblem):

    def __init__(self):
        self.name = "Forrester"
        self.dimension = 1
        self.bounds = np.array([[0, 1]])

    def evaluate_high(self, X):
        x = X[:, 0]
        return (6 * x - 2) ** 2 * np.sin(12 * x - 4)

    def evaluate_low(self, X):
        x = X[:, 0]
        return 0.7 * self.evaluate_high(X) + 2 * np.sin(0.3 * x)