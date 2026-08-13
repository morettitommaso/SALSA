import numpy as np

from .base import Problem


class Branin(Problem):

    def __init__(self):

        self.name = "Branin"
        self.dimension = 2
        self.bounds = np.array([
            [-5, 10],
            [0, 15]
        ])
        self.min = 0.39788735772973816
        self.max = 308.1290963265

    def evaluate(self, X):

        x1 = X[:, 0]
        x2 = X[:, 1]

        a = 1.0
        b = 5.1 / (4*np.pi**2)
        c = 5/np.pi
        r = 6
        s = 10
        t = 1/(8*np.pi)

        branin = a*(x2 - b*x1**2 + c*x1 - r)**2 + s*(1 - t)*np.cos(x1) + s
        bump = 2.0 * np.exp(-20*((x1 - 2.5)**2 + (x2 - 7.5)**2))

        return branin + bump

