import numpy as np

from .base import Problem


class Hartmann6D(Problem):

    def __init__(self):
        self.name = "Hartmann6D"
        self.dimension = 6
        self.bounds = np.array([[0, 1]] * 6)

    def evaluate_high(self, X):
        # X shape: (n_samples, 6)
        alpha = np.array([1.0, 1.2, 3.0, 3.2])
        A = np.array([
            [10, 3, 17, 3.5, 1.7, 8],
            [0.05, 10, 17, 0.1, 8, 14],
            [3, 3.5, 1.7, 10, 17, 8],
            [17, 8, 0.05, 10, 0.1, 14]
        ])
        P = 1e-4 * np.array([
            [1312, 1696, 5569, 124, 8283, 5886],
            [2329, 4135, 8307, 3736, 1004, 9991],
            [2348, 1451, 3522, 2883, 3047, 6650],
            [4047, 8828, 8732, 5743, 1091, 381]
        ])

        n = X.shape[0]
        result = np.zeros(n)
        for i in range(4):
            diff = X - P[i, :]          # (n, 6)
            weighted_sq = np.sum(A[i, :] * (diff ** 2), axis=1)  # (n,)
            result += alpha[i] * np.exp(-weighted_sq)
        return -result

    def evaluate_low(self, X):
        x0 = X[:, 0]
        return 0.7 * self.evaluate_high(X) + 2 * np.sin(0.3 * x0)