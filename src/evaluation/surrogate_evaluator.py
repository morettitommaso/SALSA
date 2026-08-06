import numpy as np
from scipy.stats import qmc


class SurrogateEvaluator:

    def __init__(
        self,
        problem,
        n_grid=80,
        n_test=5000,
        seed=42
    ):

        self.problem = problem
        self.n_grid = n_grid
        self.n_test = n_test
        self.seed = seed

    def generate_test_points(self):
        """
        Generate test points for evaluating the surrogate.

        - 2D problems: regular grid.
        - >2D problems: Latin Hypercube Sampling.
        """

        if self.problem.dimension == 2:

            axes = [
                np.linspace(low, high, self.n_grid)
                for low, high in self.problem.bounds
            ]

            mesh = np.meshgrid(*axes)

            X_test = np.vstack(
                [m.ravel() for m in mesh]
            ).T

        else:

            X_test = qmc.LatinHypercube(
                d=self.problem.dimension,
                seed=self.seed
            ).random(self.n_test)

            X_test = qmc.scale(
                X_test,
                self.problem.bounds[:, 0],
                self.problem.bounds[:, 1]
            )

        return X_test


    def evaluate(self, surrogate):

        X_test = self.generate_test_points()
        y_true = self.problem.evaluate(X_test)

        mean, std = surrogate.predict(
            X_test,
            return_std=True
        )

        mse = np.mean((y_true - mean) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_true - mean))

        return {
            "X_test": X_test,
            "y_true": y_true,
            "mean": mean,
            "std": std,
            "mse": mse,
            "rmse": rmse,
            "mae": mae
        }