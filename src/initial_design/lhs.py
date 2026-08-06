from scipy.stats import qmc

from .base import InitialDesign


class LHS(InitialDesign):

    def __init__(
        self,
        n_points,
        seed=42
    ):

        self.n_points = n_points
        self.seed = seed

    def generate(self, problem):

        X = qmc.LatinHypercube(d=problem.dimension, seed=self.seed, optimization="random-cd").random(self.n_points)
        X = qmc.scale(X, problem.bounds[:,0], problem.bounds[:,1])

        return {"X": X}
