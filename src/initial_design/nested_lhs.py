from scipy.stats import qmc

from src.data.multifidelity_dataset import MultiFidelityDataset

from .base import InitialDesign
from .greedymaxmin import GreedyMaximin


class NestedLHS(InitialDesign):

    def __init__(
        self,
        n_low,
        n_high,
        seed=42
    ):

        self.n_low = n_low
        self.n_high = n_high
        self.seed = seed

    def generate(self, problem):

        X_low = qmc.LatinHypercube(d=problem.dimension, seed=self.seed, optimization="random-cd").random(self.n_low)
        X_low = qmc.scale(X_low, problem.bounds[:,0], problem.bounds[:,1])

        idx = GreedyMaximin(X_low, self.n_high, random_state=self.seed)
        X_high = X_low[idx]

        return MultiFidelityDataset(
            X_low=X_low,
            y_low=problem.evaluate_low(X_low),

            X_high=X_high,
            y_high=problem.evaluate_high(X_high),

            high_idx=idx
        )
