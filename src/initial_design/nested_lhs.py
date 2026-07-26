from scipy.stats import qmc
from .kmedoids import KMedoids
from .base import InitialDesign

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

        X_L = qmc.LatinHypercube(d=problem.dimension, seed=self.seed, optimization="random-cd").random(self.n_low)
        X_L = qmc.scale(X_L, problem.bounds[:,0], problem.bounds[:,1])

        idx = KMedoids(X_L, n_clusters=self.n_high, random_state=self.seed)
        X_H = X_L[idx]
        
        return {
            "X_low": X_L,
            "X_high": X_H,
            "high_idx": idx
        }
