import numpy as np
from scipy.stats import qmc

from ..base import AcquisitionFunction


class alcSampling(AcquisitionFunction):

    def __init__(
        self,
        problem,
        n_integration_points=5000,
        integration_points=None,
        seed=42
    ):

        self.problem = problem
        self.n_integration_points = n_integration_points
        self.integration_points = integration_points
        self.seed = seed


    def compute(self, surrogate, X_candidates, dataset=None):

        # evito di rigenerare LHS per ogni candidato
        if self.integration_points is None:
            self.integration_points = qmc.LatinHypercube(d=self.problem.dimension, seed=self.seed, optimization="random-cd").random(self.n_integration_points)
            self.integration_points = qmc.scale(self.integration_points, self.problem.bounds[:,0], self.problem.bounds[:,1])

        scores = []

        for candidate in X_candidates:
            
            var = surrogate.posterior_variance_after(
                self.integration_points,
                candidate.reshape(1,-1)
            )

            # inverte perchè il metodo cerca il max e non min
            scores.append(-np.sum(var))
        
        return scores