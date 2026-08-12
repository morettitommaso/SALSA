import numpy as np
from scipy.stats import norm

from ..base import AcquisitionFunction


class ProbabilityOfImprovement(AcquisitionFunction):

    def __init__(
        self,
        objective,
        xi=0.01
    ):

        self.objective = objective
        self.xi = xi

    def compute(

        self,
        surrogate,
        X_candidates,
        dataset

    ):

        mu, std = surrogate.predict(X_candidates, return_std=True)
        
        # evita errori numerici
        std = np.maximum(std, 1e-12)
        
        if self.objective == "maximize":
            f_star = np.max(dataset.y)
            z = (mu - f_star - self.xi)/std
        else: 
            f_star = np.min(dataset.y)
            z = (f_star - mu - self.xi)/std

        return norm.cdf(z)