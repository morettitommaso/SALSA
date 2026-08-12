import numpy as np
from scipy.stats import norm

from ..base import AcquisitionFunction


class ExpectedImprovement(AcquisitionFunction):

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
        dataset=None

    ):

        mu, std = surrogate.predict(X_candidates, return_std=True)
        
        # evita errori numerici
        std = np.maximum(std, 1e-12)

        if self.objective == "maximize":
            f_star = np.max(dataset.y)
            z = (mu - f_star - self.xi)/std

            return (mu - f_star)*norm.cdf(z) + std*norm.pdf(z)

        else: 
            f_star = np.min(dataset.y)
            z = (f_star - mu - self.xi)/std

            return (f_star - mu)*norm.cdf(z) + std*norm.pdf(z)

