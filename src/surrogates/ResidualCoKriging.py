import numpy as np

from .base import Surrogate
from .gaussianprocess import GaussianProcess


class SimpleDataset:

    def __init__(self, X, y):
        self.X = X
        self.y = y

class ResidualCoKriging(Surrogate):

    def __init__(self):

        self.gp_low = GaussianProcess()
        self.gp_delta = GaussianProcess()
        self.dataset = None


    def fit(self, dataset):

        self.dataset = dataset

        # GP LOW
        low_dataset = SimpleDataset(
            dataset.X_low,
            dataset.y_low
        )

        self.gp_low.fit(
            low_dataset
        )

        # costruzione residuo

        residual = (
            dataset.y_high
            - self.gp_low.predict(dataset.X_high)

        )

        # GP DELTA
        delta_dataset = SimpleDataset(
            dataset.X_high,
            residual
        )

        self.gp_delta.fit(
            delta_dataset
        )


    def predict(self, X, return_std=False):

        if return_std:

            mean_low, std_low = self.gp_low.predict(
                X, return_std=True)

            mean_delta, std_delta = self.gp_delta.predict(
                X, return_std=True)

            mean = mean_low + mean_delta
            std = np.sqrt( std_low**2 + std_delta**2 )

            return mean, std

        else:

            mean_low = self.gp_low.predict(X)
            mean_delta = self.gp_delta.predict(X)
            return mean_low + mean_delta

    
    def update(self, X_new, y_new):

        self.dataset.X_high = np.vstack(
            [self.dataset.X_high, X_new]
        )

        self.dataset.y_high = np.concatenate(
            [self.dataset.y_high, y_new]
        )

        self.fit(self.dataset)
    
    
    def kernel(self, X1, X2=None):

        K_low = self.gp_low.model.kernel_(X1, X2)
        K_delta = self.gp_delta.model.kernel_(X1, X2)

        return K_low + K_delta
    
    
    def posterior_variance_after(
        self,
        X_query,
        X_new
    ):

        var_low = self.gp_low.posterior_variance_after(
            X_query,
            X_new
        )

        var_delta = self.gp_delta.posterior_variance_after(
            X_query,
            X_new
        )

        return var_low + var_delta