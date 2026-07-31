from abc import ABC, abstractmethod


class Surrogate(ABC):

    @abstractmethod
    def fit(self, dataset):
        """Fit the surrogate model."""

    @abstractmethod
    def predict(self, X, return_std=False):
        """Predict the high-fidelity response."""

    @abstractmethod
    def update(self, X_new, y_new):
        """Update the HF dataset."""

    @abstractmethod
    def posterior_variance_after(self, X_query, X_new):
        """Posterior variance after adding X_new without refitting."""