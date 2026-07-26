from abc import ABC, abstractmethod

class Surrogate(ABC):

    @abstractmethod
    def fit(self, dataset):
        """Fit the surrogate model."""
        pass

    @abstractmethod
    def predict(self, X, return_std=False):
        """Predict the high-fidelity response."""
        pass

    @abstractmethod
    def update(self, X_new, y_new):
        """Update the HF dataset."""
        pass