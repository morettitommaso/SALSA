from abc import ABC, abstractmethod
import numpy as np

class Problem(ABC):
    """
    Base class for all benchmark problems.
    """

    def __init__(self):

        self.name = None
        self.dimension = None
        self.bounds = None

    @abstractmethod
    def evaluate_high(self, X: np.ndarray):
        """
        Evaluate the high-fidelity function.
        """
        pass

    @abstractmethod
    def evaluate_low(self, X: np.ndarray):
        """
        Evaluate the low-fidelity function.
        """
        pass