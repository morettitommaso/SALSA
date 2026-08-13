from abc import ABC, abstractmethod


class Problem(ABC):

    def __init__(self):
        self.name = None
        self.dimension = None
        self.bounds = None
        self.min = None
        self.max = None

    @abstractmethod
    def evaluate(self, X):
        pass