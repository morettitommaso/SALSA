from abc import ABC, abstractmethod


class InitialDesign(ABC):

    @abstractmethod
    def generate(self, problem):
        """Return a Dataset object."""