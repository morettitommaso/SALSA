from abc import ABC, abstractmethod

class InitialDesign(ABC):

    @abstractmethod
    def generate(self, problem):
        """
        Returns

        X_L
        y_L

        X_H
        y_H
        """
        pass