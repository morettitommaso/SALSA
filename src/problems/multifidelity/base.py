from abc import abstractmethod

from src.problems.monofidelity.base import Problem


class MultiFidelityProblem(Problem):

    @abstractmethod
    def evaluate_low(self, X):
        pass

    @abstractmethod
    def evaluate_high(self, X):
        pass

    def evaluate(self, X):
        # di default la fidelity "vera" è la high
        return self.evaluate_high(X)