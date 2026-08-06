from abc import ABC, abstractmethod


class CandidateGenerator(ABC):

    @abstractmethod
    def generate(self, problem, dataset):
        pass