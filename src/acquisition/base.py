from abc import ABC, abstractmethod


class AcquisitionFunction(ABC):

    @abstractmethod
    def compute(

        self,
        surrogate,
        X_candidates,
        dataset

    ):

        pass