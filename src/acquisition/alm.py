from .base import AcquisitionFunction


class almSampling(AcquisitionFunction):

    def compute(

        self,
        surrogate,
        X_candidates,
        dataset=None

    ):

        _, std = surrogate.predict(X_candidates, return_std=True)
        return std