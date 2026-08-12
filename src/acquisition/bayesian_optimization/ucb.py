from ..base import AcquisitionFunction


class UpperConfidenceBound(AcquisitionFunction):

    def __init__(
        self,
        objective="maximize",
        w=0.5
    ):

        self.objective = objective
        self.w = w

    def compute(

        self,
        surrogate,
        X_candidates,
        dataset=None

    ):

        # experiments maximize the output of compute
        mu, std = surrogate.predict(X_candidates, return_std=True)

        if self.objective == "maximize":
            return mu + self.w * std
        else: 
            return -mu + self.w * std
