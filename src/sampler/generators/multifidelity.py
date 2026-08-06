import numpy as np

from .base import CandidateGenerator


class MultiFidelityCandidateGenerator(CandidateGenerator):

    def generate(self, problem, dataset):

        mask = np.ones(
            len(dataset.X_low),
            dtype=bool
        )

        mask[dataset.high_idx] = False


        return (
            dataset.X_low[mask],
            np.where(mask)[0]
        )