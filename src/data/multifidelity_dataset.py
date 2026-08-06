import numpy as np

from .dataset import Dataset


class MultiFidelityDataset(Dataset):

    def __init__(
        self,
        X_low,
        y_low,
        X_high,
        y_high,
        high_idx
    ):

        super().__init__(X_high, y_high)

        self.X_low = np.asarray(X_low)
        self.y_low = np.asarray(y_low)

        self.high_idx = np.asarray(high_idx)


    @property
    def X_high(self):
        return self.X

    @property
    def y_high(self):
        return self.y

    def add_high_fidelity(self, X_new, y_new, idx):

        self.add(X_new, y_new)
        self.high_idx = np.append(self.high_idx, idx)
    