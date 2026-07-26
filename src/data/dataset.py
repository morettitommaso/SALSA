import numpy as np

class MultiFidelityDataset:

    def __init__(

        self,

        X_low,
        y_low,

        X_high,
        y_high,

        high_idx

    ):

        self.X_low = X_low
        self.y_low = y_low

        self.X_high = X_high
        self.y_high = y_high

        self.high_idx = np.asarray(high_idx)

    @property
    def candidate_points(self):
        
        """
        ritorna anche l'indice del punto nel daatset completo così
        da non doverlo ricercare di nuovo successivamente
        """

        mask = np.ones(len(self.X_low), dtype=bool)
        mask[self.high_idx] = False

        return self.X_low[mask], np.where(mask)[0]


    def add_high_fidelity(self, X_new, y_new, idx):

        # assicuriamoci che abbiano forma (1, d) e (1,)
        X_new = np.atleast_2d(X_new)
        y_new = np.atleast_1d(y_new)

        # aggiungi il punto HF
        self.X_high = np.vstack([self.X_high, X_new])
        self.y_high = np.concatenate([self.y_high, y_new])

        self.high_idx = np.append(self.high_idx, idx)


