import numpy as np

from .experiment import Experiment


class MultiFidelityExperiment(Experiment):

    def run(self):
        
        history = []
        for i in range(self.budget):

            self.surrogate.fit(self.dataset)

            # punti LF non ancora HF
            candidates, idx = self.generator.generate(                
                self.problem,
                self.dataset
            )

            scores = self.acquisition.compute(
                self.surrogate,
                candidates,
                self.dataset
            )

            best = np.argmax(scores)

            x_new = candidates[best]
            idx_new = idx[best]

            y = self.problem.evaluate_high(x_new.reshape(1, -1))

            # aggiorna dataset
            self.dataset.add_high_fidelity(
                x_new,
                y,
                idx=idx_new
            )

        # fit finale
        self.surrogate.fit(self.dataset)

        return {
            "surrogate": self.surrogate,
            "dataset": self.dataset,
            "history": history
        }