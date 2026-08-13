import numpy as np


class Experiment:

    def __init__(
        self,
        problem,
        dataset,
        surrogate,
        acquisition,
        generator,
        evaluator,
        budget
    ):

        self.problem = problem
        self.dataset = dataset
        self.surrogate = surrogate
        self.acquisition = acquisition
        self.generator = generator
        self.evaluator = evaluator
        self.budget = budget


    def run(self):
        
        history = []
        for i in range(self.budget):
        
            self.surrogate.fit(self.dataset)

            candidates = self.generator.generate(self.problem)

            scores = self.acquisition.compute(
                self.surrogate,
                candidates,
                self.dataset
            )

            idx = np.argmax(scores)

            x_new = candidates[idx]
            y_new = self.problem.evaluate(x_new.reshape(1, -1))

            self.dataset.add(x_new, y_new)

            # Best observed so far (solo se c'è objective tra gli attributi, quindi solo per BO)
            if hasattr(self.acquisition, "objective"):
                if self.acquisition.objective == "minimize":
                    best_idx = np.argmin(self.dataset.y)
                else:
                    best_idx = np.argmax(self.dataset.y)
            else:
                best_idx = None

            history_entry = {
                "iteration": i,
                "X": self.dataset.X.copy(),
                "y": self.dataset.y.copy(),
                "scores": scores.copy()
            }

            # solo per BO aggiunge il best
            if best_idx is not None:
                history_entry["best_x"] = self.dataset.X[best_idx].copy()
                history_entry["best_y"] = self.dataset.y[best_idx].copy()

            history.append(history_entry)
        
        # fit finale
        self.surrogate.fit(self.dataset)

        return {
            "surrogate": self.surrogate,
            "dataset": self.dataset,
            "history": history
        }