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

            history.append(
                {
                    "iteration": i,
                    "X": self.dataset.X.copy(),
                    "scores": scores.copy()
                }
            )

            idx = np.argmax(scores)

            x_new = candidates[idx]
            y_new = self.problem.evaluate(x_new.reshape(1, -1))

            self.dataset.add(x_new, y_new)
        
        # fit finale
        self.surrogate.fit(self.dataset)

        return {
            "surrogate": self.surrogate,
            "dataset": self.dataset,
            "history": history
        }