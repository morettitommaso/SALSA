import numpy as np

class Experiment:

    def __init__(
        self,
        problem,
        dataset,
        surrogate,
        acquisition,
        evaluator,
        budget
    ):

        self.problem = problem
        self.dataset = dataset
        self.surrogate = surrogate
        self.acquisition = acquisition
        self.budget = budget
        self.evaluator = evaluator


    def run(self):

        self.history = []

        for i in range(self.budget):
            
            """

            controlli:
            - il budget deve essere minore del numero di LF points

            """

            # 1. punti LF non ancora HF
            candidate_points, candidate_idx = self.dataset.candidate_points

            # 2. fit surrogate
            self.surrogate.fit(self.dataset)


            # salvo stato corrente del modello
            X_test = self.evaluator.generate_test_points()

            _, std = self.surrogate.predict(
                X_test,
                return_std=True
            )

            self.history.append(
                {
                    "iteration": i,
                    "X_test": X_test.copy(),
                    "std": std.copy(),
                    "X_high": self.dataset.X_high.copy(),
                    "score": None
                }
            )

            # 3. acquisition
            score = self.acquisition.compute(
                self.surrogate,
                candidate_points,
                self.dataset
            )

            # 4. selezione
            selected = np.argmax(score)

            selected_point = candidate_points[selected]
            selected_idx = candidate_idx[selected]

            # 5. valutazione HF
            y_new = self.problem.evaluate_high(
                selected_point.reshape(1, -1)
            )

            # 6. aggiorna dataset
            self.dataset.add_high_fidelity(
                selected_point,
                y_new,
                idx=selected_idx
            )

        # fit finale
        self.surrogate.fit(self.dataset)

        return {
            "surrogate": self.surrogate,
            "dataset": self.dataset, # ritorna anche il dataset HF finale
            "history": self.history
        }