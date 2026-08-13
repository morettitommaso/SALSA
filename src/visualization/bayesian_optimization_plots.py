import matplotlib.pyplot as plt
import numpy as np


class BayesianOptimizationPlotter:


    def __init__(self, problem, surrogate, history):

        self.problem = problem
        self.surrogate = surrogate
        self.history = history

    def generate_grid(self):
        
        if self.problem.dimension != 2:
            raise ValueError(
                "This plot is available only for 2D problems"
            )

        x1_min, x1_max = self.problem.bounds[0]
        x2_min, x2_max = self.problem.bounds[1]

        x1 = np.linspace(x1_min, x1_max, 100)
        x2 = np.linspace(x2_min, x2_max, 100)

        X1, X2 = np.meshgrid(x1, x2)

        X_grid = np.column_stack([
            X1.ravel(),
            X2.ravel()
        ])

        Y_grid = (self.problem.evaluate(X_grid).reshape(100, 100))
        return X1, X2, Y_grid



    def plot_surface(self, X_evaluated, Y_evaluated):

        X1, X2, Y_grid = self.generate_grid()

        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")

        surf = ax.plot_surface(
            X1,
            X2,
            Y_grid,
            cmap="viridis",
            alpha=0.7
        )

        ax.scatter(
            X_evaluated[:,0],
            X_evaluated[:,1],
            Y_evaluated,
            color="red",
            s=40
        )

        ax.set_title(f"{self.problem.name} Function")
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")
        ax.set_zlabel(r"$f(x)$")

        fig.colorbar(surf, ax=ax, shrink=0.6, aspect=15)

        plt.tight_layout()
        plt.show()


    def posterior_mean(self):

        X1, X2, _ = self.generate_grid()
        
        X_grid = np.column_stack([
            X1.ravel(),
            X2.ravel()
        ])

        mean = self.surrogate.predict(X_grid)
        mean = mean.reshape(X1.shape)

        fig = plt.figure(figsize=(7,6))
        ax = fig.add_subplot(111, projection="3d")
        
        ax.plot_surface(X1, X2, mean, cmap="viridis")
        
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_zlabel("Posterior mean")
        ax.set_title("GP posterior mean")
        ax.set_title("GP posterior mean")

        plt.show()

    def convergence_plot(self, best_values):
    
        plt.figure(figsize=(6,4))

        plt.plot(best_values, marker="o")

        plt.xlabel("BO iteration")
        plt.ylabel("Best observed value")
        plt.title("Bayesian Optimization convergence")

        plt.grid()
        plt.show()