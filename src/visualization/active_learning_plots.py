import matplotlib.pyplot as plt
import numpy as np


class ActiveLearningPlotter:


    def __init__(
        self,
        problem,
        evaluator
    ):

        self.problem = problem
        self.evaluator = evaluator


    def plot_sampling(
        self,
        X_test,
        y_true,
        X_high,
        title="Sampling trajectory"
    ):
        """
        Plot HF sampling locations over the true function.
        Only available for 2D problems.
        """

        if self.problem.dimension != 2:
            raise ValueError(
                "Sampling plot available only for 2D problems"
            )


        x1 = X_test[:,0]
        x2 = X_test[:,1]

        n = int(np.sqrt(len(X_test)))

        X1 = x1.reshape(n,n)
        X2 = x2.reshape(n,n)
        Y = y_true.reshape(n,n)


        plt.figure(figsize=(7,6))

        plt.contourf(
            X1,
            X2,
            Y,
            levels=30
        )


        plt.scatter(
            X_high[:,0],
            X_high[:,1],
            c="red",
            s=40,
            label="HF samples"
        )


        plt.xlabel("x1")
        plt.ylabel("x2")

        plt.title(title)

        plt.legend()

        plt.show()



    def plot_uncertainty(
        self,
        X_test,
        std,
        X_high,
        title="Predictive uncertainty"
    ):
        """
        Plot GP uncertainty.
        """

        if self.problem.dimension != 2:
            raise ValueError(
                "Uncertainty plot available only for 2D problems"
            )


        n = int(np.sqrt(len(X_test)))


        X1 = X_test[:,0].reshape(n,n)
        X2 = X_test[:,1].reshape(n,n)

        STD = std.reshape(n,n)


        plt.figure(figsize=(7,6))


        plt.contourf(
            X1,
            X2,
            STD,
            levels=30
        )


        plt.scatter(
            X_high[:,0],
            X_high[:,1],
            c="red",
            s=40,
            label="HF samples"
        )


        plt.colorbar(
            label="Std"
        )


        plt.title(title)

        plt.legend()

        plt.show()



    def plot_uncertainty_evolution(
        self,
        history
    ):
        """
        Plot uncertainty reduction during active learning.

        history:
        [
            {
            "X_high": ...,
            "std": ...
            },
            ...
        ]
        """

        if self.problem.dimension != 2:
            raise ValueError(
                "Only available for 2D problems"
            )


        fig, axes = plt.subplots(
            1,
            len(history),
            figsize=(5*len(history),5)
        )


        vmax = max(
            h["std"].max()
            for h in history
        )

        vmin = min(
            h["std"].min()
            for h in history
        )


        for i,h in enumerate(history):

            ax = axes[i]


            n = int(np.sqrt(len(h["std"])))


            X_test = h["X_test"]


            X1 = X_test[:,0].reshape(n,n)
            X2 = X_test[:,1].reshape(n,n)

            STD = h["std"].reshape(n,n)


            c = ax.contourf(
                X1,
                X2,
                STD,
                levels=30,
                vmin=vmin,
                vmax=vmax
            )


            ax.scatter(
                h["X_high"][:,0],
                h["X_high"][:,1],
                c="red",
                s=20
            )


            ax.set_title(
                f"Iteration {i+1}"
            )


        fig.colorbar(
            c,
            ax=axes
        )

        plt.tight_layout()

        plt.show()



    def plot_learning_curve(
        self,
        mse_history
    ):
        """
        MSE reduction during active learning.
        """


        plt.figure(figsize=(7,5))


        plt.plot(
            range(len(mse_history)),
            mse_history,
            marker="o"
        )


        plt.xlabel(
            "HF evaluations"
        )

        plt.ylabel(
            "MSE"
        )

        plt.yscale(
            "log"
        )


        plt.title(
            "Active learning convergence"
        )


        plt.grid()

        plt.show()