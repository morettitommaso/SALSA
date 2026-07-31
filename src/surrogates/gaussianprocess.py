import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import ConstantKernel as C


class GaussianProcess:

    def __init__(

        self,
        kernel=None,
        normalize_y=True,
        n_restarts_optimizer=3

    ):

        if kernel is None:
            kernel = C(1.0) * RBF(1.0)

        self.model = GaussianProcessRegressor(

            kernel=kernel,
            normalize_y=normalize_y,
            n_restarts_optimizer=n_restarts_optimizer

        )


    def fit(self, X, y):
        self.model.fit(X, y)


    def predict(self, X, return_std=False):

        return self.model.predict(
            X, return_std=return_std
        )


    def posterior_variance(self, X):

        """
        Compute posterior variance of the GP.
        Calcola sigma^2(x) = k(x,x) - k(x,X) K^{-1} k(X,x).
        Usando la Cholesky.
        """

        K_trans = self.model.kernel_(
            X,
            self.model.X_train_
        )

        K_inv_K_trans = np.linalg.solve(
            self.model.L_.T,
            np.linalg.solve(
                self.model.L_,
                K_trans.T
            )
        )

        K_xx = self.model.kernel_(X)

        var = np.diag(
            K_xx - K_trans @ K_inv_K_trans
        )

        return var


    def posterior_covariance(self, X1, X2):

        """
        Compute posterior covariance matrix.
        """

        K_1X = self.model.kernel_(
            X1,
            self.model.X_train_
        )

        K_X2 = self.model.kernel_(
            self.model.X_train_,
            X2
        )

        K_12 = self.model.kernel_(
            X1,
            X2
        )

        correction = np.linalg.solve(
            self.model.L_.T,
            np.linalg.solve(
                self.model.L_,
                K_X2
            )
        )

        return K_12 - K_1X @ correction


    def posterior_variance_after(self, X_query, X_new):

        # varianza attuale sui punti del dominio
        var_query = self.posterior_variance(
            X_query
        )

        # covarianza tra dominio e nuovo punto
        cov = self.posterior_covariance(
            X_query,
            X_new
        ).reshape(-1)

        # varianza del nuovo punto
        var_new = self.posterior_variance(
            X_new
        )[0]

        updated_var = (
            var_query
            -
            cov**2 / var_new
        )

        return updated_var

