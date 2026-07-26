from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import RBF


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
            X,
            return_std=return_std
        )