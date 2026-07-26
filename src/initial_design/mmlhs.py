import numpy as np


def MaximinLHS(
    n_samples,
    n_features,
    max_iter=10000,
    random_state=None
):
    """
    Maximin Latin Hypercube Sampling.

    Generates a Latin Hypercube Design and improves the
    minimum pairwise distance between points by performing
    column-wise swaps.

    Parameters
    ----------
    n_samples : int
        Number of design points.

    n_features : int
        Number of dimensions.

    max_iter : int
        Number of swap attempts.

    random_state : int or None
        Random seed.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        Optimized Latin Hypercube in [0,1]^d.
    """

    rng = np.random.default_rng(random_state)



    # Initial Latin Hypercube
    X = np.zeros(
        (n_samples, n_features)
    )

    for j in range(n_features):

        # random permutation of strata
        perm = rng.permutation(n_samples)

        # random position inside each stratum
        X[:, j] = (
            perm + rng.random(n_samples)
        ) / n_samples


    # Distance function
    def min_distance(X):

        D = np.linalg.norm(
            X[:, None, :] - X[None, :, :],
            axis=2
        )

        # ignore diagonal
        np.fill_diagonal(
            D,
            np.inf
        )

        return np.min(D)


    # Initial criterion
    best_distance = min_distance(X)

    # Optimization by swaps
    for _ in range(max_iter):

        # choose a dimension
        column = rng.integers(
            n_features
        )

        # choose two rows
        i, j = rng.choice(
            n_samples,
            size=2,
            replace=False
        )

        # proposed swap
        X_new = X.copy()

        X_new[i, column], X_new[j, column] = (
            X_new[j, column],
            X_new[i, column]
        )

        new_distance = min_distance(
            X_new
        )

        # accept only improvements
        if new_distance > best_distance:
            X = X_new
            best_distance = new_distance



    return X