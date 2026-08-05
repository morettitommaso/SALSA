import numpy as np


def GreedyMaximin(X, n_points, random_state=None):

    rng = np.random.default_rng(random_state)
    n = len(X)

    D = np.linalg.norm(
        X[:, None, :] - X[None, :, :],
        axis=2
    )

    selected = [rng.integers(n)]

    while len(selected) < n_points:
        remaining = np.setdiff1d(np.arange(n), selected)
        d = D[np.ix_(remaining, selected)]
        min_dist = d.min(axis=1)
        best = remaining[np.argmax(min_dist)]
        selected.append(best)

    return np.array(selected)