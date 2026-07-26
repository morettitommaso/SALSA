import numpy as np


def KMedoids(X, n_clusters, max_iter=100, random_state=None):
    """
    Simple K-Medoids clustering.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Dataset.

    n_clusters : int
        Number of desired medoids.

    max_iter : int
        Maximum number of iterations.

    random_state : int or None
        Random seed.

    Returns
    -------
    medoid_idx : ndarray
        Indices of the selected medoids.
    """

    rng = np.random.default_rng(random_state)
    n_samples = len(X)

    # distance matrix
    D = np.linalg.norm(
        X[:, None, :] - X[None, :, :],
        axis=2
    )

    # Initialize medoids
    medoid_idx = rng.choice(n_samples, size=n_clusters, replace=False)

    # Main loop
    for _ in range(max_iter):

        old_medoid_idx = medoid_idx.copy()

        # Assignment step
        dist_to_medoids = D[:, medoid_idx]
        labels = np.argmin(dist_to_medoids, axis=1)

        # Update step
        for c in range(n_clusters):

            cluster_idx = np.where(labels == c)[0]

            if len(cluster_idx) == 0:
                continue

            D_cluster = D[np.ix_(cluster_idx, cluster_idx)]
            cost = D_cluster.sum(axis=1)
            best = np.argmin(cost)
            medoid_idx[c] = cluster_idx[best]

        # Check convergence
        if np.array_equal(old_medoid_idx, medoid_idx):
            break

    return medoid_idx