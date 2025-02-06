import numpy as np

from scipy.linalg import norm
from sklearn.neighbors import BallTree, kneighbors_graph

from multiclass_scatterplot_features.graph import gamma_observable_neighbors
from multiclass_scatterplot_features.class_purity import class_proportion


def sepme(
    X,
    y,
    graph_fn=gamma_observable_neighbors,
    class_purity_fn=class_proportion,
    graph_fn_kwds={},
    class_purity_fn_kwds={},
):
    """
    SepMe measures implementation based on:
    - Aupetit and Sedlmair, "SepMe: 2002 New Visual Separation Measures", In Proc. PacificVis, 2016.
    With default parameters, this function outputs GONG 0.35 DIR CPT (the average proportion of same-class neighbors among the 0.35-Observable Neighbors of each point ofthe target class).

    Parameters:
    ----------
    X: {array-like, sparse matrix} of shape (n_samples, n_features)
        Training data, where n_samples is the number of samples and n_features is the number of features.
    y: labels (length: n_samples)
        Label of each sample.
    graph_fn: function (default: gamma_observable_neighbors)
        Function that generates a neighbor graph represented as a sparse matrix of shape (n_samples, n_samples).
        For example, functions prepared in graph.py (gamma_observable_neighbors, k_nearest_neighbors, epsilon_ball_neighbors)
    class_purity_fn: function (default: class_proportion)
        Function that computes a class purity score for a given neighbor graph.
        class_purity.py currently only provides class_proportion.
    graph_fn_kwds: dict (default: {})
        Keyword arguments used for graph_fn.
    class_purity_fn_kwds: dict (default: {})
        Keyword arguments used for class_purity_fn.
    Returns:
    ----------
    score: float
        SepMe score.
    """
    G = graph_fn(X, **graph_fn_kwds)
    score = class_purity_fn(G=G, y=y, **class_purity_fn_kwds)
    return score


def distance_consistency(X, y):
    """
    Distance consistency (DSC) implementaiton based on
    - Sips et al., "Selecting good views of high-dimensional data using class consistency", CGF 2009.

    Parameters:
    ----------
    X: {array-like, sparse matrix} of shape (n_samples, n_features)
        Training data, where n_samples is the number of samples and n_features is the number of features.
    y: labels (length: n_samples)
        Label of each sample.
    Returns:
    ----------
    score: float
        DSC score.
    """
    uniq_labels, y_int = np.unique(y, return_inverse=True)
    centroids = [X[y_int == label].mean(axis=0) for label in uniq_labels]

    # generate a matrix with row: instances, col: distance to each class centroid
    dists_to_centroids = np.array([np.sum((X - c) ** 2, axis=1) for c in centroids]).T

    # judge whether closest centroid is the belonging class or not
    closest_centroid_labels = np.argmin(dists_to_centroids, axis=1)
    score = np.sum((closest_centroid_labels - y_int) == 0) / X.shape[0]

    return score


def distribution_consistency(X, y, sigma=0.05, resolution=100, axis_limits=None):
    """
    Distribution_consistency (DC) implementaiton based on
    - Sips et al., "Selecting good views of high-dimensional data using class consistency", CGF 2009.

    Parameters:
    ----------
    X: {array-like, sparse matrix} of shape (n_samples, n_features)
        Training data, where n_samples is the number of samples and n_features is the number of features.
        Curently only supports cases with n_features=2.
    y: labels (length: n_samples)
        Label of each sample.
    sigma: float (default: 0.05)
        The kernel width parameter.
    resolution: int (default: 100)
        Resolution of grids used to create regions to compute entropy.
        For example, 100 makes 100x100 grids.
    axis_limits: array-like with shape (2, 2) or None (default: None)
        x and y-axes limits. The array row and col correspond to (x, y), (min, max), repectively.
        distribution_consistency function assumes X is in ranges of these limits.
        If None, automatically set as [[min of X[0, :], max of X[0, :]], [min of X[1, :], max of X[1, :]].
    Returns:
    ----------
    score: float
        DC score.
    """
    # NOTE: currently support only 2D data due to the use of np.meshgrid

    if axis_limits is None:
        # first row: xaxis_lim, second row: yaxis_lim, ...
        axis_limits = np.vstack((X.min(axis=0), X.max(axis=0))).T

    uniq_labels, y_int = np.unique(y, return_inverse=True)
    m = len(uniq_labels)

    # scale positions by limits. i.e., coordinates bounded in [0, 1]
    X_scaled = (X - axis_limits[:, 0]) / (axis_limits[:, 1] - axis_limits[:, 0])

    grid_size = 1 / resolution
    grid_coords_1d = np.arange(0.0, 1.0, grid_size) + grid_size / 2
    grid_xs, grid_ys = np.meshgrid(grid_coords_1d, grid_coords_1d)
    grid_centers = np.vstack((grid_xs.flatten(), grid_ys.flatten())).T

    # get epsilon-neighborhoods (i.e., neighbors within sigma radius)
    tree = BallTree(X_scaled)
    points_in_grid_areas = tree.query_radius(grid_centers, r=sigma)

    # TODO: make this faster by avoiding for-loop
    total_weighted_entropy = 0
    total_n_points_in_grids = 0
    for points in points_in_grid_areas:
        labels_in_grid = y[points]
        n_points_in_grid = len(labels_in_grid)
        if n_points_in_grid > 0:
            _, pcs = np.unique(labels_in_grid, return_counts=True)
            H = 0
            for pc in pcs:
                H -= (pc / n_points_in_grid) * np.log2(pc / n_points_in_grid)
            total_weighted_entropy += n_points_in_grid * H
            total_n_points_in_grids += n_points_in_grid

    score = 1 - total_weighted_entropy / (np.log2(m) * total_n_points_in_grids)

    return score


def density_aware_distance_consistency(X, y, summary_measure=True):
    """
    Density-aware distance concistency (density-aware DSC) implementation based on:
    - Wang et al., "A perception-driven approach to supervised dimensionality reduction for visualization", TVCG 2018.
    Note: This paper's DSC doesn't look precisely following the original DSC (i.e., something wrong in Eq 6).
    But, it does not influence on the implmentation of density-aware DSC (i.e., we follow Eq. 9).

    Parameters:
    ----------
    X: {array-like, sparse matrix} of shape (n_samples, n_features)
        Training data, where n_samples is the number of samples and n_features is the number of features.
        Curently only supports cases with n_features=2.
    y: labels (length: n_samples)
        Label of each sample.
    summary_measure: bool (default: True)
        If True, return the mean of all samples' density-aware DSC. Otherwise, return each sample's density-aware DSC
    Returns:
    ----------
    score: float or np.array
        The mean of all samples' density-aware DSC if summary_measure is True. Otherwise, each sample's density-aware DSC.
    """
    uniq_labels, y_int = np.unique(y, return_inverse=True)
    centroids = [X[y_int == label].mean(axis=0) for label in uniq_labels]

    # generate a matrix with row: instances, col: distance to each class centroid
    dists_to_centroids = np.array([np.sum((X - c) ** 2, axis=1) for c in centroids]).T
    a = dists_to_centroids[np.arange(len(y_int)), y_int]

    dists_to_centroids[np.arange(len(y_int)), y_int] = np.finfo(float).max
    b = np.min(dists_to_centroids, axis=1)
    s = (b - a) / np.vstack((a, b)).max(axis=0)
    if summary_measure:
        return s.mean()
    else:
        return s


def density_aware_knng(X, y, summary_measure=True):
    """
    Density-aware KNNG implementation based on:
    - Wang et al., "A perception-driven approach to supervised dimensionality reduction for visualization", TVCG 2018.

    Parameters:
    ----------
    X: {array-like, sparse matrix} of shape (n_samples, n_features)
        Training data, where n_samples is the number of samples and n_features is the number of features.
        Curently only supports cases with n_features=2.
    y: labels (length: n_samples)
        Label of each sample.
    summary_measure: bool (default: True)
        If True, return the mean of all samples' density-aware DSC. Otherwise, return each sample's density-aware DSC
    Returns:
    ----------
    score: float or np.array
        The mean of all samples' density-aware KNNG if summary_measure is True. Otherwise, each sample's density-aware KNNG.
    """
    n_neighbors = 2
    G = kneighbors_graph(
        X, n_neighbors=n_neighbors, mode="connectivity", include_self=False
    )

    # neighbor_labels: l(y_j) and l(y_k) in the paper
    neighbor_labels = np.reshape(y[G.nonzero()[1]], (X.shape[0], n_neighbors))
    same_label_judges = neighbor_labels == y[:, None]

    dists = np.reshape(
        norm(np.repeat(X, 2, axis=0) - X[G.nonzero()[1]], axis=1),
        (X.shape[0], n_neighbors),
    )

    a = dists[np.arange(X.shape[0]), same_label_judges.argmax(axis=1)]
    b = dists[np.arange(X.shape[0]), same_label_judges.argmin(axis=1)]
    max_ab = np.vstack((a, b)).max(axis=0)
    max_ab[max_ab == 0] = np.finfo(float).eps
    s = (b - a) / max_ab

    s[same_label_judges.prod(axis=1) == 1] = 1
    s[(~same_label_judges).prod(axis=1) == 1] = -1

    if summary_measure:
        return s.mean()
    else:
        return s
