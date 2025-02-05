import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.neighbors import kneighbors_graph, BallTree
from sklearn.metrics import pairwise_distances


def gamma_observable_neighbors(X, gamma=0.35, directed=True):
    """gamma-Observable Neighbor Graph implemented based on
    - M. Aupetit et al., "\gamma-observable neighbours for vector quantization". Neural Networks, 2002.
    Parameters:
    ----------
    X: {array-like, sparse matrix} of shape (n_samples, n_features)
        Training data, where n_samples is the number of samples and n_features is the number of features.
    gamma: float (default: 0.35)
        gamma parameter in a range of [0, 1].
    directed: bool (default: True)
        Whether symmetrize an output graph or not. If False, symmetrize a graph to make it undirected.
    Returns:
    ----------
    G: sparse matrix of shape (n_samples, n_samples)
        Graph where G[i, j] is assigned an edge that connects i to j, following the output format of sklearn.neighbors.kneighbors_graph
    """
    n = X.shape[0]
    d = X.shape[1]

    a = X * (1 - gamma)
    b = X * gamma
    # making a matrix contains nd points by repeating a and b
    repeated_a = np.broadcast_to(a, (n, n, d))
    repeated_b = np.broadcast_to(b, (n, n, d))
    # intermed_points[i, j] equals to a[i] + b[j]
    intermed_points = np.swapaxes(repeated_a, 0, 1) + repeated_b

    D = np.linalg.norm(intermed_points - X, axis=2)
    # fill diag with very large number to avoid select x_i
    np.fill_diagonal(D, np.finfo(np.float64).max)
    nearest_neighbors = np.argmin(D, axis=0)

    # make consitent with sklearn's kneighbors_graph
    neighbor_graph = csr_matrix(
        (np.ones(n), (np.arange(n), nearest_neighbors)), shape=(n, n), dtype=bool
    )

    if not directed:
        neighbor_graph = neighbor_graph + neighbor_graph.T

    return neighbor_graph


# gong gasalias for gamma_observable_neighbors
gong = gamma_observable_neighbors


def k_nearest_neighbors(X, k=2, directed=True):
    """k-Nearest Neighbor Graph
    Parameters
    ----------
    X: {array-like, sparse matrix} of shape (n_samples, n_features)
        Training data, where n_samples is the number of samples and n_features is the number of features.
    k: integer (default: 2)
        Number of neighbors
    directed: bool (default: True)
        Whether symmetrize an output graph or not. If False, symmetrize a graph to make it undirected.
    Returns
    ----------
    G: sparse matrix of shape (n_samples, n_samples)
        Graph where G[i, j] is assigned an edge that connects i to j, following the output format of sklearn.neighbors.kneighbors_graph
    """
    neighbor_graph = kneighbors_graph(X, k)
    if not directed:
        neighbor_graph = neighbor_graph + neighbor_graph.T
    return neighbor_graph


def epsilon_ball_neighbors(X, epsilon_ratio=0.1, directed=True):
    """epsilon-Ball Graph
    Parameters
    ----------
    X: {array-like, sparse matrix} of shape (n_samples, n_features)
        Training data, where n_samples is the number of samples and n_features is the number of features.
    epsilon_ratio: float (default: 0.1)
        Ratio of Epsilon / maximum distance of sample pairs. i.e., epsilon = esplion_ratio * maximum distance.
    directed: bool (default: True)
        Whether symmetrize an output graph or not. If False, symmetrize a graph to make it undirected.
    Returns
    ----------
    G: sparse matrix of shape (n_samples, n_samples)
        Graph where G[i, j] is assigned an edge that connects i to j, following the output format of sklearn.neighbors.kneighbors_graph
    """
    max_dist = pairwise_distances(X).max()
    epsilon = max_dist * epsilon_ratio
    tree = BallTree(X)
    nearest_neighbors = tree.query_radius(X, r=epsilon)

    # NOTE: maybe there is a faster way to construct csr_matrix
    neighbor_graph = lil_matrix((X.shape[0], X.shape[0]))
    for row, neighbor_cols in enumerate(nearest_neighbors):
        neighbor_graph[row, neighbor_cols] = 1
    neighbor_graph = neighbor_graph.tocsr()

    if not directed:
        neighbor_graph = neighbor_graph + neighbor_graph.T

    return neighbor_graph
