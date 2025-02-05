import numpy as np


def class_proportion(G, y, target_labels=[1]):
    """Class Proportion Score implemented based on
    - M. Aupetit and M. Sedlmair, "SepMe: 2002 New Visual Separation Measures". Proc. PacificVis 2016.

    Parameters:
    ----------
    G: sparse matrix of shape (n_samples, n_samples)
        Graph where G[i, j] is assigned an edge that connects i to j, following the output format of sklearn.neighbors.kneighbors_graph
    y: labels (length: n_samples)
        Label of each sample
    target_labels: list (default: [1])
        Target labels used to comput class purity. The default, [1], corresponds to CPT in the SepMe paper. [0, 1] corresponds to CPA.
    Returns:
    ----------
    score: float
        Class purity score.
    """

    selected = np.zeros_like(y, dtype=bool)
    for target_label in target_labels:
        selected += y == target_label

    true_labels = y[selected]
    neighbor_labels = y[G[selected].tocoo().col]

    n_nbrs_by_node = np.array(G[selected].sum(axis=1)).flatten().astype(int)
    if np.all(n_nbrs_by_node == n_nbrs_by_node[0]):
        # all node has the same number of neighbors
        score = np.mean(
            neighbor_labels.reshape((n_nbrs_by_node[0], len(true_labels)))
            == true_labels
        )
    else:
        # each node has a different number of neighbors
        nbr_labels_by_node = np.split(neighbor_labels, np.cumsum(n_nbrs_by_node))[:-1]
        score = np.mean(
            [(t == ls).mean() for t, ls in zip(true_labels, nbr_labels_by_node)]
        )
    return score
