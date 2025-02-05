import numpy as np

from scipy.linalg import norm
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, breadth_first_tree
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

from alphashape import alphashape
import shapely

import time


class Scagnostics:
    """Scagnostics implemented by referring to
    - Wilkinson et al., "High-Dimensional Visual Analytics: Interactive Exploration Guided by Pairwise Views of Point Distributions". TVCG 2006.

    Note: There are issues in the details of the original paper:
    1. Although they stated "In our feature calculations, we ignore outliers (except for the outlier measure)",
    if we build MST with outliers, it influences most of measures (e.g., R implementaiton's alphashape and skew use q90 of the MST with outliers).
    Wilkinson et al. did not clarify whether we should rebuild MST after removing outliers.
    2. The paper's description of the Runt graph does not follow Struetzle's. Struetzle said "Break all MST edges that are as long or longer than e".
    3. R implementaiton of Clumpy provided by Wilkinson does not seem to follow the paper's equation (not 100% sure).

    1 influences most of scagnostics measures. This implementation first builds MST to detect outliers and then rebuilds MST after removing outliers.
    2 and 3 influence the implementation of Clumpy. But, this implementation follows the R implementation by Wilkinson.

    Parameters:
    ----------
    compute_clumpy: bool (default=True)
        If True, compute Clumpy measure. Clumpy measure requires more computations than other measures.
    compute_striate: bool (default=True)
        If True, compute Striate measure. Striate measure requires more computations than other measures.
    verbose: bool (default=False)
        If True, print out completion time for each measure computation.
    Attributes:
    ----------
    compute_clumpy: bool
        Reference to compute_clumpy parameter.
    compute_striate: bool
        Reference to compute_striate parameter.
    verbose: bool
        Reference to verbose parameter.
    outlying_: float or None
        Outlying measure. Before fitting, None is assigned.
    skew_: float or None
        Skew measure. Before fitting, None is assigned.
    sparse_: float or None
        Sparse measure. Before fitting, None is assigned.
    clumpy_: float or None
        Clumpy measure. Before fitting, None is assigned.
    striate_: float or None
        Striate measure. Before fitting, None is assigned.
    convex_: float or None
        Convex measure. Before fitting, None is assigned.
    skinny_: float or None
        Skinny measure. Before fitting, None is assigned.
    stringy_: float or None
        Stringy measure. Before fitting, None is assigned.
    monotonic_: float or None
        Monotonic measure. Before fitting, None is assigned.
    """

    def __init__(self, compute_clumpy=True, compute_striate=True, verbose=False):
        self.compute_clumpy = compute_clumpy
        self.compute_striate = compute_striate
        self.verbose = verbose

        self.outlying_ = None
        self.skew_ = None
        self.sparse_ = None
        self.clumpy_ = None
        self.striate_ = None
        self.convex_ = None
        self.skinny_ = None
        self.stringy_ = None
        self.monotonic_ = None

    def fit(self, X):
        """Fit the model with X.
        Parameters:
        ----------
        X: {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples and n_features is the number of features. Currently, n_features is assumed to be n_features=2. The other cases are not tested.
        Returns:
        ----------
        self: object
            Returns the instance itself.
        """
        if self.verbose:
            print("detecting outliers")

        start = time.time()
        # tree with outliers
        tree = minimum_spanning_tree(csr_matrix(squareform(pdist(X))))
        q90 = np.quantile(tree.data, 0.9)
        q75 = np.quantile(tree.data, 0.75)
        q50 = np.quantile(tree.data, 0.5)
        q25 = np.quantile(tree.data, 0.25)
        q10 = np.quantile(tree.data, 0.1)
        omega = q75 + 1.5 * (q75 - q25)

        # Category 1: Outlier
        # Wilkinson et al. TVCG 2006's outlying def
        self.outlying_ = tree[tree > omega].sum() / tree.sum()
        if self.verbose:
            print("outlying", time.time() - start)

        if self.verbose:
            print("deleting outliers")
        start = time.time()
        # things without outliers
        # remove outliers from X
        tree.data[tree.data > omega] = 0
        tree.eliminate_zeros()
        tree.data = np.ones_like(tree.data)
        self._X = X[np.array(tree.sum(axis=0) + tree.sum(axis=1).T).flatten() > 0]
        if self._X.shape[0] == 0:
            # basically all edges are judged as connections to outliers
            print("MST doesn't have any edges after removing outliers")
            return self
        else:
            self._tree = minimum_spanning_tree(csr_matrix(squareform(pdist(self._X))))
            self._convex_hull = shapely.MultiPoint(self._X).convex_hull

        # NOTE: this initial alpha condition can sometimes make an empty polygon
        alpha = 1 / max(q90, np.finfo(float).min)
        try:
            self._alpha_hull = alphashape(self._X, alpha=alpha)
            while self._alpha_hull.is_empty:
                alpha *= 0.95
                self._alpha_hull = alphashape(self._X, alpha=alpha)
        except Exception as e:
            print("alphashape failed and replaced with covex hull")
            self._alpha_hull = self._convex_hull

        # adjancy matrix (MST with all edge weights=1)
        A = self._tree.copy()
        A.data = np.ones_like(A.data)
        degrees = np.array(A.sum(axis=0) + A.sum(axis=1).T).flatten()
        if self.verbose:
            print("outlier deletion", time.time() - start)

        # Category 2: Density
        if self.verbose:
            print("computing density related measures")
        self.skew_ = (q90 - q50) / (q90 - q10)
        self.sparse_ = q90

        # clumpy
        # NOTE: following R implementaiton provided by Wilkinson
        start = time.time()
        if self.compute_clumpy:
            if self.verbose:
                print("computing clumpy")
            self.clumpy_ = self.clumpy(self._tree)
        if self.verbose:
            print("clumpy", time.time() - start)

        # striate
        start = time.time()
        if self.compute_striate:
            if self.verbose:
                print("computing striate")
            self.striate_ = self.striate(self._X, A, degrees)
        if self.verbose:
            print("striate", time.time() - start)

        # Category 3: Shape
        if self.verbose:
            print("computing shape related measures")
        self.convex_ = self._alpha_hull.area / max(
            self._convex_hull.area, np.finfo(float).eps
        )
        self.skinny_ = (
            1 - np.sqrt(4 * np.pi * self._alpha_hull.area) / self._alpha_hull.length
        )
        self.stringy_ = (degrees == 2).sum() / (
            (degrees > 0).sum() - (degrees == 1).sum()
        )

        # Category 4: Association
        if self.verbose:
            print("computing association related measures")
        start = time.time()
        self.monotonic_ = (spearmanr(*self._X.T).statistic) ** 2
        if self.verbose:
            print("monotonic", time.time() - start)

        return self

    def striate(self, X_without_outliers, A, degrees):
        v = X_without_outliers[degrees == 2]
        connected_vertices = (A.T + A)[degrees == 2].nonzero()[1]
        vec_a = X_without_outliers[connected_vertices[0::2]] - v
        vec_b = X_without_outliers[connected_vertices[1::2]] - v
        cosines = np.diag(vec_a @ vec_b.T) / norm(vec_a, axis=1) * norm(vec_b, axis=1)
        return (cosines < -0.75).sum() / (degrees > 0).sum()

    def clumpy(self, tree_without_outliers):
        max_val = 0.0
        for w, s, t in zip(
            tree_without_outliers.data, *tree_without_outliers.nonzero()
        ):
            filtered_tree = tree_without_outliers.copy()
            filtered_tree.data[filtered_tree.data >= w] = 0
            filtered_tree.eliminate_zeros()
            s_runt = breadth_first_tree(filtered_tree, s, directed=False)
            t_runt = breadth_first_tree(filtered_tree, t, directed=False)
            s_runt_size = 1 + (s_runt.sum(axis=0) > 0).sum()
            t_runt_size = 1 + (t_runt.sum(axis=0) > 0).sum()

            # R implementation multiply with runt size, which is not mentioned in the paper
            if s_runt_size < t_runt_size:
                val = s_runt_size * (1 - s_runt.max() / w)
            elif s_runt_size > t_runt_size:
                val = t_runt_size * (1 - t_runt.max() / w)
            else:
                val = s_runt_size * (1 - min(s_runt.max(), t_runt.max()) / w)
            max_val = max(max_val, val)

        return 2 * max_val / len(tree_without_outliers.data)
