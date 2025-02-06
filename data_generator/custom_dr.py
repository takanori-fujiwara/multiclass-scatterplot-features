import numpy as np

from sklearn.metrics import pairwise_distances
import sklearn.discriminant_analysis
import sklearn.decomposition
import sklearn.manifold
import sklearn.random_projection
import sklearn.neighbors

import umap
import phate
import cpca
import ccpca
import ulca.ulca


# custom LDA to handle when n_classes >= n_components
class LDA:
    def __init__(self, n_components=2, **kwargs):
        self.dr = sklearn.discriminant_analysis.LinearDiscriminantAnalysis(
            n_components=n_components, **kwargs
        )

    def fit_transform(self, X, y, **kwargs):
        if self.dr.n_components >= len(np.unique(y)):
            self.dr.n_components = len(np.unique(y)) - 1
        Z = self.dr.fit_transform(X, y, **kwargs)

        # even when n_components=2, some special case LDA generates 1D result
        if Z.shape[1] == 1:
            Z = np.hstack((Z, np.random.rand(X.shape[0], 1)))

        return Z


# custom cpca to handle when n_classes > 2
class CPCA:
    def __init__(self, n_components=2, **kwargs):
        self.dr = cpca.CPCA(n_components=n_components, **kwargs)

    def fit_transform(self, X, y, n_attrs_limit=1000, **kwargs):
        Z = None
        if X.shape[1] <= n_attrs_limit:
            uniq_labels, label_counts = np.unique(y, return_counts=True)
            fg = X[y == uniq_labels[np.argmax(label_counts)]]
            bg = X[y != uniq_labels[np.argmax(label_counts)]]
            self.dr.fit(fg, bg, **kwargs)
            Z = self.dr.transform(X)
        return Z


# custom ccpca to handle when n_classes > 2
class CCPCA:
    def __init__(self, n_components=2, **kwargs):
        self.dr = ccpca.CCPCA(n_components=n_components, **kwargs)

    def fit_transform(self, X, y, n_attrs_limit=1000, **kwargs):
        Z = None
        if X.shape[1] <= n_attrs_limit:
            uniq_labels, label_counts = np.unique(y, return_counts=True)
            fg = X[y == uniq_labels[np.argmax(label_counts)]]
            bg = X[y != uniq_labels[np.argmax(label_counts)]]
            self.dr.fit(fg, bg, **kwargs)
            Z = self.dr.transform(X)
        return Z


class RandomPramULCA:
    def __init__(self, n_components=2, **kwargs):
        self.dr = ulca.ulca.ULCA(n_components=n_components, **kwargs)

    def fit_transform(
        self, X, y=None, n_insts_limit=1000, n_attrs_limit=1000, **kwargs
    ):
        n_classes = len(np.unique(y))
        w_tg = np.random.rand(n_classes)
        w_bg = np.random.rand(n_classes)
        w_bw = np.random.rand(n_classes)
        alpha = np.random.rand() * 10

        Z = None
        if X.shape[0] <= n_insts_limit and X.shape[1] <= n_attrs_limit:
            Z = self.dr.fit_transform(
                X, y=y, w_tg=w_tg, w_bg=w_bg, w_bw=w_bw, alpha=alpha, **kwargs
            )

        return Z


# To ignore y param (e.g., UMAP uses y for supervised learning if y is set)
class Random:
    def __init__(self, n_components=2, **kwargs):
        self.dr = sklearn.random_projection.GaussianRandomProjection(
            n_components=n_components, **kwargs
        )

    def fit_transform(self, X, y=None, **kwargs):
        return self.dr.fit_transform(X, **kwargs)


class IPCA:
    def __init__(self, n_components=2, **kwargs):
        self.dr = sklearn.decomposition.IncrementalPCA(
            n_components=n_components, **kwargs
        )

    def fit_transform(self, X, y=None, **kwargs):
        return self.dr.fit_transform(X, **kwargs)


class TSNE:
    def __init__(self, n_components=2, **kwargs):
        self.dr = sklearn.manifold.TSNE(n_components=n_components, **kwargs)

    def fit_transform(self, X, y=None, **kwargs):
        return self.dr.fit_transform(X, **kwargs)


class UMAP:
    def __init__(self, n_components=2, **kwargs):
        self.dr = umap.UMAP(n_components=n_components, **kwargs)

    def fit_transform(self, X, y=None, **kwargs):
        return self.dr.fit_transform(X, **kwargs)


class PHATE:
    def __init__(self, n_components=2, verbose=0, **kwargs):
        self.dr = phate.PHATE(n_components=n_components, verbose=verbose, **kwargs)

    def fit_transform(self, X, y=None, **kwargs):
        return self.dr.fit_transform(X, **kwargs)


# Avoid precision error related to nonsymmetric distance mat
class MDS:
    def __init__(self, n_components=2, dissimilarity="precomputed", **kwargs):
        self.dr = sklearn.manifold.MDS(
            n_components=n_components, dissimilarity=dissimilarity, **kwargs
        )

    def fit_transform(self, X, y=None, n_insts_limit=1000, **kwargs):
        D = pairwise_distances(X, metric="euclidean")
        D = (D + D.T) / 2

        Z = None
        if X.shape[0] <= n_insts_limit:
            Z = self.dr.fit_transform(D, **kwargs)

        return Z
