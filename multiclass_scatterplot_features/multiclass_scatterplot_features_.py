import numpy as np

from scipy.linalg import norm
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.metrics import pairwise_distances

from multiclass_scatterplot_features.scagnostics import Scagnostics
from multiclass_scatterplot_features.vis_class_separation import (
    sepme,
    distance_consistency,
    density_aware_distance_consistency,
    density_aware_knng,
)

from pymfe.complexity import MFEComplexity


class MulticlassScatterFeatures:
    """
    Multi-class Scatterplot Features implemented by following:
    - S. S. Bae, T. Fujiwara, C. Tseng, and D. A. Szafir. "Uncovering How Scatterplot Features Skew Visual Class Separation". Proc. CHI, 2025.
    For features correponding to classification complexity (Sec. 3.2), we rely on the PyMFE's complexity implementations (https://pymfe.readthedocs.io/en/latest/index.html).

    Parameters:
    ----------
    hist_measures_n_repeats: int (default: 10)
        How many repeats for drawing samples to compute histogram related measures.
    Attributes (after fit):
    ----------
    points_: numpy array with shape (n_classes, )
        Number of data points (N^points in the paper) for each class. Category: Scale (within-class)
    areas_: numpy array with shape (n_classes, )
        Alpha-hull area (Area^alpha-hull) for each class. Category: Scale (within-class)
    densities_: numpy array with shape (n_classes, )
        Desity of alpha-hull area (Density^alpha-hull) for each class. Category: Point Distance (within-class)
    skews_: numpy array with shape (n_classes, )
        Scagnostics' skewed score (Skew) for each class. Category: Point Distance (within-class)
    sparses_: numpy array with shape (n_classes, )
        Scagnostics' sparse score (Sparse) for each class. Category: Point Distance (within-class)
    clumpinesses_: numpy array with shape (n_classes, )
        Scagnostics' clumpiness score (Clumpiness) for each class. Category: Point Distance (within-class)
    outliers_: numpy array with shape (n_classes, )
        Scagnostics' outlier score (Outlier) for each class. Category: Point Distance (within-class)
    convexes_: numpy array with shape (n_classes, )
        Scagnostics' convex score (Convex) for each class. Category: Shape (within-class)
    skinnies_: numpy array with shape (n_classes, )
        Scagnostics' skinny score (Skinny) for each class. Category: Shape (within-class)
    stringies_: numpy array with shape (n_classes, )
        Scagnostics' stringy score (Stringy) for each class. Category: Shape (within-class)
    monotonics_: numpy array with shape (n_classes, )
        Scagnostics' monotonic score (Monotonic) for each class. Category: Shape (within-class)
    data_alpha_centroid_diffs_: numpy array with shape (n_classes, )
        Distance between the centroid of data points and the centroid of alpha-hull (CentroidDiff^alpha-hull) for each class. Category: Position (within-class)
    data_convex_centroid_diffs_: numpy array with shape (n_classes, )
        Distance between the centroid of data points and the centroid of covex-hull (CentroidDiff^convex) for each class. Category: Position (within-class)
    kurtosises_: numpy array with shape (n_classes, )
        Kurtosis (Kurtosis) for each class. Category: Position (within-class)
    distribution_overlaps_: numpy array with shape (n_classes, )
        Histogram intersection between real data points and points sampled from the corresponding normal distritbution (DistributionOverlap) for each class. Category: Position (within-class)
    distribution_distances_: numpy array with shape (n_classes, )
        Hellinger distance between real data points and points sampled from the corresponding normal distritbution (DistributionDistance) for each class. Category: Position (within-class)
    points_classes_ratio_: float
        Number of points divided by number of classes (Points/Classes). Category: Scale (between-class)
    std_point_: float
        Standard deviation of points_ (sigma^N^points). Category: Scale (between-class)
    std_area_: float
        Standard deviation of areas_ (sigma^Area^alpha-hull).Category: Scale (between-class)
    std_density_: float
        Standard deviation of densities_ (sigma^Density^alpha-hull). Category: Point Distance (between-class)
    std_skew_: float
        Standard deviation of skews_ (sigma^Skewed). Category: Point Distance (between-class)
    std_sparse_: float
        Standard deviation of sparses_ (sigma^Sparse). Category: Point Distance (between-class)
    equidistant_: float
        Equidistant measure (Equidistant). Category: Point Distance (between-class)
    split_: float
        Split measure (Split). Category: Point Distance (between-class)
    std_convex_: float
        Standard deviation of convexes_ (sigma^Convex). Category: Shape (between-class)
    std_skinny_: float
        Standard deviation of skinnies (sigma^Skinny). Category: Shape (between-class)
    std_stringy_: float
        Standard deviation of stringies_ (sigma^Stringy). Category: Shape (between-class)
    std_monotonic_: float
        Standard deviation of monotonics_ (sigma^Monotonic). Category: Shape (between-class)
    inner_occlusion_ratio_: float
        Inner Occlusion Ratio measure (InnerOcclusionRatio). Category: Position (between-class)
    inner_occlusion_ratios_: numpy array with shape (n_classes, )
        InnerOcclusionRatio_a. inner_occlusion_ratio_ is the maximum of inner_occlusion_ratios_.
    convex_overlap_: float
        Overlap convex-hull area among classes (Overlap^convex). Category: Position (between-class)
    alpha_overlap_: float
        Overlap alpha-hull area among classes (Overlap^alpha-hull). Category: Position (between-class)
    gong_035_dir_cp_t0_: float
        GONG 0.35 DIR CPT using Class 0 as a target class (GONG 0.35 DIR CPT)_t0). Category: Position (between-class)
    gong_035_dir_cp_t1_: float
        GONG 0.35 DIR CPT using Class 1 as a target class (GONG 0.35 DIR CPT)_t1). Category: Position (between-class)
    distance_consistency_: float
        Distance consistency measure (DSC). Category: Position (between-class)
    density_aware_distance_consistency_: float
        Density-aware distance consistency measure (density-awareDSC). Category: Position (between-class)
    density_aware_knng_: float
        Density-aware KNNG (density-awareKNNG). Category: Position (between-class)
    c_f1v_: float
        Classification complexity measure F1V (C^F1V). Category: Axis Feature
    c_f2_: float
        Classification complexity measure F2 (C^F2). Category: Axis Feature
    c_f3_: float
        Classification complexity measure F3 (C^F3). Category: Axis Feature
    c_f4_: float
        Classification complexity measure F4 (C^F4). Category: Axis Feature
    c_l1_: float
        Classification complexity measure L1 (C^L1). Category: Linearity Feature
    c_l2_: float
        Classification complexity measure L2 (C^L2). Category: Linearity Feature
    c_l3_: float
        Classification complexity measure L3 (C^L3). Category: Linearity Feature
    c_n1_: float
        Classification complexity measure N1 (C^N1). Category: Neighborhood Feature
    c_n2_: float
        Classification complexity measure N2 (C^N2). Category: Neighborhood Feature
    c_n3_: float
        Classification complexity measure N3 (C^N3). Category: Neighborhood Feature
    c_n4_: float
        Classification complexity measure N4 (C^N4). Category: Neighborhood Feature
    c_t1_: float
        Classification complexity measure T1 (C^T1). Category: Neighborhood Feature
    c_lsc_: float
        Classification complexity measure LSC (C^LSC). Category: Neighborhood Feature
    c_c1_: float
        Classification complexity measure C1 (C^C1). Category: Class Imbalance Feature
    c_c2_: float
        Classification complexity measure C2 (C^C2). Category: Class Imbalance Feature
    c_cls_coef_: float
        Classification complexity measure ClsCoef (C^ClsCoef). Category: Network-based Feature
    c_density_: float
        Classification complexity measure Density (C^Density). Category: Network-based Feature
    c_hub_: float
        Mean of classification complexity measure Hubs (C^Hubs). Category: Network-based Feature
    min_point_: float
        Minimum of points_ (N^points_min)
    min_area_: float
        Minimum of areas_ (Area^alpha-hull_min)
    min_density_: float
        Minimum of densities_ (Density^alpha-hull_min)
    min_clumpiness_: float
        Minimum of clumpinesses_ (Clumpiness_min)
    min_skew_: float
        Minimum of skews_ (Skeq_min)
    min_sparse_: float
        Minimum of sparses (Sparse_min)
    min_outlier_: float
        Minimum of outliers_ (Outlier_min)
    min_convex_: float
        Minimum of convexes_ (Convex_min)
    min_skinny_: float
        Minimum of skinnies (Skinny_min)
    min_stringy_: float
        Minimum of stringies_ (Stringy_min)
    min_monotonic_: float
        Minimum of monotonics (Monotonic_min)
    min_data_alpha_centroid_diff_: float
        Minimum of data_alpha_centroid_diffs_ (ConvexDiff^alpha-hull_min)
    min_data_convex_centroid_diff_: float
        Minimum of data_convex_centroid_diffs_ (ConvexDiff^convex_min)
    min_kurtosis_: float
        Minimum of kurtosises_ (Kurtosis_min)
    min_distribution_overlap_: float
        Minimum of distribution_overlaps_ (DistributionOverlap_min)
    min_distribution_distance_: float
        Minimum of distribution_distances_ (DistributionDistance_min)
    max_point_: float
        Maximum of points_ (N^points_max)
    max_area_: float
        Maximum of areas_ (Area^alpha-hull_max)
    max_density_: float
        Maximum of densities_ (Density^alpha-hull_max)
    max_clumpiness_: float
        Maximum of clumpinesses_ (Clumpiness_max)
    max_skew_: float
        Maximum of skews_ (Skeq_max)
    max_sparse_: float
        Maximum of sparses (Sparse_max)
    max_outlier_: float
        Maximum of outliers_ (Outlier_max)
    max_convex_: float
        Maximum of convexes_ (Convex_max)
    max_skinny_: float
        Maximum of skinnies (Skinny_max)
    max_stringy_: float
        Maximum of stringies_ (Stringy_max)
    max_monotonic_: float
        Maximum of monotonics (Monotonic_max)
    max_data_alpha_centroid_diff_: float
        Maximum of data_alpha_centroid_diffs_ (ConvexDiff^alpha-hull_max)
    max_data_convex_centroid_diff_: float
        Maximum of data_convex_centroid_diffs_ (ConvexDiff^convex_max)
    max_kurtosis_: float
        Maximum of kurtosises_ (Kurtosis_max)
    max_distribution_overlap_: float
        Maximum of distribution_overlaps_ (DistributionOverlap_max)
    max_distribution_distance_: float
        Maximum of distribution_distances_ (DistributionDistance_max)
    """

    def __init__(self, hist_measures_n_repeats=10):
        self._hist_n_repeats = hist_measures_n_repeats

    def fit(self, X, y, axis_limits=None):
        """
        Fit the model with X and y.
        Parameters:
        ----------
        X: {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples and n_features is the number of features. Currently, n_features is assumed to be n_features=2. The other cases are not tested.
        y: array-like of shape (n_samples, )
            Class labels.
        axis_limits: array-like with shape (2, 2) or None (default: None)
            x and y-axes limits. The array row and col correspond to (x, y), (min, max), repectively.
            distribution_consistency function assumes X is in ranges of these limits.
            If None, [[min of X[0, :], max of X[0, :]], [min of X[1, :], max of X[1, :]] is used.
        Returns:
        ----------
        self: object
            Returns the instance itself.
        """
        self._uniq_labels, self.points_ = np.unique(y, return_counts=True)
        if self.points_.min() <= 3:
            print("too small points in a class")
        else:
            self._fit_within_class_factors(X, y, axis_limits=axis_limits)
            if len(np.unique(y)) > 1:
                self._fit_between_class_factors(X, y, axis_limits=axis_limits)
                self._fit_classif_complexity_measures(X, y, axis_limits=axis_limits)
        return self

    def _fit_within_class_factors(self, X, y, axis_limits=None):
        if axis_limits is None:
            # first row: xaxis_lim, second row: yaxis_lim, ...
            axis_limits = np.vstack((X.min(axis=0), X.max(axis=0))).T
        axis_range = axis_limits[:, 1] - axis_limits[:, 0]
        # handle zeros in scale
        axis_range[axis_range == 0] = 1

        # scale positions by limits. i.e., coordinates bounded in [0, 1]
        _X = (X - axis_limits[:, 0]) / axis_range

        scags = [Scagnostics().fit(_X[y == label]) for label in self._uniq_labels]
        # save hulls for _fit_between_class_factors
        self._alpha_hulls = [s._alpha_hull for s in scags]
        self._convex_hulls = [s._convex_hull for s in scags]

        # 1. scale related
        # self.points_  # computed alredy when fit called
        self.areas_ = np.array([s._alpha_hull.area for s in scags])
        self.areas_[self.areas_ == 0] = 1e-30

        # 2. point distance related
        self.densities_ = self.points_ / self.areas_
        self.clumpinesses_ = np.array([s.clumpy_ for s in scags])
        self.skews_ = np.array([s.skew_ for s in scags])
        self.sparses_ = np.array([s.sparse_ for s in scags])
        # self.striates_ = np.array([s.striate_ for s in scags])
        self.outliers_ = np.array([s.outlying_ for s in scags])

        # 3. shape related
        self.convexes_ = np.array([s.convex_ for s in scags])
        self.skinnies_ = np.array([s.skinny_ for s in scags])
        self.stringies_ = np.array([s.stringy_ for s in scags])
        self.monotonics_ = np.array([s.monotonic_ for s in scags])

        # 4. position related
        centroids = np.array([s._X.mean(axis=0) for s in scags])
        a_centroids = np.array([[*s._alpha_hull.centroid.xy] for s in scags])[:, :, 0]
        c_centroids = np.array([[*s._convex_hull.centroid.xy] for s in scags])[:, :, 0]
        self.data_alpha_centroid_diffs_ = norm(centroids - a_centroids, axis=1)
        self.data_convex_centroid_diffs_ = norm(centroids - c_centroids, axis=1)
        # non-gaussianity measures from "Independent component analysis: algorithms and applications"
        # NOTE: consider scaling? (right now simply taking sum)
        self.kurtosises_ = np.abs(
            [(s._X**4).mean(axis=0) - 3 * ((s._X**2).mean(axis=0)) ** 2 for s in scags]
        ).sum(axis=1)

        # other non-gaussinanity inspired by "Independent component analysis: algorithms and applications"
        # NOTE: current implementation of wasserstein_distances is too slow and excluded
        self.distribution_overlaps_ = None
        self.distribution_distances_ = None
        # self.wasserstein_distances_ = None

        for _ in range(self._hist_n_repeats):
            # generation of points following gaussian distribution
            normals = [
                np.random.multivariate_normal(
                    s._X.mean(axis=0), np.cov(s._X.T), s._X.shape[0]
                )
                for s in scags
            ]
            bins = [
                [
                    np.histogram_bin_edges(
                        np.hstack((s._X[:, i], nor[:, i])), bins="auto"
                    )
                    for i in range(nor.shape[1])
                ]
                for s, nor in zip(scags, normals)
            ]
            normal_hists = [
                np.histogramdd(nor, bins=b)[0] for nor, b in zip(normals, bins)
            ]
            X_hists = [np.histogramdd(s._X, bins=b)[0] for s, b in zip(scags, bins)]

            if self.distribution_overlaps_ is None:
                self.distribution_overlaps_ = np.zeros((len(normal_hists)))
            if self.distribution_distances_ is None:
                self.distribution_distances_ = np.zeros((len(normal_hists)))
            # if self.wasserstein_distances_ is None:
            #     self.wasserstein_distances_ = np.zeros((len(normal_hists)))

            self.distribution_overlaps_ += np.array(
                [
                    np.minimum(nh, xh).sum() / nh.sum()
                    for nh, xh in zip(normal_hists, X_hists)
                ]
            )
            # NOTE: This measure is often referred as the Bhattacharyya distance but actually the Hellinger distance
            self.distribution_distances_ += np.array(
                [
                    np.sqrt(
                        1
                        - np.sum(np.sqrt(nh * xh))
                        / np.sqrt(nh.mean() * xh.mean() * nh.size**2)
                    )
                    for nh, xh in zip(normal_hists, X_hists)
                ]
            )

            # to normalize wasserstein distance generate random distribution (to avoid influence from data size)
            # randoms = [np.random.rand(*s._X.shape) for s in scags]
            # self.wasserstein_distances_ += np.array(
            #     [
            #         wasserstein_distance_nd(s._X, nor) / wasserstein_distance_nd(r, nor)
            #         for s, nor, r in zip(scags, normals, randoms)
            #     ]
            # )
        self.distribution_overlaps_ /= self._hist_n_repeats
        self.distribution_distances_ /= self._hist_n_repeats
        # self.wasserstein_distances_ /= self._hist_n_repeats

        # summarize measures
        self.min_point_ = self.points_.min()
        self.min_area_ = self.areas_.min()
        self.min_density_ = self.densities_.min()
        self.min_clumpiness_ = self.clumpinesses_.min()
        self.min_skew_ = self.skews_.min()
        self.min_sparse_ = self.sparses_.min()
        # self.min_striate_ = self.striates_.min()
        self.min_outlier_ = self.outliers_.min()
        self.min_convex_ = self.convexes_.min()
        self.min_skinny_ = self.skinnies_.min()
        self.min_stringy_ = self.stringies_.min()
        self.min_monotonic_ = self.monotonics_.min()
        self.min_data_alpha_centroid_diff_ = self.data_alpha_centroid_diffs_.min()
        self.min_data_convex_centroid_diff_ = self.data_convex_centroid_diffs_.min()
        self.min_kurtosis_ = self.kurtosises_.min()
        self.min_distribution_overlap_ = self.distribution_overlaps_.min()
        self.min_distribution_distance_ = self.distribution_distances_.min()
        # self.min_wasserstein_distance_ = self.wasserstein_distances_.min()

        self.max_point_ = self.points_.max()
        self.max_area_ = self.areas_.max()
        self.max_density_ = self.densities_.max()
        self.max_clumpiness_ = self.clumpinesses_.max()
        self.max_skew_ = self.skews_.max()
        self.max_sparse_ = self.sparses_.max()
        # self.max_striate_ = self.striates_.max()
        self.max_outlier_ = self.outliers_.max()
        self.max_convex_ = self.convexes_.max()
        self.max_skinny_ = self.skinnies_.max()
        self.max_stringy_ = self.stringies_.max()
        self.max_monotonic_ = self.monotonics_.max()
        self.max_data_alpha_centroid_diff_ = self.data_alpha_centroid_diffs_.max()
        self.max_data_convex_centroid_diff_ = self.data_convex_centroid_diffs_.max()
        self.max_kurtosis_ = self.kurtosises_.max()
        self.max_distribution_overlap_ = self.distribution_overlaps_.max()
        self.max_distribution_distance_ = self.distribution_distances_.max()
        # self.max_wasserstein_distance_ = self.wasserstein_distances_.max()

        return self

    def _fit_between_class_factors(self, X, y, axis_limits=None):
        if axis_limits is None:
            # first row: xaxis_lim, second row: yaxis_lim, ...
            axis_limits = np.vstack((X.min(axis=0), X.max(axis=0))).T

        # 1. scale related
        self.points_classes_ratio_ = X.shape[0] / len(self._uniq_labels)
        self.std_point_ = self.points_.std()
        self.std_area_ = self.areas_.std()

        # 2. point distance related
        self.std_density_ = self.densities_.std()
        self.std_skew_ = self.skews_.std()
        self.std_sparse_ = self.sparses_.std()

        self.equidistant_ = 0
        counts = 0
        for label in np.unique(y):
            counts += 1
            X_a = X[y == label]
            X_b = X[y != label]
            dists = pairwise_distances(X_a, X_b)
            self.equidistant_ += 1.0 / (
                (
                    (dists.min(axis=0) / dists.min(axis=0).mean()).std()
                    + (dists.min(axis=1) / dists.min(axis=1).mean()).std()
                )
                / 2
                + np.finfo(float).eps  # to avoid zero div
            )
        self.equidistant_ /= counts

        # NOTE: there might be a better design to capture split
        self.split_ = 0
        counts = 0
        for i in range(len(self._alpha_hulls)):
            for j in range(i, len(self._alpha_hulls)):
                counts += 1
                diff_i = self._alpha_hulls[i] - self._alpha_hulls[j]
                diff_j = self._alpha_hulls[j] - self._alpha_hulls[i]
                split_i = 0
                split_j = 0
                # if split, diff shape should be multipolygon
                if diff_i.geom_type == "MultiPolygon":
                    # further adding condition to say split (at least 10% of area)
                    split_hull_area_i = np.sum(
                        [
                            g.area if g.area / self._alpha_hulls[i].area >= 0.1 else 0
                            for g in diff_i.geoms
                        ]
                    )
                    split_i = split_hull_area_i / self._alpha_hulls[i].area
                if diff_j.geom_type == "MultiPolygon":
                    # further adding condition to say split (at least 10% of area)
                    split_hull_area_j = np.sum(
                        [
                            g.area if g.area / self._alpha_hulls[j].area >= 0.1 else 0
                            for g in diff_j.geoms
                        ]
                    )
                    split_j = split_hull_area_j / self._alpha_hulls[j].area
                self.split_ += max(split_i, split_j)
        self.split_ /= counts

        # 3. shape related
        self.std_convex_ = self.convexes_.std()
        self.std_skinny_ = self.skinnies_.std()
        self.std_stringy_ = self.stringies_.std()
        self.std_monotonic_ = self.monotonics_.std()

        # 4. position related
        # NOTE: there might be better design of this measure
        self.inner_occlusion_ratios_ = np.empty(len(self._convex_hulls))
        for a in range(len(self._convex_hulls)):
            combined_region = None
            for b in range(len(self._convex_hulls)):
                if a == b:
                    continue

                if combined_region is None:
                    combined_region = self._convex_hulls[b]
                else:
                    combined_region = combined_region.union(self._convex_hulls[b])
            self.inner_occlusion_ratios_[a] = (
                self._convex_hulls[a].intersection(combined_region).area
            ) / max(
                min(self._convex_hulls[a].area, combined_region.area),
                np.finfo(float).eps,
            )
        self.inner_occlusion_ratio_ = np.max(self.inner_occlusion_ratios_)

        # # This is previous design of inner_occulusion_ratio
        # self.inner_occlusion_ratio_ = 0
        # counts = 0
        # for i in range(len(self._convex_hulls)):
        #     for j in range(i, len(self._convex_hulls)):
        #         counts += 1
        #         self.inner_occlusion_ratio_ += self._convex_hulls[i].intersection(
        #             self._convex_hulls[j]
        #         ).area / max(
        #             min(self._convex_hulls[i].area, self._convex_hulls[j].area),
        #             np.finfo(float).eps,
        #         )
        # self.inner_occlusion_ratio_ /= counts

        # class separation related_
        self.gong_035_dir_cp_t0_ = sepme(
            X, y, class_purity_fn_kwds={"target_labels": [0]}
        )
        self.gong_035_dir_cp_t1_ = sepme(
            X, y, class_purity_fn_kwds={"target_labels": [1]}
        )
        self.distance_consistency_ = distance_consistency(X, y)
        self.density_aware_distance_consistency_ = density_aware_distance_consistency(
            X, y
        )
        self.density_aware_knng_ = density_aware_knng(X, y)

        self.alpha_overlap_ = 0
        for i in range(len(self._alpha_hulls)):
            for j in range(i, len(self._alpha_hulls)):
                self.alpha_overlap_ += (
                    self._alpha_hulls[i].intersection(self._alpha_hulls[j]).area
                )

        self.convex_overlap_ = 0
        for i in range(len(self._convex_hulls)):
            for j in range(i, len(self._convex_hulls)):
                self.convex_overlap_ += (
                    self._convex_hulls[i].intersection(self._convex_hulls[j]).area
                )

        return self

    def _fit_classif_complexity_measures(self, X, y, axis_limits=None):
        if axis_limits is None:
            # first row: xaxis_lim, second row: yaxis_lim, ...
            axis_limits = np.vstack((X.min(axis=0), X.max(axis=0))).T

        mfec = MFEComplexity()

        # precomputations to save time
        adj_graph = mfec.precompute_adjacency_graph(X, y, n_jobs=-1)["adj_graph"]
        precom_comp = mfec.precompute_complexity(y)
        # classes = precom_comp["classes"]
        ovo_comb = precom_comp["ovo_comb"]
        cls_inds = precom_comp["cls_inds"]
        class_freqs = precom_comp["class_freqs"]
        svc_pipeline = mfec.precompute_complexity_svm(y=y)["svc_pipeline"]
        precom_norm_dist = mfec.precompute_norm_dist_mat(X)
        N_scaled = precom_norm_dist["N_scaled"]
        norm_dist_mat = precom_norm_dist["norm_dist_mat"]
        orig_dist_mat_min = precom_norm_dist["orig_dist_mat_min"]
        orig_dist_mat_ptp = precom_norm_dist["orig_dist_mat_ptp"]

        # 1. Axis features
        self.c_f1v_ = mfec.ft_f1v(
            X, y, ovo_comb=ovo_comb, cls_inds=cls_inds, class_freqs=class_freqs
        )[0]
        self.c_f2_ = mfec.ft_f2(X, y, ovo_comb=ovo_comb, cls_inds=cls_inds)[0]
        self.c_f3_ = mfec.ft_f3(
            X, y, ovo_comb=ovo_comb, cls_inds=cls_inds, class_freqs=class_freqs
        )[0]
        self.c_f4_ = mfec.ft_f4(
            X, y, ovo_comb=ovo_comb, cls_inds=cls_inds, class_freqs=class_freqs
        )[0]

        # 2. Linearity features
        self.c_l1_ = mfec.ft_l1(
            X,
            y,
            ovo_comb=ovo_comb,
            cls_inds=cls_inds,
            class_freqs=class_freqs,
            svc_pipeline=svc_pipeline,
        )[0]
        self.c_l2_ = mfec.ft_l2(
            X, y, ovo_comb=ovo_comb, cls_inds=cls_inds, svc_pipeline=svc_pipeline
        )[0]
        self.c_l3_ = mfec.ft_l3(
            X, y, ovo_comb=ovo_comb, cls_inds=cls_inds, svc_pipeline=svc_pipeline
        )[0]

        # 3. Neighborhood features
        # self.c_n1_ = mfec.ft_n1(X, y, N_scaled=N_scaled, norm_dist_mat=norm_dist_mat)
        self.c_n1_ = _ft_n1(y, norm_dist_mat=norm_dist_mat)
        self.c_n2_ = mfec.ft_n2(
            X,
            y,
            class_freqs=class_freqs,
            cls_inds=cls_inds,
            N_scaled=N_scaled,
            norm_dist_mat=norm_dist_mat,
        ).mean()
        self.c_n3_ = mfec.ft_n3(
            X, y, N_scaled=N_scaled, norm_dist_mat=norm_dist_mat
        ).mean()
        self.c_n4_ = mfec.ft_n4(
            X,
            y,
            cls_inds=cls_inds,
            N_scaled=N_scaled,
            norm_dist_mat=norm_dist_mat,
            orig_dist_mat_min=orig_dist_mat_min,
            orig_dist_mat_ptp=orig_dist_mat_ptp,
        ).mean()
        self.c_t1_ = mfec.ft_t1(
            X,
            y,
            cls_inds=cls_inds,
            N_scaled=N_scaled,
            norm_dist_mat=norm_dist_mat,
            orig_dist_mat_min=orig_dist_mat_min,
            orig_dist_mat_ptp=orig_dist_mat_ptp,
        )
        # self.c_lsc_ = mfec.ft_lsc(
        #     X, y, cls_inds=cls_inds, N_scaled=N_scaled, norm_dist_mat=norm_dist_mat
        # )
        self.c_lsc_ = _ft_lsc(mfec, y, cls_inds=cls_inds, norm_dist_mat=norm_dist_mat)

        # 4. Class imbalance features
        self.c_c1_ = mfec.ft_c1(y, class_freqs=class_freqs)
        self.c_c2_ = mfec.ft_c2(y, class_freqs=class_freqs)

        # 5. Network-based features
        self.c_cls_coef_ = mfec.ft_cls_coef(
            X,
            y,
            n_jobs=-1,
            cls_inds=cls_inds,
            norm_dist_mat=norm_dist_mat,
            adj_graph=adj_graph,
        )
        self.c_density_ = mfec.ft_density(
            X,
            y,
            n_jobs=-1,
            cls_inds=cls_inds,
            norm_dist_mat=norm_dist_mat,
            adj_graph=adj_graph,
        )
        # self.c_hub_ = mfec.ft_hubs(
        #     X,
        #     y,
        #     n_jobs=-1,
        #     cls_inds=cls_inds,
        #     norm_dist_mat=norm_dist_mat,
        #     adj_graph=adj_graph,
        # ).mean()
        self.c_hub_ = np.mean(1.0 - np.asarray(adj_graph.hub_score(), dtype=np.float64))

        return self


def _ft_n1(y, norm_dist_mat=None):
    """
    Modified version from PyMFE. PyMFE has np.asfarrray, which is deprecated by Numpy 2.0.
    """
    _norm_dist_mat = np.asarray(norm_dist_mat, dtype=np.float64)
    mst = minimum_spanning_tree(csgraph=np.triu(_norm_dist_mat, k=1), overwrite=True)

    node_id_i, node_id_j = np.nonzero(mst)

    which_have_diff_cls = y[node_id_i] != y[node_id_j]

    borderline_inst_num = np.unique(
        np.concatenate(
            [
                node_id_i[which_have_diff_cls],
                node_id_j[which_have_diff_cls],
            ]
        )
    ).size

    n1 = borderline_inst_num / y.size

    return n1


def _ft_lsc(cls, y, cls_inds, norm_dist_mat, nearest_enemy_dist=None):
    """
    Modified version from PyMFE. PyMFE has np.asfarrray, which is deprecated by Numpy 2.0.
    """
    _norm_dist_mat = np.asarray(norm_dist_mat, dtype=np.float64)

    if nearest_enemy_dist is None:
        nearest_enemy_dist, _ = cls._calc_nearest_enemies(
            norm_dist_mat=_norm_dist_mat, cls_inds=cls_inds
        )

    lsc = 1.0 - np.sum(_norm_dist_mat < nearest_enemy_dist) / (y.size**2)

    return float(lsc)
