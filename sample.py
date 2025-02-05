import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs, make_moons, make_circles
from multiclass_scatterplot_features import MulticlassScatterFeatures


def plot(
    datasets,
    outputs,
    measure_names=[
        "equidistant_",
        "inner_occlusion_ratio_",
        "alpha_overlap_",
        "gong_035_dir_cp_t0_",
        "c_f1v_",
        "c_l1_",
    ],
    figsize=(12, 8),
):
    fig, axs = plt.subplots(
        nrows=1, ncols=len(datasets) + 1, figsize=figsize, tight_layout=True
    )
    axs[0].scatter(*datasets[0][0].T, s=5, alpha=0)
    axs[0].set_aspect("equal")
    axs[0].set_box_aspect(1)
    axs[0].axis("off")

    cell_colors = np.array(["#ffffff"] * len(measure_names))[:, None]
    cell_colors[1::2] = "#f4f4f4"
    table = axs[0].table(
        np.array(measure_names)[:, None],
        cellColours=cell_colors,
        loc="bottom",
        edges="closed",
    )
    table.scale(1, 3)
    for c in table.properties()["celld"].values():
        c.set(linewidth=0)

    for i, (X, y) in enumerate(datasets):
        axs[i + 1].scatter(*X.T, c=y)
        axs[i + 1].set_aspect("equal")
        axs[i + 1].set_box_aspect(1)
        axs[i + 1].set_xticks([])
        axs[i + 1].set_yticks([])

        measures = np.array(
            [getattr(outputs[i], name) for name in measure_names]
        ).astype("object")[:, None]
        table = axs[i + 1].table(
            measures, cellColours=cell_colors, loc="bottom", edges="closed"
        )
        table.scale(1, 3)
        for c in table.properties()["celld"].values():
            c.set(linewidth=0)
    plt.show()


if __name__ == "__main__":
    blobs = make_blobs(centers=2, cluster_std=0.5, random_state=0)
    moons = make_moons(random_state=0)
    circles = make_circles(random_state=0)

    outputs = []
    for X, y in [blobs, moons, circles]:
        mcsf = MulticlassScatterFeatures()
        mcsf.fit(X, y)
        print("gong_035_dir_cp_t0_", mcsf.gong_035_dir_cp_t0_)
        print("equidistant_", mcsf.equidistant_)
        outputs.append(mcsf)

    plot(
        [blobs, moons, circles],
        outputs,
        measure_names=[
            "gong_035_dir_cp_t0_",
            "equidistant_",
            "min_skinny_",
            "min_kurtosis_",
            "inner_occlusion_ratio_",
            "convex_overlap_",
            "c_f1v_",
            "c_l1_",
        ],
    )
