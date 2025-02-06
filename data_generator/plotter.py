import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def plot_scatter(X, y, out_path=None, s=20, return_ax=False):
    cmap = "viridis"
    if y.max() <= 10:
        cmap = "tab10"
        cmap = ListedColormap(plt.cm.tab10.colors[: y.max() + 1])
    elif y.max() <= 20:
        cmap = ListedColormap(plt.cm.tab20.colors[: y.max() + 1])

    _, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6), tight_layout=True)
    ax.scatter(
        X[:, 0],
        X[:, 1],
        c=y,
        cmap=cmap,
        s=s,
        alpha=0.8,
        linewidth=0.0,
    )
    ax.set_box_aspect(1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    if return_ax:
        return ax
    else:
        plt.savefig(out_path)
        plt.close()
