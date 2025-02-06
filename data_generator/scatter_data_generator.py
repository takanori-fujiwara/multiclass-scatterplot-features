import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import scale

import plotter, custom_dr, utils


class ScatterDataGenerator:
    def __init__(self):
        self.scatter_info = []

    def run_all(
        self,
        hd_to_2d_kwargs={},
        scatter_data_to_plot_kwargs={},
        hd_data_dir="original_data",
        scatter_data_dir="scatter_data",
        scatter_info_path="scatter_info.csv",
        scatter_plot_dir="scatter_data/plot",
        binarized_scatter_data_dir="scatter_data_binarized",
        binarized_scatter_plot_dir="scatter_data_binarized/plot",
    ):
        self.hd_to_2d(
            input_dir=hd_data_dir, output_dir=scatter_data_dir, **hd_to_2d_kwargs
        )
        self.output_scatter_info(scatter_info_path)
        self.scatter_data_to_plot(
            scatter_data_dir=scatter_data_dir,
            output_dir=scatter_plot_dir,
            **scatter_data_to_plot_kwargs,
        )

        self.binarize_scatter_data(
            scatter_data_dir=scatter_data_dir, output_dir=binarized_scatter_data_dir
        )
        self.scatter_data_to_plot(
            scatter_data_dir=binarized_scatter_data_dir,
            output_dir=binarized_scatter_plot_dir,
            **scatter_data_to_plot_kwargs,
        )
        return self

    def hd_to_2d(
        self,
        input_dir="original_data",
        output_dir="scatter_data",
        n_insts_range=(50, 10000),
        scalers=[scale],
        dr_methods=[
            custom_dr.Random().fit_transform,
            custom_dr.IPCA().fit_transform,
            custom_dr.MDS().fit_transform,
            custom_dr.TSNE().fit_transform,
            custom_dr.UMAP().fit_transform,
            custom_dr.PHATE().fit_transform,
            custom_dr.LDA().fit_transform,
            custom_dr.CPCA().fit_transform,
            custom_dr.CCPCA().fit_transform,
            custom_dr.RandomPramULCA().fit_transform,
        ],
        skip=0,
    ):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        _assign_id = utils.id_generator(len(list(Path("scatter_data").glob("*.csv"))))

        n_files = len(list(Path(input_dir).glob("*.pkl")))
        for i, path in enumerate(Path(input_dir).glob("*.pkl")):
            if i < skip:
                continue

            with open(path, "rb") as f:
                original_data = pickle.load(f)

            X = original_data["X"]
            y = np.array(original_data["y"])

            print(f"{i} / {n_files}: {X.shape}")
            if (X.shape[0] < n_insts_range[0]) or (X.shape[0] > n_insts_range[1]):
                continue
            # TODO: remove this part later (should be handled in clean_data in data_downloader)
            # handling case with datetime
            if X.dtype == "object":
                X = np.array(pd.DataFrame(X).apply(pd.to_numeric))

            for scaler in scalers:
                X_scl = scaler(X)
                for dr_method in dr_methods:
                    print(scaler.__name__, str(dr_method.__self__))
                    try:
                        Z = dr_method(X_scl, y)
                    except Exception:
                        print(f"{dr_method} doesn't work well on this data.")
                        continue

                    if (Z is None) or (Z.shape[0] <= 0):
                        print(f"Z is not generated")
                    else:
                        scatter_id = _assign_id()
                        scatter_data = pd.DataFrame(
                            {
                                "x": Z[:, 0],
                                "y": Z[:, 1],
                                "label": y,
                            }
                        )
                        scatter_data.to_csv(
                            f"{output_dir}/{scatter_id}.csv", index=False
                        )

                        self.scatter_info.append(
                            {
                                "id": scatter_id,
                                "data": path.stem,
                                "dr": str(dr_method.__self__),
                                "scaler": scaler.__name__,
                            }
                        )
        return self

    def output_scatter_info(self, output_path="scatter_info.csv"):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.scatter_info).to_csv(output_path, index=False)
        return self

    def scatter_data_to_plot(
        self,
        scatter_data_dir="scatter_data_binarized",
        output_dir="scatter_data_binarized/plot",
        plot_kwargs={},
    ):
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        for path in Path(scatter_data_dir).glob("*.csv"):
            Z, y = utils.load_scatter_data(path)
            scatter_data_name = path.stem
            if Z is not None:
                # if not (y.dtype in ["int8", "int16", "int32", "int64"]):
                #     print(scatter_data_name, y.dtype, y.max())
                #     y = pd.DataFrame(y)
                #     y = y.iloc[:, 0].astype("category")
                #     target_names = y.cat.categories
                #     y = np.array(y.cat.codes)
                #     scatter_data = pd.read_csv(path)
                #     scatter_data["label"] = y
                #     scatter_data.to_csv(
                #         f"scatter_data_modified/{scatter_data_name}.csv", index=False
                #     )
                #     info = pd.read_csv("scatter_info.csv")
                #     data_name = info["data"][info["id"] == int(scatter_data_name)].iloc[
                #         0
                #     ]
                #     with open(f"original_data/{data_name}.pkl", "rb") as f:
                #         original_data = pickle.load(f)
                #     original_data["y"] = y
                #     original_data["target_names"] = target_names
                #     with open(f"original_data_modified/{data_name}.pkl", "wb") as f:
                #         pickle.dump(original_data, f)

                plotter.plot_scatter(
                    Z, y, f"{output_dir}/{scatter_data_name}.png", **plot_kwargs
                )

        return self

    def binarize_scatter_data(
        self, scatter_data_dir="scatter_data", output_dir="scatter_data_binarized"
    ):
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        for path in Path(scatter_data_dir).glob("*.csv"):
            Z, y = utils.load_scatter_data(path)
            scatter_data_name = path.stem
            if Z is not None:
                # select only one as a target class
                y = utils.target_class_selection(Z, y)
                scatter_data = pd.DataFrame(
                    {
                        "x": Z[:, 0],
                        "y": Z[:, 1],
                        "label": y,
                    }
                )
                scatter_data.to_csv(
                    f"{output_dir}/{scatter_data_name}.csv", index=False
                )
        return self
