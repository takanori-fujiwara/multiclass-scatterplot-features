from pathlib import Path

from natsort import natsorted

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

from utils import load_scatter_data
from multiclass_scatterplot_features import MulticlassScatterFeatures


class MetadataGenerator:
    def __init__(self):
        None

    def extract_scatter_metadata(
        self,
        scatter_data_dir="scatter_data_binarized",
        output_path="scatter_meta.csv",
        start_index=0,
        total=False,
    ):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        mcsf = MulticlassScatterFeatures()
        scatter_meta = []

        n_files = len(list(Path(scatter_data_dir).glob("*.csv")))
        for i, path in enumerate(natsorted(Path(scatter_data_dir).glob("*.csv"))):
            if i < start_index:
                continue

            Z, y = load_scatter_data(path)
            if Z.shape[0] - np.isnan(Z).sum() / Z.shape[1] <= 2:
                continue

            # remove nan values
            notnan = np.isnan(Z).sum(axis=1) == 0
            Z = Z[notnan]
            y = y[notnan]

            if total:
                y = np.zeros_like(y)

            _, counts = np.unique(y, return_counts=True)

            scatter_data_name = path.stem
            if (Z is not None) and counts.min() > 3:
                print(f"{i} / {n_files}: {Z.shape} ({path})")
                # should apply min-max scaling to mimic matplotlib's default axis handling
                Z = MinMaxScaler().fit_transform(Z)
                metadata = {"id": scatter_data_name}

                mcsf.fit(Z, y)
                for attr_name in dir(mcsf):
                    # all measures end with _
                    if (attr_name[-1] == "_") and (attr_name[0] != "_"):
                        attr = getattr(mcsf, attr_name)
                        if not isinstance(attr, np.ndarray):
                            metadata[attr_name] = attr
                scatter_meta.append(metadata)
                if output_path is not None:
                    pd.DataFrame(scatter_meta).to_csv(output_path, index=False)

        return pd.DataFrame(scatter_meta)
