import numpy as np
import pandas as pd


def id_generator(start_id=0):
    current_id = start_id - 1

    def _incr_id():
        nonlocal current_id
        current_id += 1
        return current_id

    return _incr_id


def clean_data(
    df, exclude_dtypes=["object"], remove_ids=True, categorical_indicator=None
):
    if categorical_indicator is not None:
        df = df[df.columns[np.invert(categorical_indicator)]]
    df = df.select_dtypes(exclude=exclude_dtypes)
    if remove_ids:
        is_int = df.dtypes == "int"
        is_last_chars_id = np.array([name[-2:].lower() == "id" for name in df.columns])
        remain_cols = df.columns[np.invert(is_int & is_last_chars_id)]
        df = df[remain_cols]

    # to handle timestamps (i.e., timestamps => int64)
    df = df.apply(pd.to_numeric)

    return df


def load_scatter_data(filepath):
    scatter_data = pd.read_csv(filepath)
    if (
        (not "x" in scatter_data)
        or (not "y" in scatter_data)
        or (not "label" in scatter_data)
    ):
        Z = None
        y = None
    else:
        Z = np.vstack((np.array(scatter_data["x"]), np.array(scatter_data["y"]))).T
        y = np.array(scatter_data["label"])
    return Z, y


def target_class_selection(X, y):
    y_new = y.copy()
    if y_new.max() > 1:
        # priorize a larger class as a target class
        uniq_labels, counts = np.unique(y_new, return_counts=True)
        target_class = np.random.choice(uniq_labels, p=counts / counts.sum())
        y_new[y != target_class] = 0
        y_new[y == target_class] = 1
    return y_new
