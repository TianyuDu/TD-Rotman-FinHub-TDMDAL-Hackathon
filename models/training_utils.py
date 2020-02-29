"""
Training and model evaluation procedures.
"""
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

from sklearn import metrics


def data_feed(path: str) -> Tuple[np.ndarray]:
    df = pd.read_csv(path)
    X = df[[
        "Negative", "Positive",
        "Uncertainty", "Litigious",
        "StrongModal", "Constraining"
    ]].values.astype(np.float32)
    y = df["nearest_day_return"].values.astype(np.float32)
    return X, y


def training_pipeline(
    model,
    data: List[np.ndarray],
    num_fold: int = 1,
) -> dict():
    """
    Train and evaluate model.
    """
    # ==== Check Arguments ====
    val_perf_lst = {
        "loss": list(),
        "directional_accuracy": list()
    }
    train_perf_lst = {
        "loss": list(),
        "directional_accuracy": list()
    }
    val_perf = dict()
    # ==== Data Preprocessing ====
    X_train_raw, y_train_raw = data
    # ==== N-Fold ====
    val_loss_fold = list()
    val_diracc_fold = list()
    for _ in range(num_fold):
        # Train validation split.
        # Note that the order of dataset returned by
        # train_test_split method is odd.
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_raw, y_train_raw,
            train_size=0.75,
            shuffle=True
        )
        X_train, X_val, y_train, y_val = map(
            np.squeeze,
            (X_train, X_val, y_train, y_val)
        )
        model.fit(X_train, y_train)
        pred_val = model.predict(X_val)
        # directional accuracy.
        direction = np.mean(np.sign(pred_val) == np.sign(y_val))
        # loss
        val_perf_lst["loss"].append(
            metrics.mean_squared_error(y_val, pred_val)
        )
        val_perf_lst["directionary_accuracy"].append(direction)
    for k, v in val_perf_lst.items():
        val_perf[k] = np.mean(v)
    return val_perf
