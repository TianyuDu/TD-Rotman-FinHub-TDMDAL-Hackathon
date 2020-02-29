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
    print("Number of transcripts with missing data:")
    print(f"{np.mean(np.sum(df.isnull(), axis=1)) * 100: 0.3f}%")
    df = df.dropna()
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
    train_perf_lst = {
        "loss": list(),
        "directional_accuracy": list()
    }
    val_perf_lst = {
        "loss": list(),
        "directional_accuracy": list()
    }
    # ==== Data Preprocessing ====
    X_train_raw, y_train_raw = data
    # ==== N-Fold ====
    for i in range(num_fold):
        print(f"Running Fold: {i}")
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

        # Evaluate on training set.
        pred_train = model.predict(X_train)
        train_acc = np.mean(np.sign(pred_train) == np.sign(y_train))
        train_loss = metrics.mean_squared_error(y_train, pred_train)

        # Evaluate on validation set.
        pred_val = model.predict(X_val)
        val_acc = np.mean(np.sign(pred_val) == np.sign(y_val))
        val_loss = metrics.mean_squared_error(y_val, pred_val)

        # Record loss.
        train_perf_lst["directional_accuracy"].append(train_acc)
        train_perf_lst["loss"].append(train_loss)

        val_perf_lst["directional_accuracy"].append(val_acc)
        val_perf_lst["loss"].append(val_loss)

    # Compute average.
    train_perf, val_perf = dict(), dict()

    for k, v in train_perf_lst.items():
        train_perf[k] = np.mean(v)

    for k, v in val_perf_lst.items():
        val_perf[k] = np.mean(v)
    return train_perf, val_perf
