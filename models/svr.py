"""
Baseline Support Vector Regressor.
"""
import numpy as np
import pandas as np

from sklearn.svm import SVR


def construct_model(
    config: dict
) -> "SVR":
    model = SVR(**config)
    return model
