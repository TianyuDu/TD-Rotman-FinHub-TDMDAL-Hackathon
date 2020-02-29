"""
Random Forest
"""
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from grid_search_util import profile_generator


def construct_model(
    config: dict
) -> "RandomForestRegressor":
    model = RandomForestRegressor(**config)
    return model


def tune_rf()
