"""
Random Forest
"""
import numpy as np
from sklearn.ensemble import RandomForestRegressor


def construct_model(
    config: dict
) -> "RandomForestRegressor":
    model = RandomForestRegressor(**config)
    return model
