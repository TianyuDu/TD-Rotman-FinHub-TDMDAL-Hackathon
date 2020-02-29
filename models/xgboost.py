"""
XGBoost
Note that use xgb.DMatrix() to change the data type.
"""
import argparse

import numpy as np
import xgboost as xgb

from grid_search_util import profile_generator, grid_search
from training_utils import data_feed, training_pipeline


def construct_model(
    config: dict
) -> "XGBRegressor":
    model = xgb.XGBRegressor(**config)
    return model


if __name__ == "__main__":
    random_grid = {
        "max_depth": max_depth,
        "n_estimators": n_estimators,
        "learning_rate": learning_rate}

    DATA_PATH = "../TD-Rotman-FinHub-TDMDAL-Hackathon/sentiment_data/LMD_data_all_returns.csv"

    grid_search(
        scope=random_grid,
        model_constructor=construct_model,
        data=data_feed(DATA_PATH),
        training_pipeline=training_pipeline,
        log_dir="./rf_grid_result_cv5.csv"
    )
