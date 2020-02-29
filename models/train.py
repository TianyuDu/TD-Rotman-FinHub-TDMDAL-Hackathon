"""
Main training file.
"""
import pandas as pd
from training_utils import data_feed, training_pipeline


DATA_PATH = "/Users/tianyudu/Documents/TD-Rotman-FinHub-TDMDAL-Hackathon/sentiment_data/LMD_data_all_returns.csv"


if __name__ == "__main__":
    (X, y) = data_feed(path=DATA_PATH)
    config = dict()
    model = construct_model(config)
    train_perf, val_perf = training_pipeline(
        model,
        data=(X, y),
        num_fold=5
    )
    print("Training Performance:")
    print(train_perf)
    print("Validation Performance:")
    print(val_perf)
