"""
Computes scores.
"""
from datetime import datetime

import nltk
import numpy as np
import pandas as pd

lmd_path = "../sentiment_data/LoughranMcDonald_SentimentWordLists_2018.xlsx"

LMD_Dataset = pd.read_excel(
    lmd_path,
    sheet_name=[
        "Negative", "Positive",
        "Uncertainty", "Litigious",
        "StrongModal", "Constraining"],
    header=None
)

LMD_hash = dict()
for k, v in LMD_Dataset.items():
    LMD_hash[k] = list(map(lambda x: x[0], v.values))


def get_transcript_LMD_score(
    body: list
) -> np.ndarray:
    """
    Compute sentiment for each body.
    """
    body = " ".join(body)
    