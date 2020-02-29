"""
Computes scores.
"""
import numpy as np
import pandas as pd

from datetime import datetime


lmd_path = "../sentiment_data/LoughranMcDonald_SentimentWordLists_2018.xlsx"

LMD = pd.read_excel(
    lmd_path,
    sheet_name=["Negative", ],
    header=False
)


def get_transcript_LMD_score(
    body: str
) -> np.ndarray:
    """
    Compute sentiment
    """
    raise NotImplementedError
