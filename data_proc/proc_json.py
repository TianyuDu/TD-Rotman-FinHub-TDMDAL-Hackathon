"""
Process json files.
"""
import json
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd

TRANSCRIPT_DIR = "../hackathon_data/company_transcripts/"


def load_individual_transcript(
    company: str,
    subset: List[str] = None
) -> dict:
    path = TRANSCRIPT_DIR + company + ".json"
    with open(path, "r") as f:
        data = json.load(f)
    return data


def convert_time(s: str) -> datetime:
    return datetime.fromtimestamp(s / 1000).strftime("%Y-%m-%d %H:%M:%S.%f")
