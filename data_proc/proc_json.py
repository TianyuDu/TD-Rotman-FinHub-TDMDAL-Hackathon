"""
Process json files.
"""
from typing import List

import numpy as np
import pandas as pd

import json


TRANSCRIPT_DIR = "../hackathon_data/company_transcripts/"


def load_individual_transcript(
    company: str,
    subset: List[str] = None
) -> dict:
    path = TRANSCRIPT_DIR + company + ".json"
    with open(path, "r") as f:
        data = json.load(f)
    return data
