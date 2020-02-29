"""
Computes scores.
"""
import sys
sys.path.append("../")

from datetime import datetime

from typing import Union

import nltk
from nltk.stem import WordNetLemmatizer

import numpy as np
import pandas as pd

from data_proc.proc_json import load_individual_transcript, convert_time

from tqdm import tqdm


COMPANY_PATH = "../hackathon_data/companies.csv"
JSON_PATH = "../hackathon_data/company_transcripts/"
LMD_PATH = "../sentiment_data/LoughranMcDonald_SentimentWordLists_2018.xlsx"


LMD_Dataset = pd.read_excel(
    LMD_PATH,
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
    Compute sentiment for each body paragraph.
    Returns a dictionary of six values,
    counts the number of occurences of each type of words.
    """
    # Combine body.
    sentence = " ".join(body)
    counts = dict((k, 0) for k in LMD_hash.keys())
    # Tokenize.
    tokens = nltk.word_tokenize(sentence)
    lemmatizer = WordNetLemmatizer()
    for word_type in counts.keys():
        for w in tokens:
            w = w.lower()
            c = lemmatizer.lemmatize(w)
            if c.upper() in LMD_hash[word_type]:
                counts[word_type] += 1
    return counts


def generate_dataset(
    letter_starts: Union[str, None]
) -> pd.DataFrame:
    df_company = pd.read_csv(COMPANY_PATH)
    company_lst = df_company["Ticker symbol"].values.astype(str)
    print(f"Number of companies: {len(company_lst)}")
    df_collection = {
        "Code": [],
        "Time": [],
        "Negative": [],
        "Positive": [],
        "Uncertainty": [],
        "Litigious": [],
        "StrongModal": [],
        "Constraining": [],
    }
    for num, company in enumerate(company_lst):
        if company.startswith(letter_starts) or letter_starts is None:
            print(f"Current Company: {company}")
            data = load_individual_transcript(company)
            ids = list(data["title"].keys())
            for i in ids:
                t = data["date"][i]
                date = convert_time(t)

                body = data["body"][i]
                counts = get_transcript_LMD_score(body)

                transcript_code = str(company) + "_" + str(i)
                info = {
                    "Code": transcript_code,
                    "Time": date
                }
                info.update(counts)
                for k, v in info.items():
                    df_collection[k].append(v)
    df = pd.DataFrame.from_dict(df_collection)
    df.to_csv(f"./LMD_company_{letter_starts}.csv")
    print("Saving dataset", f"./LMD_company_{letter_starts}.csv")
    return df
