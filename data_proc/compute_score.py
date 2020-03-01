"""
Computes dataset.
"""
from typing import Union

import numpy as np
import pandas as pd

import nltk
from nltk.stem import WordNetLemmatizer

from proc_json import load_individual_transcript, convert_time

from split_qa import load_splitted_transcript


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

D_SPLITTED_BDOY, Q_SPLITTED_BDOY = load_splitted_transcript()


TYPES = [
    "Negative", "Positive",
    "Uncertainty", "Litigious",
    "StrongModal", "Constraining"
]


financial_dataset = pd.read_excel(
    # "/Users/tianyudu/Documents/TD-Rotman-FinHub-TDMDAL-Hackathon/sentiment_data/Finance_Dic.xlsx"
    "../sentiment_data/Finance_Dic.xlsx"
)


def get_score(
    body: str,
    prefix: str
) -> np.ndarray:
    """
    Compute sentiment for each body paragraph.
    Returns a dictionary of six values,
    counts the number of occurences of each type of words.
    """
    counts = dict((prefix + k, 0) for k in LMD_hash.keys())
    # Tokenize.
    tokens = nltk.word_tokenize(body)
    lemmatizer = WordNetLemmatizer()
    for word_type in TYPES:
        for w in tokens:
            w = w.lower()
            c = lemmatizer.lemmatize(w)
            if c.upper() in LMD_hash[word_type]:
                counts[prefix + word_type] += 1
            if c.lower() in 
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
        "d_Negative": [],
        "d_Positive": [],
        "d_Uncertainty": [],
        "d_Litigious": [],
        "d_StrongModal": [],
        "d_Constraining": [],
        "qa_Negative": [],
        "qa_Positive": [],
        "qa_Uncertainty": [],
        "qa_Litigious": [],
        "qa_StrongModal": [],
        "qa_Constraining": [],
    }

    for num, company in enumerate(company_lst):
        if company.startswith(letter_starts) or letter_starts is None:
            print(f"Current Company: {company}")
            data = load_individual_transcript(company)
            trainscript_ids = list(data["title"].keys())
            for i in trainscript_ids:
                transcript_code = str(company) + "_" + str(i)

                t = data["date"][i]
                date = convert_time(t)

                # Compute scores for each part.
                discussion_part = D_SPLITTED_BDOY[transcript_code]
                qa_part = Q_SPLITTED_BDOY[transcript_code]

                discussion_counts = get_score(discussion_part, prefix="d_")
                qa_counts = get_score(qa_part, prefix="qa_")

                info = {
                    "Code": transcript_code,
                    "Time": date
                }

                info.update(discussion_counts)
                info.update(qa_counts)

                for k, v in info.items():
                    df_collection[k].append(v)
    df = pd.DataFrame.from_dict(df_collection)
    df.to_csv(f"./LMD_company_{letter_starts}.csv")
    print("Saving dataset", f"./LMD_QA_company_{letter_starts}.csv")
    return df
