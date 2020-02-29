"""
Load discussion and QA parts of each transcript.
"""
from typing import List

import numpy as np
import pandas as pd

import proc_json


def load_splitted_transcript() -> List[dict]:
    df = pd.read_csv(
        "./sentiment_data/LMD_data_all_returns.csv"
    )

    df_compnay = pd.read_csv(
        "./hackathon_data/companies.csv"
    )
    company_lst = df_compnay["Ticker symbol"].values

    def qa_check(sentence: str) -> bool:
        sentence = sentence.lower()
        if len(sentence) > 100:
            return False
        if "questions" in sentence and "answers" in sentence:
            return True
        if "question" in sentence and "answer" in sentence:
            return True
        return False

    ds_dis = dict()
    ds_qa = dict()

    places, length = list(), list()
    failed = 0
    for company in company_lst:
        all_transcript = proc_json.load_individual_transcript(
            company=company, path="./hackathon_data/company_transcripts/")
        for transcript_id in all_transcript["title"].keys():
            unique_id = company + "_" + transcript_id

            title = all_transcript["title"][transcript_id]
            date = all_transcript["date"][transcript_id]
            body = all_transcript["body"][transcript_id]
            # Total length of body.
            length.append(len(body))

            qa_begins = None
            for p, sentence in enumerate(body):
                if qa_check(sentence):
                    qa_begins = p
                    break
            if qa_begins is None:
                # If we cannot find such QA identifier.
                # Assume 0.3 Speech + 0.7 QA (aggregate prior).
                qa_begins = int(len(body) * 0.3)
                failed += 1
            places.append(qa_begins)
            # Split the dataset
            ds_dis.update({str(unique_id): " ".join(body[:qa_begins])})
            ds_qa.update({str(unique_id): " ".join(body[qa_begins:])})
    return ds_dis, ds_qa
