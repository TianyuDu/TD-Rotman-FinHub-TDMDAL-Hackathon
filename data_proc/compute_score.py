"""
Computes scores.
"""
from datetime import datetime

import nltk
from nltk.stem import WordNetLemmatizer

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
    sentence = " ".join(body)
    counts = dict((k, 0) for k in LMD_hash.keys())
    tokens = nltk.word_tokenize(sentence)
    lemmatizer = WordNetLemmatizer()
    for k in counts.keys():
        for w in tokens:
            w = w.lower()
            c = lemmatizer.lemmatize(w)
            if c.upper() in LMD_hash[k]:
                counts[k] += 1
    return counts


def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))


def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result
