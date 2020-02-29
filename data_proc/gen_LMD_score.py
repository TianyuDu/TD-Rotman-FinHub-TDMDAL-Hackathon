"""
Generate sentiment scores using 
"""
import argparse
from compute_score_LMD import generate_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--l", type=str)
    args = parser.parse_args()
    letter = args.l.upper()
    df = generate_dataset(letter_starts=letter)
