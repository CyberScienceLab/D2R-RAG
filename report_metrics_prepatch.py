import sys
from evaluation import evaluate_rag
import pandas as pd


if __name__ == "__main__":
    assert sys.argv[1] in ["squad"]
    if sys.argv[1] == "squad":
        filepath = "out"
    elif sys.argv[1] == "hotpotqa":
        filepath = "out2"
        
    df = pd.read_csv(f"{filepath}/bandit_eval_dataset.csv", sep="\t")

    original_response_indices = []
    for idx, chunk_df in df.groupby("index"):
        original_response_indices.append(chunk_df.index[0])

    df = df.iloc[original_response_indices]
    evaluate_rag(df)