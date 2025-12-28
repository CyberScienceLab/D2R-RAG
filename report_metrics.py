import sys
from evaluation import evaluate_rag
import pandas as pd


if __name__ == "__main__":
    dataset_name = sys.argv[1]
    assert dataset_name in ["fever", "fever_ts", "hotpotqa", "hotpotqa_ts", "fever_selfrag", "hotpotqa_selfrag"]
    if dataset_name == "fever":
        filepath = "files_fever_v"
    elif dataset_name == "fever_ts":
        dataset_name = "fever"
        filepath = "files_fever_ts_v"
    elif dataset_name == "hotpotqa":
        filepath = "files_hotpotqa_v"
    elif dataset_name == "hotpotqa_ts":
        filepath = "files_hotpotqa_ts_v"
    elif dataset_name == "fever_selfrag":
        dataset_name = "fever"
        filepath = "files_fever_selfrag_v"
    elif dataset_name == "hotpotqa_selfrag":
        filepath = "files_hotpotqa_selfrag_v"
        
    df = pd.read_csv(f"{filepath}/failure_statistics.csv", sep="\t")

    print("Prepatch-latency (per failure label):", df[["failure_label", "latency"]].groupby("failure_label").mean())
    print("Prepatch-latency (overall):", df["latency"].mean())

    print("Prepatch-VRAM (per failure label):", df[["failure_label", "vram_usage"]].groupby("failure_label").mean())
    print("Prepatch-VRAM (overall):", df["vram_usage"].mean())

    df['is_consistent'] = df['kg_consistency'] == 'CONSISTENT'
    print("Prepatch-KG Consistency (per failure label):", df[["failure_label", "is_consistent"]].groupby("failure_label").mean())
    print("Prepatch-KG Consistency (overall):", (df['kg_consistency'] == 'CONSISTENT').mean())

    for label in df["failure_label"].unique():
        print("evaluaing", label)
        df_label = df[df["failure_label"] == label]
        evaluate_rag(df_label, dataset_name)
        print("evaluaing", label, "done", df_label.shape)