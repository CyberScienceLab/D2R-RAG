import sys
from evaluation import evaluate_rag
import pandas as pd


if __name__ == "__main__":
    assert sys.argv[1] in ["fever", "hotpotqa"]
    if sys.argv[1] == "fever":
        filepath = "files_fever_v"
    elif sys.argv[1] == "hotpotqa":
        filepath = "files_hotpotqa_v"
        
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
        evaluate_rag(df_label, sys.argv[1])
        print("evaluaing", label, "done")

    print("evaluaing overall")
    evaluate_rag(df, sys.argv[1])
    print("evaluaing overall done")