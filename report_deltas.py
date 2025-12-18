import pandas as pd
import sys


if __name__ == "__main__":
    assert sys.argv[1] in ["squad"]
    if sys.argv[1] == "squad":
        filepath = "out"
    elif sys.argv[1] == "hotpotqa":
        filepath = "out2"

    df = pd.read_csv(f"{filepath}/bandit_eval_dataset.csv", sep="\t")

    original_response_indices = []
    patched_response_indices = []
    for idx, chunk_df in df.groupby("index"):
        if chunk_df.shape[0] > 1:
            original_response_indices.append(chunk_df.index[0])
            patched_response_indices.append(chunk_df.index[-1])

    df_origi = df.iloc[original_response_indices].reset_index(drop=True)
    df_patch = df.iloc[patched_response_indices].reset_index(drop=True)

    df_origi["latency_delta"] = df_patch[["latency"]] - df_origi[["latency"]]
    df_origi["vram_delta"] = df_patch[["vram_usage"]] - df_origi[["vram_usage"]]

    print(df_origi[["failure_label", "latency_delta", "vram_delta"]].groupby("failure_label").mean())