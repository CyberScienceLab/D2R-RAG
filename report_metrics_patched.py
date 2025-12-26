import sys
from evaluation import evaluate_rag
import pandas as pd
import numpy as np


def translate_actions(action):
    action = eval(action)
    if "reranker" in action:
        action = "Reranker activation"
    elif "prompt_edit" in action:
        if action["prompt_edit"] == "paraphrase_qa":
            action = "Paraphrase query"
        elif action["prompt_edit"] == "simplify_qa":
            action = "Simplify query"
    elif "retriever" in action:
        action = f"{action['retriever']}, top-{action['topk']}, reindex={action['reindex']}"
            
    return action

if __name__ == "__main__":
    assert sys.argv[1] in ["fever", "hotpotqa"]
    if sys.argv[1] == "fever":
        filepath = "files_fever_v"
    elif sys.argv[1] == "hotpotqa":
        filepath = "files_hotpotqa_v"
        
    pre_df = pd.read_csv(f"{filepath}/failure_statistics.csv", sep="\t")
    post_df = pd.read_csv(f"{filepath}/bandit_eval_dataset.csv", sep="\t")

    failure_action_set = []
    delta_latency_per_patch = {}
    delta_vram_per_patch = {}
    for (_, pre_row), (_, post_row) in zip(pre_df.iterrows(), post_df.iterrows()):
        if pre_row["failure_label"] != "NO_FAILURE":
            failure_action_set.append((pre_row["failure_label"], translate_actions(post_row["params"])))

        delta_latency = delta_latency_per_patch.get(post_row["params"], [])
        delta_latency.append(post_row["latency"] - pre_row["latency"])
        delta_latency_per_patch[post_row["params"]] = delta_latency

        delta_vram = delta_vram_per_patch.get(post_row["params"], [])
        delta_vram.append(post_row["vram_usage"] - pre_row["vram_usage"])
        delta_vram_per_patch[post_row["params"]] = delta_vram

    action_frequency_by_failure_type = np.unique(failure_action_set, return_counts=True, axis=0)
    print("Action frequency by failure type:", action_frequency_by_failure_type)

    delta_latency_per_patch = {k:np.mean(v) for k, v in delta_latency_per_patch.items()}
    print("Delta latency per patch:", delta_latency_per_patch)

    delta_vram_per_patch = {k:np.mean(v) for k, v in delta_vram_per_patch.items()}
    print("Delta VRAM per patch:", delta_vram_per_patch)

    print("Postpatch-latency (per failure label):", post_df[["failure_label", "latency"]].groupby("failure_label").mean())
    print("Postpatch-latency (overall):", post_df["latency"].mean())

    print("Postpatch-VRAM (per failure label):", post_df[["failure_label", "vram_usage"]].groupby("failure_label").mean())
    print("Postpatch-VRAM (overall):", post_df["vram_usage"].mean())

    post_df['is_consistent'] = post_df['kg_consistency'] == 'CONSISTENT'
    print("Postpatch-KG Consistency (per failure label):", post_df[["failure_label", "is_consistent"]].groupby("failure_label").mean())
    print("Postpatch-KG Consistency (overall):", (post_df['kg_consistency'] == 'CONSISTENT').mean())

    for label in post_df["failure_label"].unique():
        print("evaluaing", label)
        df_label = post_df[post_df["failure_label"] == label]
        evaluate_rag(df_label, sys.argv[1])
        print("evaluaing", label, "done")

    print("evaluaing overall")
    evaluate_rag(post_df, sys.argv[1])
    print("evaluaing overall done")