import sys
from evaluation import evaluate_rag
import pandas as pd
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt


def plot_action_heatmaps(filepath, heatmap_data1, heatmap_data2):
    elements1 = sorted({pair[0] for pair in heatmap_data1})
    elements2 = sorted({pair[1] for pair in heatmap_data1})
    index_map1 = {elem: idx for idx, elem in enumerate(elements1)}
    index_map2 = {elem: idx for idx, elem in enumerate(elements2)}
    pair_counts = Counter(heatmap_data1)
    matrix = np.zeros((len(elements1), len(elements2)), dtype=int)

    for (a, b), count in pair_counts.items():
        i, j = index_map1[a], index_map2[b]
        matrix[i, j] = count

    plt.figure(figsize=(40, 22))
    plt.imshow(matrix, cmap='hot')
    plt.xticks(ticks=range(len(elements2)), labels=elements2, rotation=45, fontsize=30)
    plt.yticks(ticks=range(len(elements1)), labels=elements1, rotation=45, fontsize=30)
    plt.savefig(f"{filepath}/action_vs_label.pdf")

    elements1 = sorted({pair[0] for pair in heatmap_data2})
    elements2 = sorted({pair[1] for pair in heatmap_data2})
    index_map1 = {elem: idx for idx, elem in enumerate(elements1)}
    index_map2 = {elem: idx for idx, elem in enumerate(elements2)}
    pair_counts = Counter(heatmap_data2)
    matrix = np.zeros((len(elements1), len(elements2)), dtype=int)

    for (a, b), count in pair_counts.items():
        i, j = index_map1[a], index_map2[b]
        matrix[i, j] = count

    plt.figure(figsize=(16, 14))
    plt.imshow(matrix, cmap='hot')
    plt.xticks(ticks=range(len(elements2)), labels=elements2, rotation=45, fontsize=30)
    plt.yticks(ticks=range(len(elements1)), labels=elements1, rotation=45, fontsize=30)
    plt.savefig(f"{filepath}/action_vs_component.pdf")

def translate_actions(action):
    action = eval(action)
    if "reranker" in action:
        action = "Reranker activation"
    elif "prompt_edit" in action:
        if action["prompt_edit"] == "paraphrase_qa":
            action = "Paraphrse query"
        elif action["prompt_edit"] == "simplify_qa":
            action = "Simplify query"
    elif "retriever" in action:
        action = f"{action['retriever']}, top-{action['topk']}"
            
    return action

if __name__ == "__main__":
    assert sys.argv[1] in ["squad"]
    if sys.argv[1] == "squad":
        filepath = "out"
    elif sys.argv[1] == "hotpotqa":
        filepath = "out2"
        
    df = pd.read_csv(f"{filepath}/bandit_eval_dataset.csv", sep="\t")

    patched_response_indices = []
    heatmap_data1 = []
    heatmap_data2 = []
    for idx, chunk_df in df.groupby("index"):
        patched_response_indices.append(chunk_df.index[-1])

        if len(chunk_df) > 1:
            last_action = chunk_df.iloc[0]["params"]
            last_failure_label = chunk_df.iloc[0]["failure_label"]
            current_action = chunk_df.iloc[-1]["params"]
            heatmap_data1.append((last_failure_label, translate_actions(current_action)))

            if last_failure_label in ["RETRIEVER_FAILURE", "INSUFFICIENT_EVIDENCE_FAILURE"]:
                heatmap_data2.append(("Retrieval", translate_actions(current_action)))
            elif last_failure_label in ["WRONG_PREDICATE_FAILURE", "WRONG_RESPONSE_FAILURE", "LABEL_EVIDENCE_MISMATCH_FAILURE"]:
                heatmap_data2.append(("Generation", translate_actions(current_action)))

    plot_action_heatmaps(filepath, heatmap_data1, heatmap_data2)

    df = df.iloc[patched_response_indices]
    evaluate_rag(df)