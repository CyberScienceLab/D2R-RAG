import os
# prevent JAX from using gpu to avoid bm25s eat up the gpu memory
os.environ['JAX_PLATFORMS'] = 'cpu'

import sys
from knowledge_graph import KnowledgeGraph
from utils import setup_settings, load_fever
from selfrag_engine import SelfRAGEngine
import tqdm
from datasets import Dataset
from patcher import BanditPatcher 
from qa_metrics.em import em_match


if __name__ == "__main__":
    dataset_name = sys.argv[1]
    assert dataset_name in ["fever"]
    if dataset_name == "fever":
        filepath = "files_fever_selfrag_v"
        dataset, knowledgebase = load_fever(filepath, split='validation')
        
    setup = setup_settings(dataset_name)

    kgraph = KnowledgeGraph(filepath, **setup)
    kgraph.build(knowledgebase)
    rag = SelfRAGEngine(filepath, knowledgebase, knowledge_graph=kgraph, **setup)

    patcher = BanditPatcher(filepath)

    data = {
        "index": [],
        "question": [],
        "response": [],
        "response_label": [],
        "contexts": [],
        "kg_consistency": [],
        "query_entailment": [],
        "response_entailment": [],
        "latency": [],
        "vram_usage": [],
        "gt_response": [],
        "gt_contexts": [],
        "action": [],
        "params": [],
        "failure_label": [],
        "EM": [],
    }
    query_idx = -1
    for row in tqdm.tqdm(dataset[0:200]):
        query_idx += 1
        gt_context, question, gt_answer = tuple(row)

        response_obj = rag.query(question, consistency_check=True, entailment_check=True)

        failure_label, new_label = patcher.get_failure_label(response_obj)
        response_obj["label"] = new_label
        print(response_obj)

        data["index"].append(query_idx)
        retrieved_context = [item["text"] for item in response_obj["retrieved_context"]]
        data["question"].append(question)
        data["response"].append(response_obj["response"])
        data["response_label"].append(response_obj["label"])
        data["contexts"].append(retrieved_context)
        data["kg_consistency"].append(response_obj["consistency_check"])
        data["query_entailment"].append(response_obj["entailment_check"]["query"])
        data["response_entailment"].append(response_obj["entailment_check"]["response"])
        data["latency"].append(response_obj["latency"])
        data["vram_usage"].append(response_obj["vram_usage"])
        data["gt_response"].append(gt_answer)
        data["gt_contexts"].append(gt_context)
        data["action"].append(-1)
        data["params"].append("")
        data["failure_label"].append(failure_label)
        data["EM"].append(em_match([gt_answer], response_obj["label"]))

    dataset = Dataset.from_dict(data)
    dataset = dataset.to_pandas()

    for c in ["kg_consistency", "query_entailment", "response_entailment", "failure_label"]:
        print(f"######## {c} (Correct) #######")
        print(dataset[dataset["EM"] == True][c].value_counts())
        print(f"######## {c} (Incorrect) #######")
        print(dataset[dataset["EM"] == False][c].value_counts())
    
    print(f"######## {c} (Correct) #######")
    print(dataset[dataset["EM"] == True]["gt_response"].value_counts())
    print(dataset[dataset["EM"] == True]["response_label"].value_counts())
    print(f"######## {c} (Incorrect) #######")
    print(dataset[dataset["EM"] == False]["gt_response"].value_counts())
    print(dataset[dataset["EM"] == False]["response_label"].value_counts())

    dataset.to_csv(f"{filepath}/bandit_eval_dataset.csv", sep="\t", index=False)