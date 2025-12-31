import os
# prevent JAX from using gpu to avoid bm25s eat up the gpu memory
os.environ['JAX_PLATFORMS'] = 'cpu'

import sys 
from knowledge_graph import KnowledgeGraph
from utils import load_hotpotqa, setup_settings
from selfrag_engine import SelfRAGEngine
import tqdm
from datasets import Dataset
from patcher import BanditPatcher
from qa_metrics.em import em_match


if __name__ == "__main__":
    dataset_name = sys.argv[1]
    assert dataset_name in ["hotpotqa"]
    if dataset_name == "hotpotqa":
        filepath = "files_hotpotqa_selfrag_v"
        dataset, knowledgebase = load_hotpotqa(split='validation')
        
    setup = setup_settings(dataset_name)

    kgraph = KnowledgeGraph(filepath, **setup)
    kgraph.build(knowledgebase)
    rag = SelfRAGEngine(filepath, knowledgebase, knowledge_graph=kgraph, **setup)

    patcher = BanditPatcher(filepath)

    data = {
        "index": [],
        "question": [],
        "response": [],
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
    for row in tqdm.tqdm(dataset):
        query_idx += 1
        gt_context, question, gt_answer = tuple(row)

        response_obj = rag.query_shortanswer(question, consistency_check=True, entailment_check=True)

        failure_label, _ = patcher.get_failure_label(response_obj)
        
        data["index"].append(query_idx)
        retrieved_context = [item["text"] for item in response_obj["retrieved_context"]]
        data["question"].append(question)
        data["response"].append(response_obj["response"])
        data["contexts"].append(retrieved_context)
        data["kg_consistency"].append(response_obj["consistency_check"])
        data["query_entailment"].append(response_obj["entailment_check"]["query"])
        data["response_entailment"].append(response_obj["entailment_check"]["response"])
        data["latency"].append(response_obj["latency"])
        data["vram_usage"].append(response_obj["vram_usage"])
        data["gt_response"].append(gt_answer[0])
        data["gt_contexts"].append(gt_context)
        data["action"].append(-1)
        data["params"].append("")
        data["failure_label"].append(failure_label)
        data["EM"].append(em_match(list(map(lambda item: item.lower(), gt_answer)), response_obj["response"].lower()))

    dataset = Dataset.from_dict(data)
    dataset = dataset.to_pandas()

    for c in ["kg_consistency", "query_entailment", "response_entailment", "failure_label"]:
        print(f"######## {c} (Correct) #######")
        print(dataset[dataset["EM"] == True][c].value_counts())
        print(f"######## {c} (Incorrect) #######")
        print(dataset[dataset["EM"] == False][c].value_counts())
    
    dataset.to_csv(f"{filepath}/bandit_eval_dataset.csv", sep="\t", index=False)