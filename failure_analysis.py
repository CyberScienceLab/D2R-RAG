import os
# prevent JAX from using gpu to avoid bm25s eat up the gpu memory
os.environ['JAX_PLATFORMS'] = 'cpu'

import random
import sys
from knowledge_graph import KnowledgeGraph
from utils import setup_settings, load_fever
from rag_engine import RAGEngine
import tqdm
from datasets import Dataset
from patcher import BanditPatcher 
from qa_metrics.em import em_match


if __name__ == "__main__":
    assert sys.argv[1] in ["fever"]
    if sys.argv[1] == "fever":
        filepath = "out"
        dataset, contexts = load_fever(split='validation')
        
    setup = setup_settings(sys.argv[1])

    kgraph = KnowledgeGraph(filepath, **setup)
    kgraph.build(contexts[0])
    kgraph.plot()
    rag = RAGEngine(filepath, contexts[1], knowledge_graph=kgraph, **setup)

    patcher = BanditPatcher(filepath, latency_budget=3, vram_budget=5000, method="linucb", alpha=2.)

    data = {
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
    for row in tqdm.tqdm(dataset):
        gt_context, question, gt_answer = tuple(row)

        params = {'retriever': 'dense', 'topk': 5, 'reranker': False, 'prompt_edit': "simple_qa", 'reindex': False}
        response_obj = rag.query(question, params=params, consistency_check=True, entailment_check=True)

        failure_label = patcher.get_failure_label(response_obj)

        data["question"].append(question)
        data["response"].append(response_obj["response"])
        data["response_label"].append(response_obj["label"])
        data["contexts"].append([item["text"] for item in response_obj["retrieved_context"]])
        data["kg_consistency"].append(response_obj["consistency_check"])
        data["query_entailment"].append(response_obj["entailment_check"]["query"])
        data["response_entailment"].append(response_obj["entailment_check"]["response"])
        data["latency"].append(response_obj["latency"])
        data["vram_usage"].append(response_obj["vram_usage"])
        data["gt_response"].append(gt_answer)
        data["gt_contexts"].append(gt_context)
        data["action"].append(-1)
        data["params"].append(str(params))
        data["failure_label"].append(failure_label)
        data["EM"].append(em_match([gt_answer], response_obj["label"]))

    dataset = Dataset.from_dict(data)
    dataset = dataset.to_pandas()

    for c in ["kg_consistency", "query_entailment", "response_entailment", "failure_label"]:
        print(f"######## {c} (Correct) #######")
        print(dataset[dataset["EM"] == True][c].value_counts())
        print(f"######## {c} (Incorrect) #######")
        print(dataset[dataset["EM"] == False][c].value_counts())
    
    dataset.to_csv(f"{filepath}/failure_statistics.csv", sep="\t", index=False)