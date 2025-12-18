import os
# prevent JAX from using gpu to avoid bm25s eat up the gpu memory
os.environ['JAX_PLATFORMS'] = 'cpu'

import sys 
from knowledge_graph import KnowledgeGraph
from utils import load_squad, setup_settings
from rag_engine import RAGEngine
import tqdm
from datasets import Dataset
from patcher import BanditPatcher


if __name__ == "__main__":
    assert sys.argv[1] in ["squad"]
    if sys.argv[1] == "squad":
        filepath = "out"
        dataset, contexts = load_squad(split='validation')
    elif sys.argv[1] == "hotpotqa":
        filepath = "out2"
        # dataset = ...
        
    setup = setup_settings(sys.argv[1])

    kgraph = KnowledgeGraph(filepath, **setup)
    kgraph.build(contexts)
    kgraph.plot()
    rag = RAGEngine(filepath, contexts, knowledge_graph=kgraph, **setup)

    patcher = BanditPatcher(filepath, latency_budget=3, vram_budget=14000, method="linucb", alpha=2.)
    patcher.load_bandit()

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
        "bandit_reward": [],
    }
    query_idx = -1
    for row in tqdm.tqdm(dataset):
        query_idx += 1
        gt_context, question, gt_answer = tuple(row)

        params = {'retriever': 'dense', 'topk': 5, 'reranker': False, 'prompt_edit': "simple_qa", 'reindex': False}
        response_obj = rag.query(question, params=params, consistency_check=True, entailment_check=True)

        failure_label = patcher.get_failure_label(response_obj)
        reward = patcher.calculate_reward(failure_label, response_obj["consistency_check"], response_obj["entailment_check"]["response"], response_obj["latency"], response_obj["vram_usage"])

        data["index"].append(query_idx)
        data["question"].append(question)
        data["response"].append(response_obj["response"])
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
        data["bandit_reward"].append(str(reward))

        if failure_label != "NO_FAILURE":
            print(response_obj)
            print(failure_label)

            context = patcher.get_context(question, len(response_obj["retrieved_context"]), failure_label, response_obj["consistency_check"], response_obj["entailment_check"]["query"], response_obj["entailment_check"]["response"], response_obj["latency"])
            action = patcher.predict(context)
            action_idx, params_updates = action
            print(params_updates)

            params.update(params_updates)
            response_obj_patched = rag.query(question, params=params, consistency_check=True, entailment_check=True)
            print(response_obj_patched)

            failure_label_patched = patcher.get_failure_label(response_obj_patched)
            reward = patcher.calculate_reward(failure_label_patched, response_obj_patched["consistency_check"], response_obj_patched["entailment_check"]["response"], response_obj_patched["latency"], response_obj_patched["vram_usage"])

            data["index"].append(query_idx)
            data["question"].append(question)
            data["response"].append(response_obj_patched["response"])
            data["contexts"].append([item["text"] for item in response_obj_patched["retrieved_context"]])
            data["kg_consistency"].append(response_obj_patched["consistency_check"])
            data["query_entailment"].append(response_obj_patched["entailment_check"]["query"])
            data["response_entailment"].append(response_obj_patched["entailment_check"]["response"])
            data["latency"].append(response_obj_patched["latency"])
            data["vram_usage"].append(response_obj_patched["vram_usage"])
            data["gt_response"].append(gt_answer)
            data["gt_contexts"].append(gt_context)
            data["action"].append(action_idx)
            data["params"].append(str(params_updates))
            data["failure_label"].append(failure_label_patched)
            data["bandit_reward"].append(str(reward))

            print(failure_label_patched)
            print(reward)

    dataset = Dataset.from_dict(data)
    dataset.to_pandas().to_csv("out/bandit_eval_dataset.csv", sep="\t", index=False)