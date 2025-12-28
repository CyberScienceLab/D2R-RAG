import os
# prevent JAX from using gpu to avoid bm25s eat up the gpu memory
os.environ['JAX_PLATFORMS'] = 'cpu'

import random
import sys
from knowledge_graph import KnowledgeGraph
from utils import load_hotpotqa, setup_settings
from rag_engine import RAGEngine
import tqdm
from patcher import BanditPatcher 


if __name__ == "__main__":
    dataset_name = sys.argv[1]
    assert dataset_name in ["hotpotqa", "hotpotqa_ts"]
    if dataset_name == "hotpotqa":
        method = "linucb"
        filepath = "files_hotpotqa_t"
        dataset, knowledgebase = load_hotpotqa(split='train')
    elif dataset_name == "hotpotqa_ts":
        dataset_name = "hotpotqa"
        method = "thompsonsampling"
        filepath = "files_hotpotqa_ts_t"
        dataset, knowledgebase = load_hotpotqa(split='train')
        
    setup = setup_settings(dataset_name)

    kgraph = KnowledgeGraph(filepath, **setup)
    kgraph.build(knowledgebase)
    rag = RAGEngine(filepath, knowledgebase, knowledge_graph=kgraph, **setup)

    patcher = BanditPatcher(filepath, latency_budget=3, vram_budget=6000, method=method, alpha=2.)

    EPOCHS = 2

    for epoch in range(EPOCHS):
        print("Epoch:", epoch+1)
        random.shuffle(dataset)
        for row in tqdm.tqdm(dataset):
            gt_context, question, gt_answer = tuple(row)

            params = {'retriever': 'dense', 'topk': 5, 'reranker': False, 'prompt_edit': "simple_qa", 'reindex': False}
            response_obj = rag.query_shortanswer(question, params=params, consistency_check=True, entailment_check=True)

            failure_label, _ = patcher.get_failure_label(response_obj)
            reward = patcher.calculate_reward(failure_label, response_obj["consistency_check"], response_obj["entailment_check"]["response"], response_obj["latency"], response_obj["vram_usage"])

            if failure_label != "NO_FAILURE":
                print(gt_answer, response_obj)
                print(failure_label)

                context = patcher.get_context(question, failure_label, response_obj["consistency_check"], response_obj["entailment_check"]["query"], response_obj["entailment_check"]["response"], response_obj["latency"])
                action = patcher.predict(context)
                action_idx, params_updates = action
                print(params_updates)

                params.update(params_updates)
                response_obj_patched = rag.query_shortanswer(question, params=params, consistency_check=True, entailment_check=True)

                failure_label_patched, _ = patcher.get_failure_label(response_obj_patched)
                print(gt_answer, response_obj_patched)

                reward = patcher.calculate_reward(failure_label_patched, response_obj_patched["consistency_check"], response_obj_patched["entailment_check"]["response"], response_obj_patched["latency"], response_obj_patched["vram_usage"])
                patcher.update_bandit(context, action_idx, reward["total_reward"])

                print(failure_label_patched)
                print(reward)

            patcher.save_bandit()