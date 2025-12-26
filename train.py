import os
# prevent JAX from using gpu to avoid bm25s eat up the gpu memory
os.environ['JAX_PLATFORMS'] = 'cpu'

import random
import sys
from knowledge_graph import KnowledgeGraph
from utils import load_fever, setup_settings
from rag_engine import RAGEngine
import tqdm
from patcher import BanditPatcher 


if __name__ == "__main__":
    assert sys.argv[1] in ["fever"]
    if sys.argv[1] == "fever":
        filepath = "files_fever_t"
        dataset, knowledgebase = load_fever(split='train')
        
    setup = setup_settings(sys.argv[1])

    kgraph = KnowledgeGraph(filepath, **setup)
    kgraph.build(knowledgebase)
    rag = RAGEngine(filepath, knowledgebase, knowledge_graph=kgraph, **setup)

    patcher = BanditPatcher(filepath, latency_budget=3, vram_budget=6000, method="linucb", alpha=2.)

    EPOCHS = 2

    for epoch in range(EPOCHS):
        print("Epoch:", epoch+1)
        random.shuffle(dataset)
        for row in tqdm.tqdm(dataset):
            gt_context, question, gt_answer = tuple(row)

            params = {'retriever': 'dense', 'topk': 5, 'reranker': False, 'prompt_edit': "simple_qa", 'reindex': False}
            response_obj = rag.query(question, params=params, consistency_check=True, entailment_check=True)

            failure_label, new_label = patcher.get_failure_label(response_obj)
            response_obj["label"] = new_label
            reward = patcher.calculate_reward(failure_label, response_obj["consistency_check"], response_obj["entailment_check"]["response"], response_obj["latency"], response_obj["vram_usage"])

            if failure_label != "NO_FAILURE":
                print(gt_answer, response_obj)
                print(failure_label)

                context = patcher.get_context(question, failure_label, response_obj["consistency_check"], response_obj["entailment_check"]["query"], response_obj["entailment_check"]["response"], response_obj["latency"])
                patchset = "retriever" if response_obj["label"] == "UNVERIFIED" else "all"
                print("patchset:", patchset)
                action = patcher.predict(context, patchset=patchset)
                action_idx, params_updates = action
                print(params_updates)

                params.update(params_updates)
                response_obj_patched = rag.query(question, params=params, consistency_check=True, entailment_check=True)

                failure_label_patched, new_label_patched = patcher.get_failure_label(response_obj_patched)
                response_obj_patched["label"] = new_label_patched
                print(gt_answer, response_obj_patched)

                reward = patcher.calculate_reward(failure_label_patched, response_obj_patched["consistency_check"], response_obj_patched["entailment_check"]["response"], response_obj_patched["latency"], response_obj_patched["vram_usage"])
                patcher.update_bandit(context, action_idx, reward["total_reward"])

                print(failure_label_patched)
                print(reward)

            patcher.save_bandit()