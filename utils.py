import os

import yaml
os.environ['OPENAI_API_KEY'] = "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"
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import Settings
import torch
from llama_index.llms.openai import OpenAI
from datasets import load_dataset
import json
import csv
import random
random.seed(42)
import pickle
import tqdm


def setup_settings(dataset):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2", device=device)

    with open("prompts.yaml", 'r') as f:
        prompts = yaml.load(f, Loader=yaml.FullLoader)
    
    # llm_model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    # llm_model = HuggingFaceLLM(
    #     model_name=llm_model_name,
    #     tokenizer_name=llm_model_name,
    #     context_window=8192,
    #     is_chat_model=True,
    #     generate_kwargs={"do_sample": False},
    #     # model_kwargs={"load_in_4bit": True, "dtype": torch.bfloat16},
    #     model_kwargs={"dtype": torch.bfloat16},
    #     max_new_tokens=256,
    # )
    llm_model = OpenAI(model="gpt-4o-mini")
    
    Settings.llm = llm_model
    Settings.embed_model = embedding_model

    similarity_top_k = 3
    reranker_top_n = 3
    similarity_cutoff = 0.4
    rag_storage_dir = "storage_rag" if dataset == "fever" else "storage_rag2"
    kg_storage_dir = "storage_kg" if dataset == "fever" else "storage_kg2"
    dense_retriever_storage = "dense_storage" if dataset == "fever" else "dense_storage2"
    bm25_retriever_storage = "bm25_storage" if dataset == "fever" else "bm25_storage2"

    return {
        "device": device, 
        "prompts": prompts,
        # "prompt_rag_qa": rag_qa_prompt, 
        # "prompt_WP": prompt_WP,
        # "prompt_WR": prompt_WR,
        "similarity_top_k": similarity_top_k, 
        "reranker_top_n": reranker_top_n,
        "similarity_cutoff": similarity_cutoff,
        "rag_storage_dir": rag_storage_dir,
        "kg_storage_dir": kg_storage_dir,
        # "KG_completion_prompt": knowledge_graph_completion_prompt,
        "dense_retriever_storage": dense_retriever_storage,
        "bm25_retriever_storage": bm25_retriever_storage}

def load_squad(split='train'):
    dataset = load_dataset('squad', split=split)

    dataset_pd = dataset.to_pandas()
    dataset_pd = dataset_pd[['context', 'question', 'answers']]
    dataset_pd['answers'] = dataset_pd['answers'].apply(lambda val: val['text']) #" and ".join(val['text']))
    dataset_pd['context'] = dataset_pd['context'].apply(lambda val: val.strip())

    contexts = dataset_pd.iloc[:10000]['context'].unique().tolist() 
    print("Num unique contexts:", len(contexts), len(contexts))

    dataset_pd = dataset_pd.iloc[:1000]
    dataset = dataset_pd.values.tolist()
    dataset = list(map(lambda item: (item[0], item[1], item[2].tolist()), dataset))

    return dataset, (contexts, contexts)

def load_triviaqa(split='train'):
    dataset = load_dataset('mandarjoshi/trivia_qa', 'rc', split=f"{split}[:1000]")

    contexts_fine = list(map(lambda item: item['entity_pages']['wiki_context'], dataset))
    contexts_fine = contexts_fine[:500]
    contexts_fine = list(set([item for ctx in contexts_fine for item in ctx]))
    contexts_gran = [passage for ctx in contexts_fine for passage in ctx.split("\n\n")]
    print("Num unique contexts:", len(contexts_fine), len(contexts_gran))

    dataset = [([], q_item, [a_item["value"]]) for q_item, a_item in zip(dataset['question'], dataset['answer'])]

    return dataset, (contexts_fine, contexts_gran)

def load_nq(split='train'):
    with open("./psgs_w100.tsv") as input_file:
        contexts = []
        tr = csv.reader(input_file, delimiter='\t')
        next(tr)
        for line in tr:
            paragraph_text = line[1]
            # title = line[2]
            contexts.append(paragraph_text)

            if len(contexts) >= 10000:
                break

    with open(f"./biencoder-nq-{split}.json", "r") as file:
        instance = json.load(file)

    # contexts = list(set([ctx["text"] for inst in instance for ctx in inst["positive_ctxs"]]))
    # contexts = random.choices(contexts, k=10000)
    print("Num unique contexts:", len(contexts), len(contexts))

    instance = instance[:1000]
    dataset = list(map(lambda item: ([], item["question"]+"?", item["answers"]), instance))

    return dataset, (contexts, contexts)

def load_fever(split='train'):
    with open("./out/knowledge_base.pkl", 'rb') as f:
        content = pickle.load(f)
        meta_data = content["meta_data"]
        evidences = content["sentences"]
        evidence_dict = {j["doc_id"]:i for i, j in zip(evidences, meta_data)}

    contexts = list(set([sentence.replace("\t", " ") for evidence in evidences for sentence in evidence if len(sentence) > 0]))
    print("Num unique contexts:", len(contexts), len(contexts))

    with open("./shared_task_dev.jsonl", 'r') as json_file:
        if split == "train":
            json_list = list(json_file)[:1000]
        else:
            json_list = list(json_file)[1000:2000]

        fever_dataset = []
        for json_str in tqdm.tqdm(json_list):
            result = json.loads(json_str)
            evidence_sentences = []
            for evidence in result['evidence']: 
                if evidence[0][2] is not None: 
                    evidence_sentences.append(evidence_dict[evidence[0][2]][1:][evidence[0][3]].replace("\t", " "))
            
            evidence_sentences = list(set(evidence_sentences))
            fever_dataset.append((evidence_sentences, result['claim'], result['label'].replace(' ', '')))

    print("fever_dataset size:", len(fever_dataset))

    return fever_dataset, (contexts, contexts)

def singleton(cls, *args, **kwargs):
    instances = {}

    def _singleton(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)

        return instances[cls]

    return _singleton