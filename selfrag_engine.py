import torch
import json
from rag_engine import RAGEngine
from triplet_extractor import TripletExtractor
from json_repair import repair_json
from entailment import EntailmentChecker
import time
from self_rag_pack.llama_index.packs.self_rag.base import SelfRAGQueryEngine


class SelfRAGEngine(RAGEngine):

    def __init__(self, output_path, knowledgebase, knowledge_graph=None, **kwargs):
        self.output_path = output_path
        self.kwargs = kwargs
        self.knowledge_graph = knowledge_graph
        self.knowledgebase = knowledgebase
        self.knowledgebase.reset()

        self.build_nodes()

        self.first_time_dense_index = True
        self.first_time_bm25_index = True
        retriever = self.build_dense_retriever(5)
        self.query_engine = self.build_query_engine(retriever)

        self.triplet_extractor = TripletExtractor(**kwargs)
        self.entailment_checker = EntailmentChecker(**kwargs)

    def build_query_engine(self, retriever, similarity_posprocess=False, reranker=False, prompt_edit=False):
        if self.kwargs["device"] == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            self.start_mem = torch.cuda.memory_allocated()
        else:
            self.start_mem = 0.

        model_path = "selfrag_llama2_7b.q4_k_m.gguf"
        query_engine = SelfRAGQueryEngine(str(model_path), retriever, verbose=False, n_gpu_layers=-1)
        print(f"[MSG] SelfRAG query engine is ready to go.")

        return query_engine
    
    def query(self, query_str, params={}, consistency_check=False, entailment_check=False):
        response_obj = {}

        tic = time.perf_counter()

        query = """You are a fact checking assistant.
        Your task is to determine whether the context supports, refutes, or does not provide enough information about the claim.

        Claim:
        """ + query_str + """

        Output strictly in valid JSON format only, with no extra text, explanations, or comments.

        Output format:
        {
        "label": "SUPPORTS" | "REFUTES" | "NOTENOUGHINFO",
        "response": "short, direct evidence extracted or paraphrased from context"
        }"""

        response_object = self.query_engine.query(query)
        response = response_object.response.strip()
        if response == "Empty Response":
            # In case no node found by the retriever:
            # https://github.com/run-llama/llama_index/blob/fe72a2f5dbefb92d8c91cb460d4299de5637aa5a/llama-index-core/llama_index/core/response_synthesizers/base.py#L284

            response = {"label": "NOTENOUGHINFO", "response": "Context is not sufficient."}
        else:
            response = json.loads(repair_json(response))

        response_obj.update(response)

        latency = time.perf_counter() - tic # seconds
        if self.kwargs["device"] == "cuda":
            torch.cuda.synchronize()
            peak_after = torch.cuda.max_memory_allocated()
            vram_usage = (peak_after - self.start_mem) / 1048576 # MB, 1024 * 1024
        else:
            vram_usage = 0.

        response_obj["query"] = query_str
        response_obj["retrieved_context"] = [{"text": n.node.text.split("<paragraph>")[-1].split("</paragraph>")[0], 'node_id': n.id_, 'score': n.score} for n in response_object.source_nodes]

        if self.knowledge_graph and consistency_check:
            response_obj["consistency_check"] = self.knowledge_graph.consistency_check(response_obj["response"])

        if entailment_check:
            response_obj["entailment_check"] = self.entailment_checker.check(query_str, response_obj["response"], response_obj["retrieved_context"])

        response_obj["latency"] = latency
        response_obj["vram_usage"] = vram_usage

        return response_obj
    
    def query_shortanswer(self, query_str, params={}, consistency_check=False, entailment_check=False):
        response_obj = {}

        tic = time.perf_counter()

        query = """You are a helpful assistant.

        Answer the following Question based on the above Context only. Only answer from the Context. If the answer does not exist in the context, respond with 'The provided context is insufficient regarding ...'.
        For yes/no questions, you must repond with a long answer containing "yes" or "no". 
        For questions with "who", "what", "which", "how", "where", and "when", you should repond with a long answer. 

        Question: 
        """ + query_str

        response_object = self.query_engine.query(query)
        response = response_object.response.strip()
        if response == "Empty Response":
            # In case no node found by the retriever:
            # https://github.com/run-llama/llama_index/blob/fe72a2f5dbefb92d8c91cb460d4299de5637aa5a/llama-index-core/llama_index/core/response_synthesizers/base.py#L284

            response = {"response": "The provided context is insufficient."}
        else:
            response = {"response": response.replace("Response 1: ", "")}

        response_obj.update(response)

        latency = time.perf_counter() - tic # seconds
        if self.kwargs["device"] == "cuda":
            torch.cuda.synchronize()
            peak_after = torch.cuda.max_memory_allocated()
            vram_usage = (peak_after - self.start_mem) / 1048576 # MB, 1024 * 1024
        else:
            vram_usage = 0.

        response_obj["query"] = query_str
        response_obj["retrieved_context"] = [{"text": n.node.text.split("<paragraph>")[-1].split("</paragraph>")[0], 'node_id': n.id_, 'score': n.score} for n in response_object.source_nodes]

        if self.knowledge_graph and consistency_check:
            response_obj["consistency_check"] = self.knowledge_graph.consistency_check(response_obj["response"])

        if entailment_check:
            response_obj["entailment_check"] = self.entailment_checker.check(query_str, response_obj["response"], response_obj["retrieved_context"])

        response_obj["latency"] = latency
        response_obj["vram_usage"] = vram_usage

        return response_obj
    
