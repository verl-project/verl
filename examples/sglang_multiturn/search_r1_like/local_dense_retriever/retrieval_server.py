"""
# Put all caches on project disk

should use the conda env conda activate vllm07 
CUDA_VISIBLE_DEVICES=1,2
export XDG_CACHE_HOME=/ocean/projects/med230010p/yji3/.cache
export HF_HOME=/ocean/projects/med230010p/yji3/.cache/huggingface
export HF_DATASETS_CACHE=/ocean/projects/med230010p/yji3/.cache/huggingface/datasets
export TRANSFORMERS_CACHE=/ocean/projects/med230010p/yji3/.cache/huggingface/transformers
export HF_HUB_CACHE=/ocean/projects/med230010p/yji3/.cache/huggingface/hub

# Also move temp files (datasets multiprocessing can write a lot of temp)
export TMPDIR=/ocean/projects/med230010p/yji3/.tmp
mkdir -p "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" "$HF_HUB_CACHE" "$TMPDIR"

python examples/sglang_multiturn/search_r1_like/local_dense_retriever/retrieval_server.py \
  --index_path /ocean/projects/med230010p/yji3/data/retrieval/e5_Flat.index \
  --corpus_path /ocean/projects/med230010p/yji3/data/retrieval/data00/jiajie_jin/flashrag_indexes/wiki_dpr_100w/wiki_dump.jsonl \
  --topk 3 \
  --retriever_name e5 \
  --retriever_model intfloat/e5-base-v2 \
  --faiss_gpu \
  --port 8003


 
"""
import argparse
import json
import warnings
from typing import Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor

import faiss
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer

# ============ 优化 1: 预加载 corpus 到内存 ============
import gzip
import json

def load_corpus_to_memory(corpus_path: str):
    """加载 corpus 到内存，支持多种格式"""
    print(f"Loading corpus from {corpus_path}...")
    corpus = []
    
    # 检测文件类型
    with open(corpus_path, 'rb') as f:
        header = f.read(2)
    
    # Gzip 文件 (以 0x1f 0x8b 开头)
    if header == b'\x1f\x8b':
        print("Detected gzip compressed file")
        with gzip.open(corpus_path, 'rt', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    corpus.append(json.loads(line))
    # 普通 JSONL
    else:
        # 尝试多种编码
        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            try:
                with open(corpus_path, 'r', encoding=encoding) as f:
                    for line in f:
                        if line.strip():
                            corpus.append(json.loads(line))
                print(f"Successfully loaded with encoding: {encoding}")
                break
            except (UnicodeDecodeError, json.JSONDecodeError) as e:
                print(f"Failed with {encoding}: {e}")
                corpus = []
                continue
    
    print(f"Loaded {len(corpus)} documents into memory")
    return corpus

def load_model(model_path: str, use_fp16: bool = True):
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    model.cuda()
    if use_fp16:
        model = model.half()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    return model, tokenizer


def pooling(pooler_output, last_hidden_state, attention_mask=None, pooling_method="mean"):
    if pooling_method == "mean":
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    elif pooling_method == "cls":
        return last_hidden_state[:, 0]
    elif pooling_method == "pooler":
        return pooler_output
    else:
        raise NotImplementedError


class FastEncoder:
    """优化的 Encoder - 移除不必要的 cache 清理"""
    
    def __init__(self, model_name, model_path, pooling_method, max_length, use_fp16):
        self.model_name = model_name
        self.pooling_method = pooling_method
        self.max_length = max_length
        self.model, self.tokenizer = load_model(model_path, use_fp16)
        
        # ============ 优化 2: 预热模型 ============
        print("Warming up encoder...")
        self.encode(["warmup query"] * 10)
        print("Encoder ready!")

    @torch.no_grad()
    def encode(self, query_list: list[str], is_query=True) -> np.ndarray:
        if isinstance(query_list, str):
            query_list = [query_list]

        # 添加前缀
        if "e5" in self.model_name.lower():
            prefix = "query: " if is_query else "passage: "
            query_list = [f"{prefix}{q}" for q in query_list]
        elif "bge" in self.model_name.lower() and is_query:
            query_list = [f"Represent this sentence for searching relevant passages: {q}" for q in query_list]

        inputs = self.tokenizer(
            query_list, 
            max_length=self.max_length, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        )
        inputs = {k: v.cuda() for k, v in inputs.items()}

        output = self.model(**inputs, return_dict=True)
        query_emb = pooling(
            output.pooler_output, 
            output.last_hidden_state, 
            inputs["attention_mask"], 
            self.pooling_method
        )
        query_emb = torch.nn.functional.normalize(query_emb, dim=-1)
        
        # ============ 优化 3: 移除 torch.cuda.empty_cache() ============
        # 这个操作非常慢，只在必要时才调用
        return query_emb.cpu().numpy().astype(np.float32)


class FastDenseRetriever:
    """优化的 Dense Retriever"""
    
    def __init__(self, config):
        self.topk = config.retrieval_topk
        
        # ============ 优化 4: 预加载 corpus 到内存 ============
        self.corpus = load_corpus_to_memory(config.corpus_path)
        
        # 加载 FAISS index
        print(f"Loading FAISS index from {config.index_path}...")
        self.index = faiss.read_index(config.index_path)
        
        # ============ 优化 5: GPU FAISS ============
        if config.faiss_gpu and faiss.get_num_gpus() > 0:
            print("Moving FAISS index to GPU...")
            co = faiss.GpuMultipleClonerOptions()
            co.useFloat16 = True
            co.shard = True
            self.index = faiss.index_cpu_to_all_gpus(self.index, co=co)
        
        # 初始化 encoder
        self.encoder = FastEncoder(
            model_name=config.retrieval_method,
            model_path=config.retrieval_model_path,
            pooling_method=config.retrieval_pooling_method,
            max_length=config.retrieval_query_max_length,
            use_fp16=config.retrieval_use_fp16,
        )
        print("Retriever ready!")

    def _load_docs_fast(self, doc_idxs):
        """优化的文档加载 - 直接从内存 list 读取"""
        return [self.corpus[int(idx)] for idx in doc_idxs]

    def search(self, query: str, num: int = None, return_score: bool = False):
        if num is None:
            num = self.topk
        query_emb = self.encoder.encode([query])
        scores, idxs = self.index.search(query_emb, k=num)
        results = self._load_docs_fast(idxs[0])
        if return_score:
            return results, scores[0].tolist()
        return results

    def batch_search(self, query_list: list[str], num: int = None, return_score: bool = False):
        if isinstance(query_list, str):
            query_list = [query_list]
        if num is None:
            num = self.topk

        # ============ 优化 6: 一次性 encode 所有 queries ============
        all_emb = self.encoder.encode(query_list)
        all_scores, all_idxs = self.index.search(all_emb, k=num)
        
        results = []
        scores = []
        for i in range(len(query_list)):
            docs = self._load_docs_fast(all_idxs[i])
            results.append(docs)
            scores.append(all_scores[i].tolist())

        if return_score:
            return results, scores
        return results


# ============ FastAPI 服务 ============

class QueryRequest(BaseModel):
    queries: list[str]
    topk: Optional[int] = None
    return_scores: bool = False


app = FastAPI()

# ============ 优化 7: 使用线程池处理请求 ============
executor = ThreadPoolExecutor(max_workers=4)


@app.post("/retrieve")
async def retrieve_endpoint(request: QueryRequest):
    topk = request.topk or retriever.topk
    
    # 在线程池中执行（避免阻塞事件循环）
    loop = asyncio.get_event_loop()
    results, scores = await loop.run_in_executor(
        executor,
        lambda: retriever.batch_search(
            query_list=request.queries, 
            num=topk, 
            return_score=True
        )
    )

    resp = []
    for i, single_result in enumerate(results):
        if request.return_scores:
            combined = [{"document": doc, "score": score} 
                       for doc, score in zip(single_result, scores[i])]
            resp.append(combined)
        else:
            resp.append(single_result)
    return {"result": resp}


@app.get("/health")
async def health():
    return {"status": "ok", "corpus_size": len(retriever.corpus)}


class Config:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_path", type=str, required=True)
    parser.add_argument("--corpus_path", type=str, required=True)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--retriever_name", type=str, default="e5")
    parser.add_argument("--retriever_model", type=str, default="intfloat/e5-base-v2")
    parser.add_argument("--faiss_gpu", action="store_true")
    parser.add_argument("--port", type=int, default=8003)
    parser.add_argument("--workers", type=int, default=1)  # uvicorn workers
    args = parser.parse_args()

    config = Config(
        retrieval_method=args.retriever_name,
        index_path=args.index_path,
        corpus_path=args.corpus_path,
        retrieval_topk=args.topk,
        faiss_gpu=args.faiss_gpu,
        retrieval_model_path=args.retriever_model,
        retrieval_pooling_method="mean",
        retrieval_query_max_length=256,
        retrieval_use_fp16=True,
    )

    retriever = FastDenseRetriever(config)

    # ============ 优化 8: 多 worker 启动 ============
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=args.port,
        workers=args.workers,  # 可以设置多个 worker
        log_level="warning"    # 减少日志开销
    )
# Copyright 2024 Bytedance Ltd. and/or its affiliates
# # Copyright 2023-2024 SGLang Team
# # Copyright 2025 Search-R1 Contributors
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# # Adapted from https://github.com/PeterGriffinJin/Search-R1/blob/main/search_r1/search/retrieval_server.py

# """
# # Put all caches on project disk
# export XDG_CACHE_HOME=/ocean/projects/med230010p/yji3/.cache
# export HF_HOME=/ocean/projects/med230010p/yji3/.cache/huggingface
# export HF_DATASETS_CACHE=/ocean/projects/med230010p/yji3/.cache/huggingface/datasets
# export TRANSFORMERS_CACHE=/ocean/projects/med230010p/yji3/.cache/huggingface/transformers
# export HF_HUB_CACHE=/ocean/projects/med230010p/yji3/.cache/huggingface/hub

# # Also move temp files (datasets multiprocessing can write a lot of temp)
# export TMPDIR=/ocean/projects/med230010p/yji3/.tmp
# mkdir -p "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" "$HF_HUB_CACHE" "$TMPDIR"

#  python examples/sglang_multiturn/search_r1_like/local_dense_retriever/retrieval_server.py   --index_path /ocean/projects/med230010p/yji3/data/retrieval/e5_Flat.index --corpus_path /ocean/projects/med230010p/yji3/data/retrieval/wiki-18.jsonl --topk 3   --retriever_name e5   --retriever_model intfloat/e5-base-v2
# """

# import argparse
# import json
# import warnings
# from typing import Optional

# import datasets
# import faiss
# import numpy as np
# import torch
# import uvicorn
# from fastapi import FastAPI
# from pydantic import BaseModel
# from tqdm import tqdm
# from transformers import AutoModel, AutoTokenizer


# def load_corpus(corpus_path: str):
#     corpus = datasets.load_dataset("json", data_files=corpus_path, split="train", num_proc=4)
#     return corpus


# def load_docs(corpus, doc_idxs):
#     results = [corpus[int(idx)] for idx in doc_idxs]
#     return results


# def load_model(model_path: str, use_fp16: bool = False):
#     model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
#     model.eval()
#     model.cuda()
#     if use_fp16:
#         model = model.half()
#     tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
#     return model, tokenizer


# def pooling(pooler_output, last_hidden_state, attention_mask=None, pooling_method="mean"):
#     if pooling_method == "mean":
#         last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
#         return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
#     elif pooling_method == "cls":
#         return last_hidden_state[:, 0]
#     elif pooling_method == "pooler":
#         return pooler_output
#     else:
#         raise NotImplementedError("Pooling method not implemented!")


# class Encoder:
#     def __init__(self, model_name, model_path, pooling_method, max_length, use_fp16):
#         self.model_name = model_name
#         self.model_path = model_path
#         self.pooling_method = pooling_method
#         self.max_length = max_length
#         self.use_fp16 = use_fp16

#         self.model, self.tokenizer = load_model(model_path=model_path, use_fp16=use_fp16)
#         self.model.eval()

#     @torch.no_grad()
#     def encode(self, query_list: list[str], is_query=True) -> np.ndarray:
#         # processing query for different encoders
#         if isinstance(query_list, str):
#             query_list = [query_list]

#         if "e5" in self.model_name.lower():
#             if is_query:
#                 query_list = [f"query: {query}" for query in query_list]
#             else:
#                 query_list = [f"passage: {query}" for query in query_list]

#         if "bge" in self.model_name.lower():
#             if is_query:
#                 query_list = [
#                     f"Represent this sentence for searching relevant passages: {query}" for query in query_list
#                 ]

#         inputs = self.tokenizer(
#             query_list, max_length=self.max_length, padding=True, truncation=True, return_tensors="pt"
#         )
#         inputs = {k: v.cuda() for k, v in inputs.items()}

#         if "T5" in type(self.model).__name__:
#             # T5-based retrieval model
#             decoder_input_ids = torch.zeros((inputs["input_ids"].shape[0], 1), dtype=torch.long).to(
#                 inputs["input_ids"].device
#             )
#             output = self.model(**inputs, decoder_input_ids=decoder_input_ids, return_dict=True)
#             query_emb = output.last_hidden_state[:, 0, :]
#         else:
#             output = self.model(**inputs, return_dict=True)
#             query_emb = pooling(
#                 output.pooler_output, output.last_hidden_state, inputs["attention_mask"], self.pooling_method
#             )
#             if "dpr" not in self.model_name.lower():
#                 query_emb = torch.nn.functional.normalize(query_emb, dim=-1)

#         query_emb = query_emb.detach().cpu().numpy()
#         query_emb = query_emb.astype(np.float32, order="C")

#         del inputs, output
#         torch.cuda.empty_cache()

#         return query_emb


# class BaseRetriever:
#     def __init__(self, config):
#         self.config = config
#         self.retrieval_method = config.retrieval_method
#         self.topk = config.retrieval_topk

#         self.index_path = config.index_path
#         self.corpus_path = config.corpus_path

#     def _search(self, query: str, num: int, return_score: bool):
#         raise NotImplementedError

#     def _batch_search(self, query_list: list[str], num: int, return_score: bool):
#         raise NotImplementedError

#     def search(self, query: str, num: int = None, return_score: bool = False):
#         return self._search(query, num, return_score)

#     def batch_search(self, query_list: list[str], num: int = None, return_score: bool = False):
#         return self._batch_search(query_list, num, return_score)


# class BM25Retriever(BaseRetriever):
#     def __init__(self, config):
#         super().__init__(config)
#         from pyserini.search.lucene import LuceneSearcher

#         self.searcher = LuceneSearcher(self.index_path)
#         self.contain_doc = self._check_contain_doc()
#         if not self.contain_doc:
#             self.corpus = load_corpus(self.corpus_path)
#         self.max_process_num = 8

#     def _check_contain_doc(self):
#         return self.searcher.doc(0).raw() is not None

#     def _search(self, query: str, num: int = None, return_score: bool = False):
#         if num is None:
#             num = self.topk
#         hits = self.searcher.search(query, num)
#         if len(hits) < 1:
#             if return_score:
#                 return [], []
#             else:
#                 return []
#         scores = [hit.score for hit in hits]
#         if len(hits) < num:
#             warnings.warn("Not enough documents retrieved!", stacklevel=2)
#         else:
#             hits = hits[:num]

#         if self.contain_doc:
#             all_contents = [json.loads(self.searcher.doc(hit.docid).raw())["contents"] for hit in hits]
#             results = [
#                 {
#                     "title": content.split("\n")[0].strip('"'),
#                     "text": "\n".join(content.split("\n")[1:]),
#                     "contents": content,
#                 }
#                 for content in all_contents
#             ]
#         else:
#             results = load_docs(self.corpus, [hit.docid for hit in hits])

#         if return_score:
#             return results, scores
#         else:
#             return results

#     def _batch_search(self, query_list: list[str], num: int = None, return_score: bool = False):
#         results = []
#         scores = []
#         for query in query_list:
#             item_result, item_score = self._search(query, num, True)
#             results.append(item_result)
#             scores.append(item_score)
#         if return_score:
#             return results, scores
#         else:
#             return results


# class DenseRetriever(BaseRetriever):
#     def __init__(self, config):
#         super().__init__(config)
#         self.index = faiss.read_index(self.index_path)
#         if config.faiss_gpu:
#             co = faiss.GpuMultipleClonerOptions()
#             co.useFloat16 = True
#             co.shard = True
#             self.index = faiss.index_cpu_to_all_gpus(self.index, co=co)

#         self.corpus = load_corpus(self.corpus_path)
#         self.encoder = Encoder(
#             model_name=self.retrieval_method,
#             model_path=config.retrieval_model_path,
#             pooling_method=config.retrieval_pooling_method,
#             max_length=config.retrieval_query_max_length,
#             use_fp16=config.retrieval_use_fp16,
#         )
#         self.topk = config.retrieval_topk
#         self.batch_size = config.retrieval_batch_size

#     def _search(self, query: str, num: int = None, return_score: bool = False):
#         if num is None:
#             num = self.topk
#         query_emb = self.encoder.encode(query)
#         scores, idxs = self.index.search(query_emb, k=num)
#         idxs = idxs[0]
#         scores = scores[0]
#         results = load_docs(self.corpus, idxs)
#         if return_score:
#             return results, scores.tolist()
#         else:
#             return results

#     def _batch_search(self, query_list: list[str], num: int = None, return_score: bool = False):
#         if isinstance(query_list, str):
#             query_list = [query_list]
#         if num is None:
#             num = self.topk

#         results = []
#         scores = []
#         for start_idx in tqdm(range(0, len(query_list), self.batch_size), desc="Retrieval process: "):
#             query_batch = query_list[start_idx : start_idx + self.batch_size]
#             batch_emb = self.encoder.encode(query_batch)
#             batch_scores, batch_idxs = self.index.search(batch_emb, k=num)
#             batch_scores = batch_scores.tolist()
#             batch_idxs = batch_idxs.tolist()

#             # load_docs is not vectorized, but is a python list approach
#             flat_idxs = sum(batch_idxs, [])
#             batch_results = load_docs(self.corpus, flat_idxs)
#             # chunk them back
#             batch_results = [batch_results[i * num : (i + 1) * num] for i in range(len(batch_idxs))]

#             results.extend(batch_results)
#             scores.extend(batch_scores)

#             del batch_emb, batch_scores, batch_idxs, query_batch, flat_idxs, batch_results
#             torch.cuda.empty_cache()

#         if return_score:
#             return results, scores
#         else:
#             return results


# def get_retriever(config):
#     if config.retrieval_method == "bm25":
#         return BM25Retriever(config)
#     else:
#         return DenseRetriever(config)


# #####################################
# # FastAPI server below
# #####################################


# class Config:
#     """
#     Minimal config class (simulating your argparse)
#     Replace this with your real arguments or load them dynamically.
#     """

#     def __init__(
#         self,
#         retrieval_method: str = "bm25",
#         retrieval_topk: int = 10,
#         index_path: str = "./index/bm25",
#         corpus_path: str = "./data/corpus.jsonl",
#         dataset_path: str = "./data",
#         data_split: str = "train",
#         faiss_gpu: bool = True,
#         retrieval_model_path: str = "./model",
#         retrieval_pooling_method: str = "mean",
#         retrieval_query_max_length: int = 256,
#         retrieval_use_fp16: bool = False,
#         retrieval_batch_size: int = 128,
#     ):
#         self.retrieval_method = retrieval_method
#         self.retrieval_topk = retrieval_topk
#         self.index_path = index_path
#         self.corpus_path = corpus_path
#         self.dataset_path = dataset_path
#         self.data_split = data_split
#         self.faiss_gpu = faiss_gpu
#         self.retrieval_model_path = retrieval_model_path
#         self.retrieval_pooling_method = retrieval_pooling_method
#         self.retrieval_query_max_length = retrieval_query_max_length
#         self.retrieval_use_fp16 = retrieval_use_fp16
#         self.retrieval_batch_size = retrieval_batch_size


# class QueryRequest(BaseModel):
#     queries: list[str]
#     topk: Optional[int] = None
#     return_scores: bool = False


# app = FastAPI()


# @app.post("/retrieve")
# def retrieve_endpoint(request: QueryRequest):
#     """
#     Endpoint that accepts queries and performs retrieval.

#     Input format:
#     {
#       "queries": ["What is Python?", "Tell me about neural networks."],
#       "topk": 3,
#       "return_scores": true
#     }

#     Output format (when return_scores=True，similarity scores are returned):
#     {
#         "result": [
#             [   # Results for each query
#                 {
#                     {"document": doc, "score": score}
#                 },
#                 # ... more documents
#             ],
#             # ... results for other queries
#         ]
#     }
#     """
#     if not request.topk:
#         request.topk = config.retrieval_topk  # fallback to default

#     # Perform batch retrieval
#     results, scores = retriever.batch_search(
#         query_list=request.queries, num=request.topk, return_score=request.return_scores
#     )

#     # Format response
#     resp = []
#     for i, single_result in enumerate(results):
#         if request.return_scores:
#             # If scores are returned, combine them with results
#             combined = []
#             for doc, score in zip(single_result, scores[i], strict=True):
#                 combined.append({"document": doc, "score": score})
#             resp.append(combined)
#         else:
#             resp.append(single_result)
#     return {"result": resp}


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Launch the local faiss retriever.")
#     parser.add_argument(
#         "--index_path", type=str, default="/home/peterjin/mnt/index/wiki-18/e5_Flat.index", help="Corpus indexing file."
#     )
#     parser.add_argument(
#         "--corpus_path",
#         type=str,
#         default="/home/peterjin/mnt/data/retrieval-corpus/wiki-18.jsonl",
#         help="Local corpus file.",
#     )
#     parser.add_argument("--topk", type=int, default=3, help="Number of retrieved passages for one query.")
#     parser.add_argument("--retriever_name", type=str, default="e5", help="Name of the retriever model.")
#     parser.add_argument(
#         "--retriever_model", type=str, default="intfloat/e5-base-v2", help="Path of the retriever model."
#     )
#     parser.add_argument("--faiss_gpu", action="store_true", help="Use GPU for computation")

#     args = parser.parse_args()

#     # 1) Build a config (could also parse from arguments).
#     #    In real usage, you'd parse your CLI arguments or environment variables.
#     config = Config(
#         retrieval_method=args.retriever_name,  # or "dense"
#         index_path=args.index_path,
#         corpus_path=args.corpus_path,
#         retrieval_topk=args.topk,
#         faiss_gpu=args.faiss_gpu,
#         retrieval_model_path=args.retriever_model,
#         retrieval_pooling_method="mean",
#         retrieval_query_max_length=256,
#         retrieval_use_fp16=True,
#         retrieval_batch_size=512,
#     )

#     # 2) Instantiate a global retriever so it is loaded once and reused.
#     retriever = get_retriever(config)

#     # 3) Launch the server. By default, it listens on http://127.0.0.1:8003
#     uvicorn.run(app, host="0.0.0.0", port=8003)
