"""Search-R1 HTTP retrieval with the shared corpus/index and FP32 E5."""

import argparse
import json
import os
from pathlib import Path
import threading
import time
from uuid import uuid4

from .retrieval_backend import FP32Encoder, RetrievalBackend


def main():
    import faiss
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    import torch
    import uvicorn
    from sd_search.retrieval import JsonlCorpus, configure_faiss_request_threads
    from sd_search.gpu_retrieval import GpuShardedIndex

    parser = argparse.ArgumentParser()
    for name in ['index','corpus','offsets','model']:
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--threads',type=int,default=20)
    parser.add_argument('--search-backend',choices=['cpu','gpu'],default='gpu')
    parser.add_argument('--gpu-count',type=int,choices=[1,4],default=4)
    parser.add_argument('--gpu-scratch-mib',type=int,default=1536)
    parser.add_argument('--port',type=int,default=8000)
    parser.add_argument('--timing-log')
    parser.add_argument('--startup-report')
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('threads must be positive')
    if os.environ.get('NVIDIA_TF32_OVERRIDE') != '0':
        raise RuntimeError('Set NVIDIA_TF32_OVERRIDE=0 for FP32 retrieval')
    torch.set_num_threads(args.threads)
    faiss.omp_set_num_threads(args.threads)
    index = faiss.read_index(args.index)
    corpus = JsonlCorpus(args.corpus,args.offsets)
    if index.ntotal != len(corpus) or index.d != 768:
        raise ValueError('Index/corpus mismatch')
    if args.search_backend == 'gpu':
        if args.gpu_count == 1:
            from .single_gpu_index import SingleGpuIndex
            index = SingleGpuIndex(index,scratch_mib=args.gpu_scratch_mib)
        else:
            index = GpuShardedIndex(index,scratch_mib=args.gpu_scratch_mib)
    encoder = FP32Encoder(args.model,args.device)
    backend = RetrievalBackend(index,corpus,encoder)
    startup = {'backend':args.search_backend,'documents':len(corpus),'dimension':index.d,
               'gpu_count':args.gpu_count if args.search_backend=='gpu' else 0,
               'encoder_device':args.device,'encoder':encoder.metadata(),'fp32':True,'tf32':False}
    if args.startup_report:
        Path(args.startup_report).write_text(json.dumps(startup,indent=2)+'\n')
    print('RETRIEVAL_STARTUP',json.dumps(startup),flush=True)
    lock = threading.Lock()
    app = FastAPI()

    class QueryRequest(BaseModel):
        queries: list[str]
        topk: int = 3
        return_scores: bool = True
        request_id: str | None = None

    @app.get('/health')
    def health():
        return {'ready':True,**startup}

    @app.post('/retrieve')
    def retrieve(request: QueryRequest):
        if not request.queries or not all(q.strip() for q in request.queries) or not 1 <= request.topk <= 100:
            raise HTTPException(400,'Nonempty queries and topk 1..100 required')
        started = time.perf_counter()
        with lock:
            wait = time.perf_counter()-started
            if args.search_backend == 'cpu':
                configure_faiss_request_threads(args.threads)
            output = backend.retrieve(request.queries,request.topk,request.return_scores)
        elapsed = time.perf_counter()-started
        stages = {'lock_wait':wait,**output['exclusive_s']}
        stages['other'] = elapsed-sum(stages.values())
        profile = {'request_id':request.request_id or uuid4().hex,'query_count':len(request.queries),
                   'total_s':elapsed,'exclusive_s':stages,'device':args.device,
                   'threads':{'torch':torch.get_num_threads(),'faiss':args.gpu_count if args.search_backend=='gpu' else args.threads},
                   'blas_threshold':faiss.cvar.distance_compute_blas_threshold,'search_backend':args.search_backend}
        if args.timing_log:
            with Path(args.timing_log).open('a') as stream:
                stream.write(json.dumps(profile,allow_nan=False)+'\n')
        return {'result':output['result'],'profile':profile}

    try:
        uvicorn.run(app,host='127.0.0.1',port=args.port,log_level='info')
    finally:
        os.close(corpus.fd)
        if args.search_backend == 'gpu':
            index.close()


if __name__ == '__main__':
    main()
