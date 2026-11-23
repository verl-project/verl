"""Shared retrieval backend with an explicitly FP32 E5 encoder."""

import time


class FP32Encoder:
    def __init__(self, model_path, device='cuda:0'):
        import torch
        from transformers import AutoModel, AutoTokenizer

        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModel.from_pretrained(model_path, local_files_only=True,
                                               torch_dtype=torch.float32).float().eval()
        self.to(device)

    def to(self, device):
        import torch
        self.device = torch.device(device)
        if self.device.type not in ['cpu', 'cuda']:
            raise ValueError('E5 device must be cpu or cuda')
        self.model.to(self.device)
        assert all(p.dtype == torch.float32 for p in self.model.parameters())
        return self

    def metadata(self):
        import torch
        return {'device': str(self.device), 'parameter_dtypes': sorted({str(p.dtype) for p in self.model.parameters()}),
                'eval_mode': not self.model.training, 'autocast': False,
                'matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
                'cudnn_tf32': torch.backends.cudnn.allow_tf32,
                'matmul_precision': torch.get_float32_matmul_precision()}

    def encode(self, queries):
        import numpy as np
        import torch
        import torch.nn.functional as F

        with torch.inference_mode(), torch.autocast(device_type=self.device.type, enabled=False):
            tick = time.perf_counter()
            tokens = self.tokenizer(['query: '+q for q in queries], padding=True, truncation=True,
                                    max_length=512, return_tensors='pt')
            tokens = {key: value.to(self.device) for key, value in tokens.items()}
            tokenization = time.perf_counter()-tick
            tick = time.perf_counter()
            hidden = self.model(**tokens).last_hidden_state
            assert hidden.dtype == torch.float32
            mask = tokens['attention_mask'].bool()
            embeddings = hidden.masked_fill(~mask[..., None], 0).sum(1)/mask.sum(1)[:, None]
            vectors = F.normalize(embeddings, p=2, dim=1).float().cpu().numpy()
            # D2H completes encoding before this timer stops.
            return np.ascontiguousarray(vectors), {'tokenization': tokenization,
                                                    'e5_encode': time.perf_counter()-tick}


class RetrievalBackend:
    def __init__(self, index, corpus, encoder):
        self.index, self.corpus, self.encoder = index, corpus, encoder

    def retrieve(self, queries, topk=3, return_scores=True, diagnostics=False):
        import numpy as np

        if not queries or not all(isinstance(q, str) and q.strip() for q in queries):
            raise ValueError('Nonempty query strings required')
        if not 1 <= topk <= 100:
            raise ValueError('topk must be 1..100')
        results, vectors_all, ids_all, scores_all = [], [], [], []
        stages = dict.fromkeys(['tokenization', 'e5_encode', 'faiss_search', 'document_read'], 0.0)
        for start in range(0, len(queries), 64):
            vectors, encoded = self.encoder.encode(queries[start:start+64])
            for key, value in encoded.items():
                stages[key] += value
            tick = time.perf_counter()
            scores, ids = self.index.search(vectors, topk)
            stages['faiss_search'] += time.perf_counter()-tick
            tick = time.perf_counter()
            for row_ids, row_scores in zip(ids, scores):
                docs = [self.corpus[int(i)] for i in row_ids]
                results.append([{'document':doc, 'score':float(score)} for doc,score in zip(docs,row_scores)]
                               if return_scores else docs)
            stages['document_read'] += time.perf_counter()-tick
            if diagnostics:
                vectors_all.append(vectors)
                ids_all.append(ids)
                scores_all.append(scores)
        output = {'result':results, 'exclusive_s':stages}
        if diagnostics:
            output.update(vectors=np.concatenate(vectors_all), ids=np.concatenate(ids_all),
                          scores=np.concatenate(scores_all))
        return output
