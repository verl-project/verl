import numpy as np
import pytest
import torch

from examples.search_r1.retrieval_backend import FP32Encoder, RetrievalBackend


@pytest.fixture
def encoder(monkeypatch):
    from types import SimpleNamespace
    import transformers

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.projection=torch.nn.Linear(3,3,bias=False,dtype=torch.float64)
            with torch.no_grad():
                self.projection.weight.copy_(torch.eye(3,dtype=torch.float64))

        def forward(self,input_ids,attention_mask):
            del attention_mask
            return SimpleNamespace(last_hidden_state=self.projection(input_ids.float()))

    def tokenize(queries,**kwargs):
        assert queries==['query: alpha','query: beta']
        assert kwargs['max_length']==512 and kwargs['truncation']
        return {'input_ids':torch.tensor([[[1,0,0],[0,1,0],[100,100,100]],
                                          [[0,0,2],[0,0,2],[0,0,2]]]),
                'attention_mask':torch.tensor([[1,1,0],[1,1,1]])}

    def load(*args,**kwargs):
        assert kwargs['local_files_only'] and kwargs['torch_dtype']==torch.float32
        return TinyModel()

    monkeypatch.setattr(transformers.AutoModel,'from_pretrained',load)
    monkeypatch.setattr(transformers.AutoTokenizer,'from_pretrained',lambda *a,**k:tokenize)
    return FP32Encoder('offline-fixture','cpu')


def test_fp32_encoder_excludes_padding_and_disables_autocast(encoder):
    with torch.autocast('cpu',dtype=torch.bfloat16):
        vectors,timing=encoder.encode(['alpha','beta'])
    expected=np.array([[2**-.5,2**-.5,0],[0,0,1]],dtype=np.float32)
    np.testing.assert_allclose(vectors,expected,rtol=1e-6)
    assert vectors.dtype==np.float32
    assert all(value>=0 for value in timing.values())
    meta=encoder.metadata()
    assert meta['parameter_dtypes']==['torch.float32'] and meta['eval_mode']
    assert not meta['autocast'] and not meta['matmul_tf32'] and not meta['cudnn_tf32']


def test_backend_chunks_and_preserves_query_document_order():
    class Encoder:
        def encode(self,queries):
            return np.array([[int(q)] for q in queries],dtype=np.float32),{'tokenization':.1,'e5_encode':.2}

    class Index:
        calls=[]
        def search(self,vectors,k):
            assert k==3
            self.calls.append(len(vectors))
            ids=vectors.astype(np.int64)+np.array([[0,100,200]])
            return np.tile([.9,.8,.7],(len(vectors),1)),ids

    index=Index()
    corpus={i:{'contents':f'doc {i}'} for i in range(270)}
    backend=RetrievalBackend(index,corpus,Encoder())
    result=backend.retrieve([str(i) for i in range(70)],diagnostics=True)
    assert index.calls==[64,6]
    assert result['ids'].shape==(70,3)
    assert result['vectors'].shape==(70,1)
    assert result['result'][69][2]['document']['contents']=='doc 269'
    assert result['exclusive_s']['e5_encode']==.4
    assert backend.retrieve(['1'],return_scores=False)['result'][0][0]=={'contents':'doc 1'}
    with pytest.raises(ValueError):
        backend.retrieve([])
