"""Bounded diagnostic-only DSA operands; never copy the full paged KV cache."""
import os
from contextlib import contextmanager
import torch
from mm_trace import freeze
_active=None

def stage(name,**values):
    if _active is not None:_active['stages'][name]={k:freeze(v) for k,v in values.items()}

def training_stage(owner,**values):
    if getattr(owner,'_alignment_capture_dsa',False):
        owner._alignment_training_dsa_trace.update({k:freeze(v) for k,v in values.items()})

@contextmanager
def capture_dsa(owner,positions,hidden):
    global _active
    from vllm.forward_context import get_forward_context,is_forward_context_available
    if (not os.environ.get('KIMI_ALIGNMENT_AUDIT_DIR') or not is_forward_context_available()
        or get_forward_context().attn_metadata is None or hidden.shape[0]<=16
        or getattr(owner,'_alignment_dsa_count',0)>=2):
        yield;return
    owner._alignment_dsa_count=getattr(owner,'_alignment_dsa_count',0)+1
    record={'input':freeze(hidden),'positions':freeze(positions),'modules':{},'stages':{}}
    wanted=('q_a_proj','q_a_layernorm','q_b_proj','fused_qkv_a_proj','kv_a_proj_with_mqa',
            'kv_a_layernorm','kv_b_proj','g_proj','o_proj','q_proj','k_proj','weights_proj')
    handles=[]
    for name,module in owner.named_modules():
        if not name.endswith(wanted):continue
        def hook(module,args,kwargs,output,name=name):
            record['modules'][name]={'args':freeze(args),'kwargs':{k:freeze(v) for k,v in kwargs.items() if isinstance(v,torch.Tensor)},
                'output':freeze(output),'class':type(module).__module__+'.'+type(module).__name__}
            record['modules'][name]['variance_epsilon']=getattr(module,'variance_epsilon',None)
            weight=getattr(module,'weight',None)
            if isinstance(weight,torch.Tensor):record['modules'][name]['weight']=freeze(weight)
        handles.append(module.register_forward_hook(hook,with_kwargs=True))
    previous=_active;_active=record
    try:yield
    finally:
        for h in handles:h.remove()
        _active=previous;owner._alignment_dsa_trace=record
