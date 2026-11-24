"""Bounded, read-only operands for the second decoder block's real MoE calls."""
import os
import dataclasses
from contextlib import contextmanager
import torch
from mm_trace import freeze

_active = None

def stage(name, **values):
    if _active is not None:
        _active['stages'][name] = {k: freeze_dataclass(v) for k,v in values.items()}

def freeze_dataclass(value):
    if dataclasses.is_dataclass(value):
        return {f.name: freeze_dataclass(getattr(value,f.name)) for f in dataclasses.fields(value)}
    return freeze(value)

@contextmanager
def capture_moe(owner, hidden):
    global _active
    prefix=getattr(owner.gate,'prefix','')
    if not os.environ.get('KIMI_ALIGNMENT_AUDIT_DIR') or '.layers.1.' not in prefix:
        yield;return
    from vllm.forward_context import get_forward_context, is_forward_context_available
    if not is_forward_context_available() or get_forward_context().attn_metadata is None:
        yield;return
    if hidden.shape[0] <= 16 or getattr(owner,'_alignment_moe_capture_count',0)>=2:
        yield;return
    owner._alignment_moe_capture_count=getattr(owner,'_alignment_moe_capture_count',0)+1
    record={'input_local':freeze(hidden),'prefix':prefix,'modules':{},'stages':{}}
    record['gate_runtime']={'weight':freeze(owner.gate.weight),'bias':freeze(owner.gate.e_score_correction_bias),'alignment':getattr(owner.gate,'_alignment_kimi_training_gate',False)}
    handles=[]
    names={'gate','routed_expert_down_proj','routed_expert_norm','routed_expert_up_proj',
           'shared_experts','shared_experts.gate_up_proj','shared_experts.act_fn',
           'shared_experts.down_proj','experts'}
    for name,module in owner.named_modules():
        if name not in names:continue
        def hook(module,args,kwargs,output,name=name):
            record['modules'][name]={'args':freeze(args),'kwargs':freeze(kwargs),
                'output':freeze(output),'class':type(module).__module__+'.'+type(module).__name__}
        handles.append(module.register_forward_hook(hook,with_kwargs=True))
    previous=_active;_active=record
    try:
        yield
    finally:
        for h in handles:h.remove()
        _active=previous
        owner._alignment_moe_trace=record
