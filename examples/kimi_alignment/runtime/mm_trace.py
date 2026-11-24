"""Read-only media and first-input-norm evidence for task-local Kimi replay."""
import os,time
from pathlib import Path
from contextlib import contextmanager
import torch

def freeze(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,dict):return {str(k):freeze(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [freeze(v) for v in value]
    if value is None or isinstance(value,(str,int,float,bool)):return value
    return {'type':type(value).__name__}

@contextmanager
def trace_media(owner,media_input):
    root=os.environ.get('KIMI_ALIGNMENT_AUDIT_DIR')
    rank=torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    count=getattr(owner,'_alignment_media_count',0)
    pixels=media_input.get('pixel_values')
    # Profile dummy images can be huge; only bounded real-sized inputs are recorded.
    if not root or rank!=0 or count>=32 or pixels.numel()>16000000:
        yield
        return
    owner._alignment_media_count=count+1
    traces={};handles=[]
    for name,module in owner.named_modules():
        if not (name in ('vision_tower','mm_projector') or name.startswith(('vision_tower.patch_embed','vision_tower.encoder.blocks.0.')) or name in ('vision_tower.encoder.blocks.0','vision_tower.encoder.final_layernorm')):continue
        def hook(module,args,kwargs,output,name=name):
            traces[name]={'args':freeze(args),'kwargs':freeze(kwargs),'output':freeze(output),
                          'class':type(module).__module__+'.'+type(module).__name__}
        handles.append(module.register_forward_hook(hook,with_kwargs=True))
    record={'rank':rank,'count':count,'input':freeze({'pixel_values':media_input['pixel_values'],'grid_thws':media_input['grid_thws']}),'traces':traces,
            'vision_dtype':str(next(owner.vision_tower.parameters()).dtype),
            'use_data_parallel':owner.use_data_parallel,'complete':False}
    try:
        yield
        record['complete']=True
    finally:
        for handle in handles:handle.remove()
        path=Path(root)/'vllm_media_trace';path.mkdir(parents=True,exist_ok=True)
        torch.save(record,path/(str(time.time_ns())+'_rank000.pt'))

def trace_input_norm(module,x,blocks,result,addend,optimize_prefill):
    if not os.environ.get('KIMI_ALIGNMENT_AUDIT_DIR') or module.prev_valid_blocks!=0:return
    module._alignment_input_norm_trace=freeze({'prefix_sum':x,'addend':addend,
        'output':result[0],'block_written':blocks[:,0],
        'weight':module.input_layernorm.weight,'eps':module.input_layernorm.variance_epsilon,
        'optimize_prefill':optimize_prefill,'split_norm':os.environ.get('KIMI_ALIGNMENT_SPLIT_INPUT_NORM','0'),
        'class':type(module).__module__+'.'+type(module).__name__})

def trace_post_attention(module, hidden, prefix, blocks, output=None, mlp_output=None):
    if not os.environ.get('KIMI_ALIGNMENT_AUDIT_DIR') or module.prev_valid_blocks != 0:return
    if output is None:
        module._alignment_post_attention_trace = freeze({
            'attention_output_local':hidden,'prefix_sum':prefix,'block0':blocks[:,0],
            'mlp_res_proj_weight':module.mlp_res_proj.weight,
            'mlp_res_norm_weight':module.mlp_res_norm.weight,
            'mlp_res_norm_eps':module.mlp_res_norm.variance_epsilon,
            'post_norm_weight':module.post_attention_layernorm.weight,
            'post_norm_eps':module.post_attention_layernorm.variance_epsilon})
    else:
        module._alignment_post_attention_trace.update(freeze({
            'normalized_output_local':output,'prefix_sum_out':prefix,'mlp_output_local':mlp_output}))
