"""Compact, read-only first-layer boundaries for corresponding real requests."""
import os,time
from pathlib import Path
import torch

def trace_prefill(module,q,k,v,g,beta,output,cu_seqlens):
    if not os.environ.get('KIMI_ALIGNMENT_AUDIT_DIR') or '.layers.0.' not in module.prefix:
        return
    from vllm.forward_context import get_forward_context
    bounds=cu_seqlens.detach().cpu().tolist() if isinstance(cu_seqlens,torch.Tensor) else list(cu_seqlens)
    indices=torch.tensor([int(x)-1 for x in bounds[1:]],device=q.device,dtype=torch.long)
    if any(int(b)<=int(a) for a,b in zip(bounds,bounds[1:])):
        return  # A diagnostic must not reinterpret padding-only requests.
    actual=get_forward_context().attn_metadata[module.prefix].num_actual_tokens
    trace={'prefill_boundaries':bounds,'global_rows':indices+(actual-q.shape[1])}
    offset=actual-q.shape[1]
    trace['prefill_global_offset']=offset
    raw_qkv=getattr(module,'_alignment_raw_qkv',None)
    if raw_qkv is not None:
        trace['projection_qkv_prefix']=raw_qkv[offset:actual].detach().clone()
    raw_beta=getattr(module,'_alignment_raw_beta',None)
    if raw_beta is not None:
        trace['raw_beta']=raw_beta[0].index_select(0,trace['global_rows']).detach().clone()
        trace['raw_beta_prefix']=raw_beta[0,offset:actual].detach().clone()
    for name,value in [('q',q),('k',k),('v',v),('g',g),('beta',beta),('core_output',output)]:
        trace[name]=value[0].index_select(0,indices).detach().clone()
        trace[name+'_prefix']=value[0].detach().clone()
    trace['A_log']=module.A_log.detach().clone()
    trace['dt_bias']=module.dt_bias.detach().clone()
    module._alignment_kda_trace=trace

def trace_norm(module,g2,norm_out):
    trace=getattr(module,'_alignment_kda_trace',None)
    if trace is None:return
    rows=trace['global_rows']
    trace['norm_weight']=module.o_norm.weight.detach().clone()
    trace['norm_eps']=float(module.o_norm.eps)
    trace['norm_activation']=module.o_norm.activation
    trace['split_gated_norm']=os.environ.get('KIMI_ALIGNMENT_SPLIT_GATED_NORM','0')
    trace['output_gate']=g2.index_select(0,rows).detach().clone()
    trace['gated_norm_output']=norm_out[0].index_select(0,rows).detach().clone()
    begin=trace['prefill_global_offset'];end=begin+int(trace['prefill_boundaries'][-1])
    trace['output_gate_prefix']=g2[begin:end].detach().clone()
    trace['gated_norm_output_prefix']=norm_out[0,begin:end].detach().clone()

def trace_output(module,hidden,output):
    trace=getattr(module,'_alignment_kda_trace',None)
    if trace is None:return
    rows=trace['global_rows']
    trace['input_hidden_local']=hidden.detach().clone()
    trace['attention_output_local']=output.detach().clone()
    if hidden.ndim!=2 or output.ndim!=2 or int(rows.max())>=min(hidden.shape[0],output.shape[0]):
        trace['output_boundary_unavailable']={'hidden_shape':list(hidden.shape),'output_shape':list(output.shape)}
        return
    trace['input_hidden']=hidden.index_select(0,rows).detach().clone()
    trace['attention_output']=output.index_select(0,rows).detach().clone()

def save_runner_traces(runner,batch,rank,step):
    root=os.environ.get('KIMI_ALIGNMENT_AUDIT_DIR')
    if not root:return
    traces={}
    for name,module in runner.model.named_modules():
        value=getattr(module,'_alignment_kda_trace',None)
        if value is None:continue
        traces[name]={k:v.detach().cpu() if isinstance(v,torch.Tensor) else v for k,v in value.items()}
    if not traces:
        layers = {name: module._alignment_layer_output_trace for name,module in runner.model.named_modules() if hasattr(module,'_alignment_layer_output_trace')}
        path = Path(root)/'vllm_kda_trace'
        path.mkdir(parents=True,exist_ok=True)
        torch.save({'rank':rank,'step':step,'request_ids':list(batch.req_ids),'traces':{},'layer_outputs':layers},path/(str(time.time_ns())+f'_rank{rank:03d}.pt'))
        return
    path=Path(root)/'vllm_kda_trace';path.mkdir(parents=True,exist_ok=True)
    record={'rank':rank,'step':step,'request_ids':list(batch.req_ids),'traces':traces}
    record['first_input_norm']={name:module._alignment_input_norm_trace for name,module in runner.model.named_modules() if hasattr(module,'_alignment_input_norm_trace')}
    record['post_attention']={name:module._alignment_post_attention_trace for name,module in runner.model.named_modules() if hasattr(module,'_alignment_post_attention_trace')}
    if not getattr(runner, '_alignment_projection_weights_saved', False):
        record['projection_runtime']={name:{'weight':module.weight.detach().cpu().clone(),
            'class':type(module).__module__+'.'+type(module).__name__,
            'custom_op':type(getattr(module,'custom_op',None)).__name__,
            'full_k_flag':os.environ.get('KIMI_ALIGNMENT_FULL_O_PROJ','0')}
            for name,module in runner.model.named_modules() if name.endswith('layers.0.self_attn.o_proj')}
        runner._alignment_projection_weights_saved=True
    record['layer_outputs']={name:module._alignment_layer_output_trace for name,module in runner.model.named_modules() if hasattr(module,'_alignment_layer_output_trace')}
    record['moe']={name:module._alignment_moe_trace for name,module in runner.model.named_modules() if hasattr(module,'_alignment_moe_trace')}
    record['dsa']={name:module._alignment_dsa_trace for name,module in runner.model.named_modules() if hasattr(module,'_alignment_dsa_trace')}
    for key in ('logits_indices','query_start_loc'):
        value=getattr(batch,key,None)
        record[key]=value.detach().cpu() if isinstance(value,torch.Tensor) else None
    torch.save(record,path/(str(time.time_ns())+f'_rank{rank:03d}.pt'))
