"""Task-local evidence capture and opt-in fixed-weight forward probes; never modifies scores."""
import os,time,json
from pathlib import Path
import torch

def cpu_tensor(value):
    if not isinstance(value,torch.Tensor): return None
    if getattr(value,'is_nested',False):
        return {'values':value.values().detach().cpu(), 'offsets':value.offsets().detach().cpu()}
    return value.detach().cpu()

def freeze_inputs(value):
    if isinstance(value, torch.Tensor): return cpu_tensor(value)
    if type(value).__name__ == 'NonTensorStack' and hasattr(value,'tolist'): return freeze_inputs(value.tolist())
    if type(value).__name__ == 'NonTensorData' and hasattr(value,'data'): return freeze_inputs(value.data)
    if isinstance(value, dict): return {str(k):freeze_inputs(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)): return [freeze_inputs(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)): return value
    return {'unserialized_type':type(value).__name__}

def capture_forward(engine, micro_batch, model_inputs, model_output, forward_only):
    root=os.environ.get('KIMI_ALIGNMENT_AUDIT_DIR')
    if not root: return
    p=Path(root)/'workers';p.mkdir(parents=True,exist_ok=True)
    lp=model_output.get('log_probs')
    if lp is None: return
    record={'rank':engine.rank,'forward_only':bool(forward_only),
            'engine_forward_only':bool(engine.engine_config.forward_only),
            'grad_enabled':torch.is_grad_enabled(),'logprobs_require_grad':lp.requires_grad,
            'training':engine.module.training,'log_probs':cpu_tensor(lp),
            'micro_batch_keys':[str(k) for k in micro_batch.keys()],
            'input_ids':cpu_tensor(model_inputs.get('input_ids')),
            'model_inputs':freeze_inputs(model_inputs)}
    for key in ('input_ids','position_ids','attention_mask','loss_mask','response_mask','responses','old_log_probs','rollout_log_probs'):
        if key in micro_batch:
            result=cpu_tensor(micro_batch[key])
            if result is not None: record['batch_'+key]=result
    for key in ('uid','rollout_key','session_id','global_steps','index'):
        if key in micro_batch:record['batch_'+key]=freeze_inputs(micro_batch[key])
    name=f'{time.time_ns()}_rank{engine.rank:03d}_grad{int(torch.is_grad_enabled())}'
    torch.save(record,p/(name+'.pt'))
    print('KIMI_ALIGNMENT_FORWARD '+json.dumps({k:v for k,v in record.items() if k in ('rank','forward_only','engine_forward_only','grad_enabled','logprobs_require_grad','training')}),flush=True)

def probe_prefix(engine, model_inputs, raw_output, micro_batch):
    """A/A and prefix-length probe at the first response position, fixed weights."""
    if os.environ.get('KIMI_ALIGNMENT_PREFIX_PROBE') != '1' or getattr(engine, '_alignment_prefix_done', False): return
    if torch.is_grad_enabled() or engine.engine_config.forward_only: return
    ids=model_inputs.get('input_ids')
    if ids is None or ids.ndim != 2 or ids.shape[0] != 1: return
    responses=micro_batch['responses']
    response_values=responses.values() if getattr(responses,'is_nested',False) else responses.reshape(-1)
    prefix_length=ids.shape[1]-response_values.numel()
    if prefix_length < 1: return
    engine._alignment_prefix_done=True
    import re
    from collections.abc import Mapping
    def first_tensor(value):
        if isinstance(value,torch.Tensor): return value
        if isinstance(value,Mapping): value=list(value.values())
        if isinstance(value,(list,tuple)):
            for item in value:
                found=first_tensor(item)
                if found is not None: return found
        return None
    def selected_logits(output):
        logits=output.logits
        if hasattr(logits,'full_tensor'): logits=logits.full_tensor()
        return logits[0,expanded_prefix_length-1].detach().cpu().clone()
    expanded_prefix_length=raw_output.logits.shape[1]-response_values.numel()
    original_logits=selected_logits(raw_output)
    traces={};phase='repeat_full';handles=[]
    for name,module in engine.module.named_modules():
        is_layer=bool(re.search(r'language_model\.model\.layers\.\d+$',name))
        is_embedding=name.endswith('language_model.model.embed_tokens')
        is_visual=name.endswith(('vision_tower','mm_projector','vision_tower.encoder.final_layernorm')) or 'vision_tower.patch_embed' in name or 'vision_tower.encoder.blocks.0.' in name or name.endswith('vision_tower.encoder.blocks.0')
        is_fine=bool(re.search(r'language_model\.model\.layers\.[013]\.',name))
        if not (is_layer or is_embedding or is_visual or is_fine): continue
        def hook(module,args,output,name=name,is_visual=is_visual,is_fine=is_fine):
            if name.endswith('layers.3.self_attn'):
                for key,value in getattr(module,'_alignment_training_dsa_trace',{}).items():
                    traces.setdefault(phase,{})[name+'/dsa/'+key]=value
                module._alignment_capture_dsa=False
            tensor=first_tensor(output)
            if tensor is None: return
            if hasattr(tensor,'to_local'):tensor=tensor.to_local()
            if name.endswith('layers.1.block_sparse_moe.experts.act_fn'):
                traces.setdefault(phase,{})[name+'/full_output']=tensor.detach().cpu().clone()
            if is_fine:
                if tensor.ndim>=3 and tensor.shape[0]==1 and tensor.shape[1]>=expanded_prefix_length:tensor=tensor[0,:expanded_prefix_length]
                elif tensor.ndim>=2 and tensor.shape[0]>=expanded_prefix_length:tensor=tensor[:expanded_prefix_length]
                else:return
            elif not is_visual:
                trace_position=(prefix_length if name.endswith('embed_tokens') else expanded_prefix_length)-1
                if tensor.ndim==3 and tensor.shape[1]>trace_position:tensor=tensor[0,trace_position]
                elif tensor.ndim==2 and tensor.shape[0]>trace_position:tensor=tensor[trace_position]
                else:return
            traces.setdefault(phase,{})[name]=tensor.detach().cpu().clone()
            if name.endswith('layers.1.block_sparse_moe.gate') and isinstance(output,tuple):
                for j,value in enumerate(output):
                    if isinstance(value,torch.Tensor):
                        traces.setdefault(phase,{})[name+'/item'+str(j)]=value.detach().cpu().clone()
        handles.append(module.register_forward_hook(hook))
    for name,module in engine.module.named_modules():
        if not ((any('.layers.'+str(i)+'.' in name for i in (0,1,3)) and name.endswith(('self_attn','self_attn.o_norm','self_attn.o_proj','input_layernorm'))) or name.endswith(('vision_tower','mm_projector','layers.1.block_sparse_moe.experts','layers.1.block_sparse_moe.experts.act_fn'))):continue
        def core_input_hook(module,args,kwargs,name=name):
            if name.endswith('layers.3.self_attn'):
                module._alignment_capture_dsa=True
                module._alignment_training_dsa_trace={}
            if name.endswith('self_attn.o_norm'):
                weight=module.weight
                if hasattr(weight,'to_local'):weight=weight.to_local()
                traces.setdefault(phase,{})[name+'/norm_weight']=weight.detach().cpu().clone()
                traces.setdefault(phase,{})[name+'/norm_eps']=torch.tensor(module.variance_epsilon,dtype=torch.float64)
            inputs=list(args) if args else [kwargs.get('hidden_states')]
            for idx,tensor in enumerate(inputs):
                if not isinstance(tensor,torch.Tensor):continue
                if hasattr(tensor,'to_local'):tensor=tensor.to_local()
                if tensor.ndim>=3 and tensor.shape[0]==1:tensor=tensor[0,:expanded_prefix_length]
                traces.setdefault(phase,{})[name+'/input'+str(idx)]=tensor.detach().cpu().clone()
        handles.append(module.register_forward_pre_hook(core_input_hook,with_kwargs=True))
    for name,module in engine.module.named_modules():
        if not name.endswith('language_model.lm_head'): continue
        def lm_head_input_hook(module,args,name=name):
            tensor=first_tensor(args)
            if tensor is None:return
            if hasattr(tensor,'to_local'):tensor=tensor.to_local()
            if tensor.ndim==3:tensor=tensor[0,expanded_prefix_length-1]
            elif tensor.ndim==2:tensor=tensor[expanded_prefix_length-1]
            else:return
            traces.setdefault(phase,{})[name+'/input']=tensor.detach().cpu().clone()
        handles.append(module.register_forward_pre_hook(lm_head_input_hook))
    try:
        repeated=engine.module(**model_inputs,use_cache=False)
        repeated_logits=selected_logits(repeated)
        del repeated
        prefix_inputs=dict(model_inputs)
        for key in ('input_ids','attention_mask','position_ids'):
            if key in prefix_inputs:prefix_inputs[key]=prefix_inputs[key][...,:prefix_length]
        phase='prefix_only'
        prefix_output=engine.module(**prefix_inputs,use_cache=False)
        prefix_logits=selected_logits(prefix_output)
        del prefix_output
    finally:
        for handle in handles:handle.remove()
    result={'rank':engine.rank,'prefix_length':prefix_length,'expanded_prefix_length':expanded_prefix_length,'response_length':response_values.numel(),
            'target_token':int(response_values[0]),'input_ids':ids.detach().cpu(),
            'original_full_logits':original_logits,'repeat_full_logits':repeated_logits,
            'prefix_logits':prefix_logits,'boundary_traces':traces}
    root=Path(os.environ['KIMI_ALIGNMENT_AUDIT_DIR'])/'prefix_probe';root.mkdir(parents=True,exist_ok=True)
    name=f'{time.time_ns()}_rank{engine.rank:03d}'
    torch.save(result,root/(name+'.pt'))
    summary={'rank':engine.rank,'prefix_length':prefix_length,
             'repeat_full_max_abs':float((original_logits.float()-repeated_logits.float()).abs().max()),
             'prefix_vs_full_max_abs':float((original_logits.float()-prefix_logits.float()).abs().max()),
             'boundary_names':list(traces.get('repeat_full',{}))}
    (root/(name+'.json')).write_text(json.dumps(summary,indent=2)+'\n')
    print('KIMI_ALIGNMENT_PREFIX_PROBE '+json.dumps(summary),flush=True)


def capture_vllm_v2_step(runner, logits, selected_hidden, batch):
    """Capture eight real V2 sampling inputs before any logits processing."""
    root=os.environ.get('KIMI_ALIGNMENT_AUDIT_DIR')
    if not root: return
    req_ids=list(batch.req_ids)
    if not req_ids or any('warmup' in str(x) for x in req_ids): return
    if os.environ.get("KIMI_ALIGNMENT_SERVING_PROBE") == "1" and any(str(r).startswith("alignment-probe-") for r in req_ids):
        from serving_invariance_probe import capture_probe_logits
        from vllm.distributed import get_tensor_model_parallel_rank
        capture_probe_logits(logits,batch,get_tensor_model_parallel_rank())
    count=getattr(runner,'_alignment_capture_count',0)
    if count >= 8: return
    runner._alignment_capture_count=count+1
    from vllm.distributed import get_tensor_model_parallel_rank
    rank=get_tensor_model_parallel_rank()
    from kda_trace import save_runner_traces
    save_runner_traces(runner,batch,rank,count)
    if rank != 0: return
    requests=[]
    for req_id in req_ids:
        idx=runner.req_states.req_id_to_index[req_id]
        prompt_len=int(runner.req_states.prompt_len.np[idx])
        requests.append({'req_id':req_id,'prompt_length':prompt_len,
            'prompt_token_ids':cpu_tensor(runner.req_states.all_token_ids.gpu[idx,:prompt_len])})
    record={'rank':rank,'step':count,'requests':requests,
            'logits':cpu_tensor(logits),'selected_hidden':cpu_tensor(selected_hidden)}
    language_model=runner.model.get_language_model()
    record['post_norm_hidden']=cpu_tensor(getattr(language_model,'_alignment_post_norm_hidden',None))
    for key in ('input_ids','positions','logits_indices','query_start_loc'):
        record[key]=cpu_tensor(getattr(batch,key,None))
    for key in ('num_computed_tokens_np','query_start_loc_np','is_prefilling_np'):
        value=getattr(batch,key,None)
        if value is not None:record[key]=value.tolist()
    path=Path(root)/'vllm_steps';path.mkdir(parents=True,exist_ok=True)
    torch.save(record,path/(str(time.time_ns())+'_rank'+str(rank)+'.pt'))
    print('KIMI_ALIGNMENT_VLLM_STEP '+json.dumps({'step':count,'rank':rank,'requests':len(requests)}),flush=True)


from contextlib import contextmanager

@contextmanager
def trace_actual_forward(engine,model_inputs,micro_batch,forward_only):
    root=os.environ.get('KIMI_ALIGNMENT_AUDIT_DIR')
    if not root or engine.rank!=0 or engine.engine_config.forward_only:
        yield
        return
    phase='old' if forward_only else 'grad'
    counter=getattr(engine,'_alignment_trace_counts',{})
    if counter.get(phase,0)>=2:
        yield
        return
    counter[phase]=counter.get(phase,0)+1;engine._alignment_trace_counts=counter
    traces={};handles=[]
    def tensor_output(x):
        if isinstance(x,torch.Tensor):return x
        if isinstance(x,(list,tuple)):
            for item in x:
                value=tensor_output(item)
                if value is not None:return value
        return None
    for name,module in engine.module.named_modules():
        if not ('language_model.model.layers.0' in name or name.endswith(('embed_tokens','vision_tower','mm_projector','language_model.model','layers.1','layers.2','layers.3'))):continue
        def hook(module,args,out,name=name):
            value=tensor_output(out)
            if value is not None:
                if hasattr(value,'to_local'):value=value.to_local()
                traces[name]=value.detach().cpu().clone()
        handles.append(module.register_forward_hook(hook))
    for name,module in engine.module.named_modules():
        if not ((any('.layers.'+str(i)+'.' in name for i in (0,1,3)) and name.endswith(('self_attn','self_attn.o_norm','self_attn.o_proj','input_layernorm'))) or name.endswith(('vision_tower','mm_projector','layers.1.block_sparse_moe.experts','layers.1.block_sparse_moe.experts.act_fn'))):continue
        def input_hook(module,args,kwargs,name=name):
            inputs=list(args) if args else [kwargs.get('hidden_states')]
            for idx,value in enumerate(inputs):
                if not isinstance(value,torch.Tensor):continue
                if hasattr(value,'to_local'):value=value.to_local()
                traces[name+'/input'+str(idx)]=value.detach().cpu().clone()
        handles.append(module.register_forward_pre_hook(input_hook,with_kwargs=True))
    complete=False
    try:
        yield
        complete=True
    finally:
        for handle in handles:handle.remove()
        path=Path(root)/'actual_forward_trace';path.mkdir(parents=True,exist_ok=True)
        record={'phase':phase,'rank':engine.rank,'complete':complete,'input_ids':cpu_tensor(model_inputs['input_ids']),
                'response_length':values_for_trace(micro_batch['responses']).numel(),'traces':traces,'model_inputs':freeze_inputs(model_inputs)}
        torch.save(record,path/(str(time.time_ns())+'_'+phase+'.pt'))

def values_for_trace(x):
    return x.values() if getattr(x,'is_nested',False) else x


def replay_fixed_failure(engine,model_inputs):
    """Rescore frozen attempt11 trajectories at initial checkpoint weights."""
    if engine.engine_config.forward_only or torch.is_grad_enabled() or getattr(engine,'_alignment_fixed_done',False):return
    engine._alignment_fixed_done=True
    w=Path(os.environ['KIMI_ALIGNMENT_WORKSPACE'])
    baseline=w/'results/kimi31_4layer_alignment_423_20261121_143057/audit/workers'
    from verl.utils.torch_functional import logprobs_from_logits
    if os.environ.get("KIMI_ALIGNMENT_V2_SCORING") == "1":
        from kimi_scoring import kimi_v2_logprobs as logprobs_from_logits
    root=Path(os.environ['KIMI_ALIGNMENT_AUDIT_DIR'])/'fixed_failure_replay';root.mkdir(parents=True,exist_ok=True)
    device=model_inputs['input_ids'].device
    for path in sorted(baseline.glob('*_rank'+f'{engine.rank:03d}'+'_grad0.pt')):
        record=torch.load(path,map_location='cpu',weights_only=True)
        if record['engine_forward_only']:continue
        inputs={k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in record['model_inputs'].items()}
        output=engine.module(**inputs,use_cache=False)
        from verl.utils.kimi_logprob import mask_kimi_generation_logits
        logits=mask_kimi_generation_logits(output.logits,engine.model_config.hf_config)
        if hasattr(logits,'full_tensor'):logits=logits.full_tensor()
        responses=record['batch_responses']['values'];n=responses.numel()
        labels=torch.zeros(logits.shape[:-1],dtype=torch.long,device=device)
        labels[0,-n-1:-1]=responses.to(device)
        scores=logprobs_from_logits(logits,labels,force_fp32=True)[0,-n-1:-1].detach().cpu()
        expected=record['batch_rollout_log_probs']['values'];old=record['log_probs']['values'][-n-1:-1]
        mask=record.get('batch_response_mask',record['batch_loss_mask'])['values'].bool()
        delta=(scores-expected).abs();valid=mask & torch.isfinite(scores) & torch.isfinite(expected)
        summary={'baseline_file':path.name,'rank':engine.rank,'comparison':'fixed_attempt11_rollout_vs_independent_rescore',
                 'valid_tokens':int(mask.sum()),'nonfinite_tokens':int((mask & ~valid).sum()),
                 'different_tokens':int(((scores!=expected)&mask).sum()),
                 'mean_abs_diff':float(delta[mask].mean()),'max_abs_diff':float(delta[mask].max()),
                 'new_vs_original_training_max_abs':float((scores-old).abs()[mask].max()),
                 'passed':bool(mask.any() and valid[mask].all() and (scores[mask]==expected[mask]).all())}
        name=str(time.time_ns())+'_rank'+str(engine.rank)
        torch.save({'summary':summary,'new_logprobs':scores,'original_training_logprobs':old,'rollout_logprobs':expected,'mask':mask,'responses':responses},root/(name+'.pt'))
        (root/(name+'.json')).write_text(json.dumps(summary,indent=2)+'\n')
        print('KIMI_ALIGNMENT_FIXED_REPLAY '+json.dumps(summary),flush=True)
        del output,logits
