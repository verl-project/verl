"""Replay actual Kimi DSA projections and attention on a captured identical prefix."""
import os,sys,json,subprocess
from pathlib import Path
from collections import defaultdict
w=Path('/mnt/share/y00938198/kimi-k31/test_wxx_skill')
dev=os.environ['KIMI_PROBE_PHYSICAL_DEVICE'];assert os.environ['ASCEND_RT_VISIBLE_DEVICES']==dev
subprocess.run(['/usr/bin/python3',str(w/'scripts/check_idle.py'),'--devices',dev],check=True)
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
mode=sys.argv[2] if len(sys.argv)>2 else 'training'
assert mode in ('training','serving')
if mode=='training':
    os.environ['VERL_NPU_TRAINING_PROCESS_PID']=str(os.getpid())
    os.environ['TORCH_EXTENSIONS_DIR']=str(w/'cache/local/torch_extensions_probe_training')
    for key in ('ASCEND_CUSTOM_OPP_PATH','LD_LIBRARY_PATH'):
        os.environ[key]=':'.join(x for x in os.environ.get(key,'').split(':') if x and '/vllm_ascend/_cann_ops_custom/' not in x)
else:
    os.environ['TORCH_EXTENSIONS_DIR']=str(w/'cache/local/torch_extensions_probe_serving')
import torch,torch_npu
from safetensors import safe_open
if mode=='serving':
    from vllm_ascend.utils import bootstrap_custom_op_env
    bootstrap_custom_op_env()
    import vllm_ascend.vllm_ascend_C
torch.set_num_threads(2);torch_npu.npu.set_device(0);torch_npu.npu.set_deterministic_level(3)
from fsdp_turbo.ops.npu.sparse_flash_attention import _sparse_flash_attention_npu
from fsdp_turbo.ops.npu.lightning_indexer import lightning_indexer_topk_npu
root=w/'results'/sys.argv[1];audit=root/'audit';read=lambda p:torch.load(p,map_location='cpu',weights_only=True)
def tensor(x):return x if isinstance(x,torch.Tensor) else x[0]
def stats(a,b):
    a,b=a.detach().cpu(),b.detach().cpu()
    if a.shape!=b.shape:return {'shape_mismatch':[list(a.shape),list(b.shape)]}
    d=(a.float()-b.float()).abs()
    return {'numel':a.numel(),'different':int((a!=b).sum()),'nonfinite':int((~(torch.isfinite(a)&torch.isfinite(b))).sum()),'max_abs':float(d.max()),'mean_abs':float(d.mean())}
groups=defaultdict(list)
for c in json.loads((root/'cross-backend-kda-boundaries.json').read_text()):groups[(c['training_file'],c['request_id'])].append(c)
records=None
for (name,req),cs in groups.items():
    if len(cs)!=8:continue
    rs=[read(audit/'vllm_kda_trace'/c['serving_file']) for c in sorted(cs,key=lambda c:c['tp_rank'])]
    if all(r['step']==0 and len(r['request_ids'])==1 and r.get('dsa') for r in rs):records=rs;break
assert records is not None
ds=[next(iter(r['dsa'].values())) for r in records]
tr=read(audit/'prefix_probe'/name)['boundary_traces']['prefix_only'];base='language_model.model.layers.3.self_attn'
n=tr[base+'.q_a_proj'].shape[0]
index=json.loads((w/'model/model.safetensors.index.json').read_text())['weight_map']
def weight(suffix):
    key=base+'.'+suffix
    with safe_open(str(w/'model'/index[key]),framework='pt',device='cpu') as f:return f.get_tensor(key).to(torch.bfloat16).contiguous().npu()
def training(key):return tr[base+'.'+key].contiguous().npu()
report={'run':root.name,'mode':mode,'training_file':name,'request_id':req,'tokens':n,'ASCEND_CUSTOM_OPP_PATH':os.environ.get('ASCEND_CUSTOM_OPP_PATH'),'checks':{}}
def check(label,a,b):
    r=stats(a,b);report['checks'][label]=r;print(json.dumps({label:r}),flush=True)
def save():
    (root/f'dsa-fixed-{mode}-probe.json').write_text(json.dumps(report,indent=2)+'\n')
with torch.no_grad():
    qa=training('q_a_proj');qweight=weight('q_a_layernorm.weight')
    sq=tensor(ds[0]['modules']['q_a_layernorm']['output']).npu()
    for eps in (1e-5,1e-6):
        qnorm=torch_npu.npu_rms_norm(qa,qweight,epsilon=eps)[0]
        check(f'qnorm_eps{eps}_training',qnorm,training('q_a_layernorm'))
        check(f'qnorm_eps{eps}_serving',qnorm,sq)
    kv=training('kv_a_proj_with_mqa');kw=weight('kv_a_layernorm.weight')
    for eps in (1e-5,1e-6):check(f'kvnorm_eps{eps}_training',torch_npu.npu_rms_norm(kv[:,:512].contiguous(),kw,epsilon=eps)[0],training('kv_a_layernorm'))
    if mode=='serving':
        from types import SimpleNamespace
        from dsa_alignment import write_raw_nope_cache
        impl=SimpleNamespace(qk_rope_head_dim=64,kv_lora_rank=512,enable_sparse_sfa_c8=False,enable_sparse_sfa_turboquant=False,
            kv_a_layernorm=lambda x:torch_npu.npu_rms_norm(x,kw,epsilon=1e-6)[0])
        pages=(n+4+127)//128
        stride=128*576+256
        def make_caches():
            storage=torch.zeros(pages*stride,device='npu',dtype=torch.bfloat16)
            return storage,[torch.as_strided(storage,(pages,128,1,z),(stride,z,z,1),storage_offset=offset) for z,offset in ((512,0),(64,128*512))]
        storage,caches=make_caches()
        slots=(torch.arange(n,device='npu',dtype=torch.int32).flip(0)+4).contiguous()
        def write_cache(raw,indices,cache):
            write_raw_nope_cache(impl,raw,indices,cache)
        write_cache(kv,slots,caches)
        check('raw_cache_latent',caches[0].reshape(-1,512)[slots.long()],training('kv_a_layernorm'))
        check('raw_cache_extra',caches[1].reshape(-1,64)[slots.long()],kv[:,512:])
        split_storage,split=make_caches()
        for lo,hi in ((0,1),(1,35),(35,n)):write_cache(kv[lo:hi],slots[lo:hi],split)
        for j in range(2):check(f'raw_cache_split_{j}',split[j],caches[j])
        check('raw_cache_full_strided_storage_split',split_storage,storage)
        before=storage.clone()
        write_cache(kv[:1],torch.full((1,),-1,device='npu',dtype=torch.int32),caches)
        check('raw_cache_negative_slot_skipped',storage,before)
        report['strided_cache_shapes']=[list(c.shape) for c in caches]
        report['strided_cache_strides']=[list(c.stride()) for c in caches]
    save()
    qb=training('q_b_proj').reshape(n,96,192)
    kp,vp=weight('kv_b_proj.weight').reshape(96,256,512).split([128,128],dim=1)
    q=torch.einsum('shd,hdr->shr',qb[...,:128],kp).contiguous()
    k=training('kv_a_layernorm').reshape(n,1,512);qr=qb[...,128:].contiguous();kr=kv[:,512:].reshape(n,1,64).contiguous()
    sqabs=torch.cat([d['stages']['pre_attention']['query_absorbed'] for d in ds],1).npu()
    sqextra=torch.cat([d['stages']['pre_attention']['query_rope'] for d in ds],1).npu()
    check('absorbed_query_actual',sqabs,q)
    # Interleaved storage may permute both Q and K channels without changing dot products.
    sb=torch.cat([tensor(d['modules']['q_b_proj']['output']).reshape(n,12,192) for d in ds],1).npu()
    check('query_extra_identity_from_serving_qb',sqextra,sb[...,128:])
    check('query_extra_interleave_from_serving_qb',sqextra,sb[...,128:].reshape(n,96,2,32).transpose(-2,-1).reshape(n,96,64))
    check('query_extra_deinterleave_from_serving_qb',sqextra,sb[...,128:].reshape(n,96,32,2).transpose(-2,-1).reshape(n,96,64))
    iq=training('indexer.wq_b').reshape(1,n,32,128)
    ik=training('indexer.k_norm').reshape(1,n,128);iw=training('indexer.weights_proj').reshape(1,n,32)*(32*128)**-0.5
    indices=lightning_indexer_topk_npu(iq,ik,iw,2048)
    si=ds[0]['stages']['pre_attention']['indices'].npu()
    check('index_order',indices[0],si)
    check('index_set',indices[0].sort(-1).values,si.sort(-1).values)
    scale=ds[0]['stages']['pre_attention']['scale']
    def training_core(q,k,qr,kr,idx):
        return _sparse_flash_attention_npu(q[None].contiguous(),k[None].contiguous(),k[None].contiguous(),idx.squeeze(-2),scale,64,qr[None].contiguous(),kr[None].contiguous(),return_softmax_stats=True)[0][0]
    fixture=root/'dsa-training-replay.pt'
    if mode=='training':core=training_core(q,k,qr,kr,indices)
    else:core=read(fixture)['core'].npu()
    value=torch.einsum('shr,hdr->shd',core,vp).reshape(n,-1)
    gated=value*training('g_proj').sigmoid()
    check('training_gated_output_reproduction',gated,training('o_proj/input0'))
    check('training_o_output_reproduction',torch.nn.functional.linear(gated,weight('o_proj.weight')),training('o_proj'))
    sy=torch.cat([d['stages']['attention_core']['output'] for d in ds],1).npu()
    sv=torch.cat([d['stages']['value_projection']['output'] for d in ds],1).npu()
    check('core_serving_vs_training',sy,core)
    check('value_serving_vs_training',sv,value)
    save()
    if mode=='training':
        torch.save({'core':core.cpu()},fixture)
        parts=[]
        for rank in range(8):parts.append(training_core(q[:,rank*12:(rank+1)*12],k,qr[:,rank*12:(rank+1)*12],kr,indices))
        check('training_operator_head12_vs96',torch.cat(parts,1),core)
        save();sys.exit(0)
    # Same TND paged op as actual nonzero-extra-channel serving branch, with reconstructed keys.
    def serving_core(q,k,qr,kr,idx):
        bs=128;pages=(n+bs-1)//bs
        def cache(x):return torch.nn.functional.pad(x,(0,0,0,0,0,pages*bs-n)).reshape(pages,bs,1,x.shape[-1]).contiguous()
        return torch.ops._C_ascend.npu_sparse_flash_attention(query=q.contiguous(),key=cache(k),value=cache(k),sparse_indices=idx.contiguous(),
            scale_value=scale,sparse_block_size=1,block_table=torch.arange(pages,device='npu',dtype=torch.int32)[None],
            actual_seq_lengths_query=torch.tensor([q.shape[0]],device='npu',dtype=torch.int32),actual_seq_lengths_kv=torch.tensor([n],device='npu',dtype=torch.int32),
            query_rope=qr.contiguous(),key_rope=cache(kr),layout_query='TND',layout_kv='PA_BSND',sparse_mode=3,attention_mode=2,return_softmax_lse=False)[0]
    oldk=torch_npu.npu_rms_norm(kv[:,:512].contiguous(),kw,epsilon=1e-5)[0].reshape(n,1,512)
    perm=lambda x:x.reshape(*x.shape[:-1],32,2).transpose(-2,-1).reshape_as(x).contiguous()
    original=[];candidate=[];raw_channel_candidate=[];last_query_candidate=[]
    for rank in range(8):
        sl=slice(rank*12,(rank+1)*12)
        old=serving_core(sqabs[:,sl],oldk,sqextra[:,sl],perm(kr),si)
        corrected=serving_core(q[:,sl],k,perm(qr[:,sl]),perm(kr),si)
        raw=serving_core(q[:,sl],k,qr[:,sl],kr,si)
        last_query_candidate.append(serving_core(q[-1:,sl],k,qr[-1:,sl],kr,si[-1:]))
        original.append(old);candidate.append(corrected);raw_channel_candidate.append(raw)
    check('original_paged_core_reproduction',torch.cat(original,1),sy)
    check('eps_corrected_paged_core_vs_training',torch.cat(candidate,1),core)
    check('eps_corrected_raw_channels_paged_core_vs_training',torch.cat(raw_channel_candidate,1),core)
    check('raw_channels_last_query_vs_prefill',torch.cat(last_query_candidate,1),core[-1:])
    corrected_q=[];corrected_v=[]
    qnorm=torch_npu.npu_rms_norm(qa,qweight,epsilon=1e-6)[0]
    for rank,d in enumerate(ds):
        sl=slice(rank*12,(rank+1)*12)
        qb_new=torch.nn.functional.linear(qnorm,d['modules']['q_b_proj']['weight'].npu()).reshape(n,12,192)
        check(f'rank{rank}_corrected_qb_vs_training',qb_new,qb[:,sl])
        q_new=torch_npu.npu_transpose_batchmatmul(qb_new[...,:128],d['stages']['pre_attention']['key_projection'].npu(),perm_x1=(1,0,2),perm_x2=(0,1,2),perm_y=(1,0,2))
        corrected_q.append(q_new)
        vv=torch_npu.npu_transpose_batchmatmul(raw_channel_candidate[rank],d['stages']['pre_attention']['value_projection'].npu(),perm_x1=(1,0,2),perm_x2=(0,1,2),perm_y=(1,0,2))
        corrected_v.append(vv.reshape(n,12,128))
    check('corrected_native_q_absorption',torch.cat(corrected_q,1),q)
    cv=torch.cat(corrected_v,1).reshape(n,-1)
    check('corrected_native_value_projection',cv,value)
    cg=cv*training('g_proj').sigmoid()
    check('corrected_native_gated_output',cg,training('o_proj/input0'))
    check('corrected_complete_DSA_output',torch.nn.functional.linear(cg,weight('o_proj.weight')),training('o_proj'))
    corrected_value=torch.einsum('shr,hdr->shd',torch.cat(candidate,1),vp).reshape(n,-1)
    check('eps_corrected_gated_vs_training',corrected_value*training('g_proj').sigmoid(),training('o_proj/input0'))
    save()
