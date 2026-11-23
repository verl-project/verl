"""Real-operand recurrence, cache, packing, and native-VJP regression."""
import os, json
from pathlib import Path
import torch, torch_npu
from safetensors import safe_open
from kda_alignment import canonical_kda, serving_decode
from fsdp_turbo.ops.kda import chunk_kda

w = Path('/mnt/share/y00938198/kimi-k31/test_wxx_skill')
trace = sorted((w/'results/kimi31_4layer_alignment_423_20261121_145504/audit/actual_forward_trace').glob('*_old.pt'))[0]
t = torch.load(trace, map_location='cpu', weights_only=True)['traces']
prefix = 'language_model.model.layers.0.self_attn.'
torch_npu.npu.set_device(0)
torch_npu.npu.set_deterministic_level(3)
raw = {name: t[prefix+module][:,:65].reshape(1,65,96,128).contiguous().npu()
       for name,module in [('q','q_conv1d'),('k','k_conv1d'),('v','v_conv1d'),('g','f_b_proj')]}
raw['beta'] = t[prefix+'b_proj'][:,:65].float().contiguous().npu()
index = json.load(open(w/'model/model.safetensors.index.json'))['weight_map']
params = {}
for name in ('A_log','dt_bias'):
    key=prefix+name
    with safe_open(str(w/'model'/index[key]), framework='pt', device='cpu') as handle:
        value=handle.get_tensor(key)
    if name=='A_log':
        assert (value[96:]==0).all()
        value=value[:96]
    params[name]=value.to(torch.bfloat16).float().contiguous().npu()

results={'scope':'real KDA operands; serving caches/packing and training native VJP; not full-model acceptance', 'trace':trace.name}
def stats(a,b):
    a=a.detach().float().cpu(); b=b.detach().float().cpu(); d=(a-b).abs()
    return {'different':int((a!=b).sum()),'max_abs':float(d.max()),
            'relative_l2':float(torch.linalg.vector_norm(d)/torch.linalg.vector_norm(b).clamp_min(1e-20)),
            'finite':bool(torch.isfinite(a).all() and torch.isfinite(b).all())}
def save():
    (w/'results/kda-alignment-validation.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results),flush=True)
def call(values,**kw):
    return canonical_kda(**values,a_log=params['A_log'],dt_bias=params['dt_bias'],beta_is_preprocessed=False,**kw)

full, final = call(raw)
head = {k:v[:,:32] for k,v in raw.items()}; tail={k:v[:,32:] for k,v in raw.items()}
first, state=call(head); rest, state2=call(tail,initial_state=state)
results['prefill_then_continue']=stats(torch.cat((first,rest),1),full)
results['continued_final_state']=stats(state2,final)
packed={k:torch.cat((v,v[:,:32]),1) for k,v in raw.items()}
out,states=call(packed,cu_seqlens=[0,65,97])
results['packed_neighbor_output']=stats(out,torch.cat((full,first),1))
results['packed_neighbor_state']=stats(states,torch.cat((final,state),0))
# Two requests in swapped cache slots; repeated one-token decode after prefill.
cache=torch.zeros((4,96,128,128),device='npu',dtype=torch.float32)
cache[2:3].copy_(state);cache[0:1].copy_(state)
decoded=[]
for pos in range(32,65):
    values=[torch.cat((raw[k][:,pos:pos+1],raw[k][:,pos:pos+1]),1) for k in ('q','k','v','g')]
    beta=torch.cat((raw['beta'][:,pos:pos+1],raw['beta'][:,pos:pos+1]),1).sigmoid()
    result=serving_decode(*values,beta,cache,[0,1,2],[2,0],params['A_log'],params['dt_bias'],lower_bound=-5.)
    assert torch.equal(result[:,:1],result[:,1:])
    decoded.append(result[:,:1])
results['prefill_decode']=stats(torch.cat((first,*decoded),1),full)
results['decode_final_state']=stats(cache[2:3],final)
assert all(v['different']==0 for v in results.values() if isinstance(v,dict))
save()

kw=dict(output_final_state=True,use_qk_l2norm_in_kernel=True,use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,safe_gate=True,lower_bound=-5.,
        transpose_state_layout=True,cu_seqlens=torch.tensor([0,65],dtype=torch.int32,device='npu'),skip_recompute=False)
torch.manual_seed(19)
cotangent=torch.randn(full.shape,dtype=torch.float32,device='cpu').to(torch.bfloat16).npu()
grads={}
for mode in ('native','aligned'):
    os.environ['KIMI_ALIGNMENT_RECURRENT_KDA']='1' if mode=='aligned' else '0'
    values={k:v.detach().clone().requires_grad_(True) for k,v in {**raw,**params}.items()}
    output, unused = chunk_kda(**values,**kw)
    assert unused is None
    if mode=='aligned':
        results['training_forward']=stats(output,full)
        assert results['training_forward']['different']==0
    output.backward(cotangent)
    grads[mode]={k:v.grad.detach().cpu() for k,v in values.items()}
results['native_vjp']={k:stats(grads['aligned'][k],grads['native'][k]) for k in grads['native']}
assert all(x['different']==0 and x['finite'] for x in results['native_vjp'].values())
results['passed']=True
save()
