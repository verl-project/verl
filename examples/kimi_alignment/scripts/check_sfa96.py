import torch,torch_npu,math,json,datetime
from pathlib import Path
from fsdp_turbo.ops.npu.sparse_flash_attention import _sparse_flash_attention_npu
from fsdp_turbo.ops.cpu.sparse_attention import sparse_flash_attention_torch
w=Path('/mnt/share/y00938198/kimi-k31/test_wxx_skill')
torch.manual_seed(739);torch_npu.npu.set_device(0);torch_npu.npu.set_deterministic_level(3)
results=[]
for packed in (False,True):
 for shared in (True,):
  S,H,D,R=8,96,512,64;prefix=(S,) if packed else (1,S)
  def random(shape):return (torch.randn(shape)*0.1).to(torch.bfloat16)
  raw=[random((*prefix,H,D)),random((*prefix,1,D)),random((*prefix,1,D)),random((*prefix,H,R)),random((*prefix,1,R))]
  dout=random((*prefix,H,D));indices=torch.full((*prefix,2048),-1,dtype=torch.int32)
  for row in range(S):
   length=row%4+1 if packed else row+1
   if packed:indices[row,:length]=torch.arange(length,dtype=torch.int32)
   else:indices[0,row,:length]=torch.arange(length,dtype=torch.int32)
  boundaries=torch.tensor([0,4,8],dtype=torch.int32) if packed else None
  outputs=[]
  for device,fn in [('cpu',sparse_flash_attention_torch),('npu',_sparse_flash_attention_npu)]:
   tensors=[x.to(device).detach().requires_grad_(True) for x in raw]
   if shared:tensors[2]=tensors[1]
   q,k,v,qr,kr=tensors
   kwargs=dict(rope_head_dim=R,query_rope=qr,key_rope=kr,cu_seqlens=None if boundaries is None else boundaries.to(device),return_softmax_stats=True)
   output,_,_=fn(q,k,v,indices.to(device),1/math.sqrt(D+R),**kwargs)
   if device=='npu':
    with torch.no_grad():repeated=fn(q,k,v,indices.to(device),1/math.sqrt(D+R),**kwargs)[0]
    assert torch.equal(output,repeated),'Forward changed with grad mode'
   output.backward(dout.to(device))
   grads=[x.grad.detach().cpu() for x in tensors]
   outputs.append((output.detach().cpu(),grads))
  pairs=[('output',outputs[0][0],outputs[1][0])]+[(name,a,b) for name,a,b in zip(('dq','dk','dv','dq_rope','dk_rope'),outputs[0][1],outputs[1][1])]
  stats=[]
  for name,a,b in pairs:
   delta=(a.float()-b.float());relative_l2=float(delta.norm()/a.float().norm().clamp_min(1e-12))
   stats.append({'name':name,'max_abs':float(delta.abs().max()),'max_reference_abs':float(a.float().abs().max()),'relative_l2':relative_l2,'finite':bool(torch.isfinite(b).all()),'passed':bool(torch.isfinite(b).all()) and relative_l2 < 0.01})
  r={'layout':'TND' if packed else 'BSND','shared_kv':shared,'query_heads':H,'stats':stats,'passed':all(x['passed'] for x in stats)}
  results.append(r);print(json.dumps(r),flush=True)
  (w/'results/sfa-96head-shared-kv-validation.json').write_text(json.dumps({'time':datetime.datetime.now().astimezone().isoformat(),'cases':results,'tolerance':'finite and relative L2 < 0.01 against BF16-input FP32 PyTorch attention reference','scope':'shared-K/V MLA operator forward/backward only; separate K/V is unsupported; not full-model alignment'},indent=2)+'\n')
assert all(x['passed'] for x in results)
