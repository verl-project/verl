import torch, torch_npu, json, math, datetime
from pathlib import Path
from fsdp_turbo.ops.cpu.lightning_indexer import lightning_indexer_kl_loss_cpu
from fsdp_turbo.ops.npu.lightning_indexer import lightning_indexer_kl_loss_npu
W=Path('/mnt/share/y00938198/kimi-k31/test_wxx_skill')
torch.manual_seed(731)
torch_npu.npu.set_device(0)
torch_npu.npu.set_deterministic_level(3)
results=[]
for packed in (False, True):
    B,S,H,D,IH,ID,R=1,8,96,512,32,128,64
    prefix=(S,) if packed else (B,S)
    def random(shape, dtype=torch.bfloat16): return (torch.randn(shape)*0.1).to(dtype)
    raw=[random((*prefix,IH,ID)),random((*prefix,ID)),random((*prefix,IH),torch.float32),
         random((*prefix,H,D)),random((*prefix,1,D)),random((*prefix,H,R)),random((*prefix,1,R))]
    indices=torch.full((*prefix,1,2048),-1,dtype=torch.int32)
    for row in range(S):
        length=(row%4)+1 if packed else row+1
        if packed:indices[row,0,:length]=torch.arange(length,dtype=torch.int32)
        else:indices[0,row,0,:length]=torch.arange(length,dtype=torch.int32)
    maximum=torch.zeros((*prefix,H));denominator=torch.ones((*prefix,H));scale=1/math.sqrt(D+R)
    boundaries=torch.tensor([0,4,8],dtype=torch.int32) if packed else None
    outputs=[]
    for device,fn in [('cpu',lightning_indexer_kl_loss_cpu),('npu',lightning_indexer_kl_loss_npu)]:
        values=[x.to(device).detach().requires_grad_(True) for x in raw]
        loss=fn(*values,indices.to(device),maximum.to(device),denominator.to(device),scale,
                None if boundaries is None else boundaries.to(device))
        loss.backward()
        assert all(x.grad is None for x in values[3:]), 'Teacher gradients must remain detached'
        grads=[x.grad.detach().cpu() for x in values[:3]]
        assert torch.isfinite(loss) and all(torch.isfinite(g).all() for g in grads)
        outputs.append((loss.detach().cpu(),grads))
    torch.testing.assert_close(outputs[1][0],outputs[0][0],atol=1e-5,rtol=1e-3)
    stats=[]
    for a,b in zip(outputs[0][1],outputs[1][1]):
        torch.testing.assert_close(b,a,atol=2e-5,rtol=0.02)
        stats.append({'max_abs':float((a.float()-b.float()).abs().max()),'max_reference_abs':float(a.float().abs().max())})
    r={'layout':'TND' if packed else 'BSND','teacher_heads':H,'cpu_loss':float(outputs[0][0]),'npu_loss':float(outputs[1][0]),'gradient_errors_q_k_weights':stats,'teacher_detached':True,'passed':True}
    results.append(r);print(json.dumps(r),flush=True)
(W/'results/indexer-kl-96head-validation.json').write_text(json.dumps({'time':datetime.datetime.now().astimezone().isoformat(),'cases':results,'loss_tolerance':{'atol':1e-5,'rtol':1e-3},'gradient_tolerance':{'atol':2e-5,'rtol':0.02},'scope':'BF16 96-head KL fallback on synthetic BSND/TND cases; not a full-model alignment result'},indent=2)+'\n')
