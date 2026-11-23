import torch,torch_npu,json
from pathlib import Path
from types import SimpleNamespace
from verl.utils.kimi_logprob import mask_kimi_generation_logits
from verl.utils.torch_functional import logprobs_from_logits
w=Path('/mnt/share/y00938198/kimi-k31/test_wxx_skill')
root=w/'results/kimi31_4layer_alignment_423_20261121_145504/audit/prefix_probe'
records=[torch.load(p,map_location='cpu',weights_only=True) for p in sorted(root.glob('*.pt'))]
raw=torch.stack([r['original_full_logits'] for r in records]);labels=torch.tensor([r['target_token'] for r in records])
config=SimpleNamespace(model_type='kimi_k31',media_placeholder_token_id=163605)
torch_npu.npu.set_device(0);torch_npu.npu.set_deterministic_level(3)
leaf=raw.npu().detach().requires_grad_(True);target=labels.npu()
with torch.no_grad():old=logprobs_from_logits(mask_kimi_generation_logits(leaf,config),target,force_fp32=True)
grad=logprobs_from_logits(mask_kimi_generation_logits(leaf,config),target,force_fp32=True)
assert grad.dtype==torch.float32 and old.dtype==torch.float32 and torch.equal(old,grad)
grad.sum().backward();actual_grad=leaf.grad.detach().cpu()
masked=mask_kimi_generation_logits(raw.float(),config)
reference=masked.log_softmax(-1).gather(1,labels[:,None]).squeeze(1)
expected_grad=-masked.softmax(-1);expected_grad.scatter_add_(1,labels[:,None],torch.ones((len(labels),1)));expected_grad=expected_grad.to(raw.dtype)
assert torch.isfinite(actual_grad).all() and (actual_grad[:,163605]==0).all()
torch.testing.assert_close(grad.detach().cpu(),reference,atol=4e-6,rtol=0)
torch.testing.assert_close(actual_grad,expected_grad,atol=1e-5,rtol=0.02)
result={'real_prefix_vectors':len(records),'vocab_size':raw.shape[-1],'input_dtype':str(raw.dtype),'old_dtype':str(old.dtype),'grad_dtype':str(grad.dtype),'old_vs_grad_different':int((old!=grad).sum()),'cpu_reference_logprob_max_abs':float((grad.detach().cpu()-reference).abs().max()),'cpu_reference_gradient_max_abs':float((actual_grad.float()-expected_grad.float()).abs().max()),'banned_token_gradient_zero':True,'passed':True,'scope':'real logits scoring and VJP; full model/KDA alignment not claimed'}
(w/'results/kimi-scoring-validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
