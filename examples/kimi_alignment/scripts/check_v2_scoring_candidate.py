import json
from pathlib import Path
import torch, torch_npu
from types import SimpleNamespace
from kimi_scoring_candidate import kimi_v2_logprobs, masked_entropy
from verl.utils.kimi_logprob import mask_kimi_generation_logits
from verl.utils.torch_functional import logprobs_from_logits
from vllm_ascend.worker.v2.sample.logprob import compute_token_logprobs

w=Path('/mnt/share/y00938198/kimi-k31/test_wxx_skill')
paths=sorted((w/'results/kimi31_4layer_alignment_423_20261121_145504/audit/prefix_probe').glob('*.pt'))
records=[torch.load(p,map_location='cpu',weights_only=True) for p in paths]
raw=torch.stack([r['original_full_logits'] for r in records])
labels=torch.tensor([r['target_token'] for r in records])
config=SimpleNamespace(model_type='kimi_k31',media_placeholder_token_id=163605)
torch_npu.npu.set_device(0);torch_npu.npu.set_deterministic_level(3)
leaf=raw.npu().detach().requires_grad_(True); target=labels.npu()
masked=mask_kimi_generation_logits(leaf,config)
with torch.no_grad():
    rollout=compute_token_logprobs(masked.float(),target[:,None]).squeeze(-1)
    old=kimi_v2_logprobs(masked,target)
    native=logprobs_from_logits(masked,target,force_fp32=True)
    single=torch.cat([kimi_v2_logprobs(masked[i:i+1],target[i:i+1]) for i in range(len(target))])
grad=kimi_v2_logprobs(masked,target)
assert old.dtype==grad.dtype==torch.float32
assert torch.equal(rollout,old) and torch.equal(old,grad) and torch.equal(old,single)
grad.sum().backward()
actual_grad=leaf.grad.detach().cpu()
cpu=mask_kimi_generation_logits(raw.double(),config).detach().requires_grad_(True)
reference=cpu.log_softmax(-1).gather(1,labels[:,None]).squeeze(-1)
reference.sum().backward()
torch.testing.assert_close(grad.detach().cpu(),reference.detach(),atol=8e-6,rtol=0,check_dtype=False)
torch.testing.assert_close(actual_grad,cpu.grad.to(raw.dtype),atol=1e-5,rtol=0.02)
assert torch.isfinite(actual_grad).all() and (actual_grad[:,163605]==0).all()

entropy_input=raw.npu().detach().requires_grad_(True)
entropy=masked_entropy(mask_kimi_generation_logits(entropy_input,config))
entropy.sum().backward()
assert torch.isfinite(entropy).all() and torch.isfinite(entropy_input.grad).all()
cpu_finite=torch.cat((raw.double()[:,:163605],raw.double()[:,163606:]),-1).requires_grad_(True)
cpu_lp=cpu_finite.log_softmax(-1)
entropy_ref=-(cpu_lp.exp()*cpu_lp).sum(-1)
entropy_ref.sum().backward()
expected=torch.cat((cpu_finite.grad[:,:163605],torch.zeros((len(raw),1)),cpu_finite.grad[:,163605:]),-1)
torch.testing.assert_close(entropy.detach().cpu(),entropy_ref.detach(),atol=8e-6,rtol=0,check_dtype=False)
torch.testing.assert_close(entropy_input.grad.cpu(),expected.to(raw.dtype),atol=1e-5,rtol=.02)
result={'passed':True,'vectors':len(raw),'vocab_size':raw.shape[-1],'cpu_reference_dtype':'float64',
        'old_grad_rollout_and_batch_different':0,
        'native_vs_v2_max_abs':float((native-rollout).abs().max()),
        'native_vs_v2_different':int((native!=rollout).sum()),
        'v2_vs_cpu_max_abs':float((grad.detach().cpu()-reference.detach()).abs().max()),
        'gradient_vs_cpu_max_abs':float((actual_grad.float()-cpu.grad.to(raw.dtype).float()).abs().max()),
        'cpu_reference_gradient_rounding_max_abs':float((cpu.grad.to(raw.dtype).double()-cpu.grad).abs().max()),
        'entropy_finite':True,'entropy_vjp_finite':True,
        'entropy_vs_cpu_max_abs':float((entropy.detach().cpu()-entropy_ref.detach()).abs().max()),
        'scope':'real-logit scoring, batch invariance and analytical VJP; not full-model acceptance'}
(w/'results/kimi-v2-scoring-validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
