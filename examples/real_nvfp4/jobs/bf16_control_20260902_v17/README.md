# BF16 control v17 - native reload

## Why v16 still failed

v16 skipped verl's vLLM-0.8.2 MoE loader workaround, but the failure was
identical. The full traceback shows why: the loader being invoked is vLLM's
**own** `RoutedExperts.weight_loader`, not the patched one.

```
routed_experts.py:914  param.weight_loader(...)
routed_experts.py:858  weight_loader -> _load_model_weight_or_group_weight_scale
routed_experts.py:356  -> _load_w13
routed_experts.py:490  hidden_dim = self._get_hidden_dim(shard_dim, expert_data.ndim)
routed_experts.py:409  ValueError: shard_dim=0 is not a valid data dimension for a 3D tensor
```

vLLM 0.26 reworked MoE loading into `RoutedExperts`, whose `weight_loader`
expects a per-expert 2D view; verl's legacy `model.load_weights` path hands it
the stacked 3D expert param. **verl main still pins vLLM 0.24, so plain BF16 MoE
weight sync has simply never run on 0.26.** That is an upstream gap, not damage
from this branch - and it is invisible in the real-NVFP4 arm because that path
never calls `model.load_weights`.

## What v17 changes

`VERL_VLLM_NATIVE_RELOAD=1` routes non-quantized weight sync through
`model_runner.reload_weights(..., is_checkpoint_format=True)` - the same API
real NVFP4 uses successfully on this exact vLLM. Default behaviour is unchanged;
only this control opts in. `train.job` asserts the marker so a silent fallback
to the broken path cannot pass as a control.

Porting verl's legacy MoE loader to the RoutedExperts API is the fuller fix, but
it is a detour from the question this control exists to answer.

```bash
./submit.sh probe && ./submit.sh build && ./submit.sh preflight
./submit.sh smoke && ./submit.sh control
```
