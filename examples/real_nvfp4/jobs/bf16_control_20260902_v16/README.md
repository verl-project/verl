# BF16 control v16 - MoE weight_loader fix

v15 turned the silent hang into a real failure in 5m23s:

```
RuntimeError: shard_dim=0 is not a valid data dimension for a 3D tensor (expected 1 or 2)
  receive_weights -> _update_weights -> model.load_weights(param_updates)
  -> qwen3_moe.py:538 -> fused_moe/routed_experts.py:914
```

`routed_experts.py:914` calls `param.weight_loader(..., expert_id=...)` per
expert. `patch_vllm_moe_model_weight_loader` had overwritten that param loader
with the module-level `experts.weight_loader`, a vLLM 0.8.2 workaround for
params that had no loader at all. On vLLM 0.26 the param already has the right
loader, so the override makes it treat the 3D expert param as shardable.

v16 skips that workaround whenever vLLM exposes `RoutedExperts`, and rebuilds.
Real NVFP4 never hit this because it bypasses `model.load_weights` entirely in
favour of `model_runner.reload_weights(..., is_checkpoint_format=True)`.

```bash
./submit.sh probe && ./submit.sh build && ./submit.sh preflight
./submit.sh smoke
./submit.sh control
```
