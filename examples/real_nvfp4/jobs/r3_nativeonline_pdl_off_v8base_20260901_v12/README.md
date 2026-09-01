# Real NVFP4 v12: PDL-off on the validated v8 image

v12 exists because v11 conflated two independent changes. It isolates the one
that is causally supported and drops the one that broke rollout numerics.

## Why

- `v8` (image `...nativeonline.20260901.v8.sqsh`) is the only real-NVFP4 image
  with verified healthy numerics: job `2694027` ran 8 nodes x 20 steps with
  `rollout_corr/kl` 0.0082 -> 0.0067, ESS 0.986 -> 0.989, entropy 0.87 -> 0.40,
  response length ~0.93k-1.17k, non-zero `actor/grad_norm`.
- `v9` reused that exact v8 image and still hit the intermittent
  "1-2 of 32 rollout servers never become ready" startup hang, so the hang is a
  probabilistic property of the v8 runtime, not of any v9 change.
- `v10` added two post-0.26 vLLM backports (#50029 expert packing, #50074
  reload kernel reuse) and still hung 30/32, so neither fixes the hang.
- `v11` added `enable_pdl=False` on top of v10 and reached 32/32 startup, but
  its rollout collapsed: every response clipped at 20,480 tokens,
  `rollout_corr/kl` 4.0-4.5, ESS 0.17-0.22, entropy 6.2, reward all -1.

The v11 collapse is explained by the #50074 backport, not by PDL. In vLLM
0.26 `Nvfp4OnlineMoEMethod._setup_kernel` rebuilds the quant config *and* the
kernel on every reload because `_quantize_weights` calls `replace_parameter`,
which rebinds `w13_weight_scale`, `w2_weight_scale`, and the `*_scale_2`
tensors to brand-new objects. `make_nvfp4_moe_quant_config` captures those
tensors by reference (`w1_scale=w13_scale`, `g1_alphas=w13_scale_2`, ...), and
`TrtllmNvfp4MoE.process_weights_after_loading` relies on the config tensor
being *the same object* as the registered parameter ("g1_alphas is set once
here ... and never changes again"). Guarding kernel creation with
`if self.moe_kernel is None:` keeps the first config alive, so after the very
first refit the kernel applies dummy-weight scales to freshly packed weights.
Forward still reads `layer.w13_weight` / `layer.w2_weight` directly
(`OnlineMoEMethodBase.apply_monolithic`), so weights are fresh while scales are
stale - exactly the observed near-uniform-entropy output.

## What v12 changes

Relative to the v8 image, exactly one runtime variable:

- vLLM `fused_moe/experts/trtllm_nvfp4_moe.py`: both FlashInfer TRTLLM NVFP4
  MoE call sites pass `enable_pdl=False`.

Not changed: CUDA graphs stay on in `FULL_DECODE_ONLY`, `max_num_seqs` stays
128, backend, quantization scope, native reload, R3 on, 0/3 loss, and every
dependency version.

## Acceptance gates

1. `preflight` asserts both `enable_pdl` API signatures and the two patched
   call sites, plus the existing TE/packing/reload/R3 regressions.
2. `short` (8 nodes, 20 steps) must reproduce two things at once: 32/32
   `launch_server` ready, and v8-class numerics
   (`rollout_corr/kl` < 0.02, ESS > 0.97, entropy < 1.5, response length far
   below the 20,480 clip, `actor/grad_norm` > 0).

Only after both does the long chain get derived from this bundle.

```bash
./submit.sh probe
./submit.sh build
./submit.sh preflight
./submit.sh short
```
