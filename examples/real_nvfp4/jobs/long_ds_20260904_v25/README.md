# v25 — two matched long runs: W4A4 routed-expert carve-out vs BF16

Both arms are R3 on, token-level TIS on, dynamic sampling on, and use the
**default** (lenient) verifier. They differ only in precision.

| | `ARM=w4a4` | `ARM=bf16` |
| --- | --- | --- |
| `PRECISION_MODE` | `real_nvfp4` | `bf16` |
| `first_last_layers_bf16` | `True`, 2 at the start / 4 at the end | off (meaningless without quantization) |
| NVFP4 MLP layers | 42 of 48 | 0 |
| rollout weight sync | native reload + vLLM online NVFP4 | native reload (`VERL_VLLM_NATIVE_RELOAD=1`) |

The carve-out matches NeMo-RL's R3-on arm: only the routed experts are
quantized, and the first two and last four decoder layers stay BF16.
`mlp_fc1_nvfp4=42` in the attestation is the independent confirmation that the
carve-out actually reached Megatron — if the knob silently did nothing the count
would still read 48 and the gate fails.

## Dynamic sampling

`FILTER_GROUPS=True`, `MAX_GEN_BATCHES=20`, `GEN_PROMPT_BSZ_MULT=3`, mirroring
NeMo's `use_dynamic_sampling=True` / `batch_multiplier=3` /
`dynamic_sampling_max_gen_batches=20`. Each step generates up to 3x the prompts
and keeps the informative groups, so step time is roughly 2-3x the v13 run's.
That is why the chain uses 40-step chunks rather than v13's 80-step first chunk.

## Why this bundle does not build an image

`rn4pt_require_runtime_image` exists to prove the runtime matches the source.
v23's image was built from this exact source commit and passed preflight, and
nothing outside `examples/real_nvfp4/jobs/**` has changed since — which the
manifest re-verifies with a `git diff` against the recorded build commit, plus
the image checksum and the recorded `uv.lock` hash. Building a byte-identical
runtime would only cost an hour, so this bundle inherits that evidence instead.
If any runtime payload changes, the diff check fails closed and a rebuild is
required.

## Usage

```bash
bash submit.sh w4a4 audit     # static + image checks only, no job
bash submit.sh w4a4 smoke     # 1 node, 3 steps, gates on numerics
bash submit.sh w4a4 release   # 8 nodes, 7 chained chunks to step 260
```

`release` refuses to run until the matching smoke has produced
`<arm>_chunk_0.pass`. Three earlier BF16 smokes were declared passing on
structural gates alone while the rollout was actually random, so every phase
also gates on `rollout_corr/kl` and `actor/entropy`.

Both arms log to W&B `shawnzzz/DAPO-NVFP4-QAT` under a single run id per arm, so
the chained chunks appear as one curve.
