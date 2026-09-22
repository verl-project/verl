# FP8 RL in verl

Last updated: 08/22/2026

verl supports two FP8 modes for accelerating RL training:

| Mode | Training Precision | Rollout Precision |
|------|-------------------|-------------------|
| **FP8 Rollout Only** | BF16 | FP8 |
| **FP8 End-to-End** | FP8 (Megatron) | FP8 (vLLM) |

> [!TIP]
> For ready-to-run scripts, see the [low-precision recipe directory](https://github.com/verl-project/verl-recipe/tree/main/low_precision).

---

## FP8 Rollout Only

FP8 rollout-only mode keeps training in BF16 and quantizes rollout inference to FP8. This reduces GPU memory during generation and speeds up rollout without affecting training precision.

### Implementation

We monkey patch several vLLM functions to enable FP8 rollout for reinforcement learning:

1. **Quantize weights**: Quantize model weights on-the-fly from higher-precision formats to FP8.
2. **Process weights after loading**: For vLLM, we replace the `vllm.model_executor.layers.quantization.fp8.Fp8LinearMethod.process_weights_after_loading` function to handle weight processing after quantization. For SGLang, this patch is not needed as it natively supports loading quantized weights.

### Support Matrix

- FP8 blockwise quantization for rollout
  - Used in Deepseek, which is 1x128 quantization for activations and 128x128 quantization for model weights
- Dense models and MoE models
- Async rollout interfaces
- vLLM 0.10.x & vLLM 0.11 & vLLM 0.12 & SGLang 0.5.5
- FSDP and Megatron training backends

### Usage

Enable in config file:

```yaml
rollout:
  quantization: "fp8"
```

Or via command line:

```bash
actor_rollout_ref.rollout.quantization=fp8
```

#### Skipping layers in SGLang FP8 rollout

When using SGLang FP8 rollout, you can skip FP8 weight quantization for
selected modules. Skipped modules stay in the rollout model dtype instead
of being converted to FP8. This is useful for layers that are not
compatible with block-wise FP8 weight quantization, or for modules that
you prefer to keep in higher precision.

Set `SGLANG_FP8_IGNORED_LAYERS` before starting training:

```bash
SGLANG_FP8_IGNORED_LAYERS=linear_attn \
python3 -m verl.trainer.main_ppo \
  actor_rollout_ref.rollout.name=sglang \
  actor_rollout_ref.rollout.quantization=fp8 \
  ...
```

Multiple entries can be separated by commas:

```bash
SGLANG_FP8_IGNORED_LAYERS=linear_attn,visual
```

You can also use the model `quantization_config`:

```json
{
  "quantization_config": {
    "ignored_layers": ["re:.*linear_attn.*"]
  }
}
```

Plain module names, full module paths, and `re:` regex patterns are
supported. verl applies the same ignored-layer rules when launching
SGLang and when syncing updated actor weights into the rollout engine.

### Experiments and Outcomes

#### Qwen3-8B-Base Dense Model

**Configuration**
- DAPO recipe. AIME24 online validation.
- vLLM(FP8 spmd rollout) + FSDP
  - Note that SPMD rollout has been deprecated, so we removed the FP8 SPMD rollout.
- Prompt batch size 32, n=16.
- Rollout batch size: 32\*3*16
- Train_batch_size & ppo_mini_batch_size 32
- Max response length 20K
- Token-level TIS, C=2
- 8*H100
- vLLM 0.10.0+CUDA 12.6 vs vLLM 0.11.0+CUDA 12.9

**Accuracy**
![Qwen3-8b-base_fp8_acc](
https://github.com/Agoniii/verl/blob/xueh/fp8_pr_images/docs/advance/images/Qwen3-8b-base_fp8_acc.png?raw=true)
*dark green: BF16, orange: FP8 rollout + token-level TIS, light green: FP8 rollout without TIS*

Results and observations:
- With TIS, FP8 rollout aligns with BF16
- Obvious accuracy drop when TIS is not enabled
- Higher mismatch kl but within acceptable range throughout the training


**Performance**

![Qwen3-8b-base_fp8_rollout_perf](
https://github.com/Agoniii/verl/blob/xueh/fp8_pr_images/docs/advance/images/Qwen3-8b-base_fp8_rollout_perf.png?raw=true)
*green: BF16, orange: FP8 rollout + CUDA12.6 + DeepGemm, purple: FP8 rollout + CUDA 12.9 + DeepGemm*

Results and observations:
- FP8 rollout leads to around ~12% rollout speedup with CUDA 12.6 + DeepGemm
- When upgrading to CUDA 12.9, speedup can be up to ~18%

#### Qwen3-30B-A3B-Base MoE Model

**Configuration**
- DAPO recipe. AIME24 online validation.
- FP8 async rollout, vLLM+FSDP
- Prompt batch size 32
- Rollout batch size: 32\*3*16
- Train_batch_size & ppo_mini_batch_size 32
- Max response length 20K
- Token-level TIS, C=2
- 2\*8*H100
- vLLM 0.10.0+CUDA 12.6

**Accuracy**
![Qwen3-30b-a3b_fp8_acc](
https://github.com/Agoniii/verl/blob/xueh/fp8_pr_images/docs/advance/images/Qwen3-30b-a3b_fp8_acc.png?raw=true)
*grey: BF16 + token-level TIS, red: FP8 rollout + token-level TIS*

Results and observations:
- Rollout & training distribution mismatch is in general higher for MoE
- Rollout correction required even for BF16
- FP8 rollout with token-level TIS aligns with BF16


**Performance**

![Qwen3-30b-a3b_fp8_perf](
https://github.com/Agoniii/verl/blob/xueh/fp8_pr_images/docs/advance/images/Qwen3-30b-a3b_fp8_perf.png?raw=true)
*grey: BF16 + token-level TIS, red: FP8 rollout + token-level TIS​*

Results and observations:
- FP8 rollout : over 35% rollout speedup
- Expecting more perf gain with CUDA 12.9

---

## FP8 End-to-End (Training + Rollout)

FP8 E2E applies FP8 to the entire RL pipeline: forward/backward passes via Transformer Engine, FP8 optimizer states, and FP8 rollout inference via vLLM. This maximizes memory savings and throughput.

### Requirements

- **CUDA 12.9+** (required for block-wise FP8 scaling)
- **Transformer Engine** with block-wise FP8 support
- Environment variable: `NVTE_FP8_BLOCK_SCALING_FP32_SCALES=1`

### Key Configuration

```yaml
# FP8 training via Transformer Engine
actor_rollout_ref.actor.megatron.override_transformer_config:
  fp8: "hybrid"              # FP8 forward + backward; also supports "e4m3"
  fp8_recipe: "blockwise"    # block-wise scaling

# FP8 optimizer
actor_rollout_ref.actor.optim.override_optimizer_config:
  fp8_recipe: "blockwise"

# FP8 rollout inference (vLLM)
actor_rollout_ref.rollout:
  quantization: fp8
```

### Support Matrix

- Megatron training backend (via Megatron-Bridge)
- Verified on Qwen3-30B-A3B and Qwen3-8B
- Block-wise FP8 scaling (`fp8_recipe: "blockwise"`)

### Experiments and Results

#### Qwen3-30B-A3B MoE Model

**Configuration**
- DAPO recipe. AIME24 online validation.
- Megatron + Megatron-Bridge, FP8 async rollout with vLLM
- MoE router in BF16 for both vLLM and Megatron-Core
- Prompt batch size 128, n=16
- Max response length 20K
- Token-level TIS, C=2
- 2\*8*H100, CUDA 12.9

![Qwen3-30b-a3b_fp8_e2e](https://github.com/user-attachments/assets/70fb1396-ec73-40d7-9a43-1d48553c0ad9)
*Orange: BF16, Green: FP8 E2E, Red: FP8 rollout + BF16 training*

Results and observations:
- FP8 E2E achieves comparable accuracy to the BF16 baseline, with the two curves closely aligned throughout training.
- The training/inference precision mismatch (measured by KL divergence) follows the ordering: FP8 rollout-only > FP8 E2E > BF16 E2E. This is expected, as FP8 E2E maintains consistent precision across both training and inference, resulting in lower distribution mismatch than the FP8 rollout-only setting where training remains in BF16.

---

## MXFP8 Training (Blackwell)

MXFP8 is the OCP microscaling FP8 format: E4M3 elements with one shared E8M0 scale per
32-element block, natively accelerated by Blackwell tensor cores. Compared to the
`blockwise` recipe above (1x128 activation / 128x128 weight scaling, designed for Hopper),
MXFP8 uses hardware-decoded block scales and needs no `NVTE_FP8_BLOCK_SCALING_FP32_SCALES`
workaround.

### Requirements

- **Blackwell GPUs** (SM100+). On Hopper, use `fp8_recipe: "blockwise"` as described in
  the FP8 End-to-End section instead — Hopper tensor cores cannot consume MXFP8 block scales.
- **Megatron-Core >= 0.13** and **Transformer Engine >= 2.1**

### Key Configuration

```yaml
# MXFP8 training via Transformer Engine
actor_rollout_ref.actor.megatron.override_transformer_config:
  fp8: "e4m3"                # element format; "hybrid" (e4m3 fwd + e5m2 bwd) also supported
  fp8_recipe: "mxfp8"        # 32-element block scaling

# MXFP8 rollout inference (SGLang)
actor_rollout_ref.rollout:
  name: sglang
  quantization: mxfp8
```

Notes:

- Training requires the Megatron-Bridge model path (`actor_rollout_ref.actor.megatron.use_mbridge=True`,
  the default). The legacy model-building path does not support FP8 recipes and fails loudly.
- Model weights stay in bf16 (`fp8_param` is not supported); only GEMM inputs are cast to
  MXFP8 on the fly, so checkpointing is unchanged.
- verl pads packed sequences to the 32-token block boundaries MXFP8 quantization requires;
  this is automatic once `fp8_recipe: "mxfp8"` is set.

#### SGLang MXFP8 GEMM backend

Which kernel SGLang uses for MXFP8 dense GEMMs depends on the SGLang version:

- **sglang <= 0.5.17** (e.g. 0.5.12) runs them on a generic Triton kernel unless a FlashInfer
  backend is requested. In a 2xB200 measurement, MXFP8 decode on that path was about 2x slower
  than bf16 decode; the FlashInfer CUTLASS backend was about 1.3x slower than bf16 in the same
  setup. verl logs a warning at launch on these versions when no backend is set. Select CUTLASS with

  ```bash
  +actor_rollout_ref.rollout.engine_kwargs.sglang.fp8_gemm_runner_backend=flashinfer_cutlass
  ```

  (`fp8_gemm_runner_backend` is the `ServerArgs` field name; the CLI spelling
  `--fp8-gemm-backend` is not accepted through `engine_kwargs`.)
- **sglang >= 0.5.18** (sgl-project/sglang#33208) removed the Triton path. With no backend set,
  Blackwell uses FlashInfer CuTe-DSL when available, otherwise FlashInfer CUTLASS; Hopper uses
  DeepGEMM. No flag is needed.

Two constraints on this path:

- Every one of these FlashInfer / DeepGEMM backends derives a kernel-specific copy of the
  weight scales at load time (`weight_scale_inv_swizzled` / `weight_scale_inv_deepgemm`), and
  SGLang's weight-update path does not rebuild it. verl therefore registers an MXFP8 refit loader
  (`mxfp8_refit_loader.py`) that re-runs the post-load processing after every weight sync; it reads
  the backend each layer actually resolved to, so it covers the version-dependent defaults above.
  Without it the engine keeps serving load-time scales against freshly synced weights and generates
  garbage from step 0. `flashinfer_trtllm` shuffles the weight tensor itself in place at load and is
  rejected by the loader.
- Do not set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for an SGLang rollout: verl
  launches SGLang with `enable_memory_saver=True`, and `torch_memory_saver` refuses that
  allocator mode, killing the server in `load_model`. The vLLM path tolerates the variable.

Neither engine showed an MXFP8 rollout throughput gain over bf16 in these runs (SGLang 0.5.12 and
vLLM 0.24.0); the measurements are in the PR description.

#### Guard rails for a quantized rollout

Two failures met during the B200 validation were silent — the run kept going with exit code 0
while every sample was garbage (a quantized `lm_head` producing `nan` logits; an engine serving
stale kernel scale layouts after a weight sync). verl now fails loudly in both cases:

- **Quantized-rollout sentinel** (trainer side, both trainers). When `rollout.quantization` is set,
  the step metrics are checked after each step: non-finite `training/rollout_probs_diff_mean`,
  `rollout_corr/kl` above 1.0 (typical values are 0.001–0.03), or every response hitting the
  length cap on two consecutive steps raise a `RuntimeError` naming the likely causes. Disable
  with `VERL_QUANT_SENTINEL=0`; tune with `VERL_QUANT_SENTINEL_KL_MAX` / `VERL_QUANT_SENTINEL_CLIP_STEPS`.
- **MXFP8 refit self-check** (engine side, vLLM and SGLang). After every weight sync the smallest
  MXFP8 linear layer's own quantized GEMM is run on a small random input and compared with a
  dequantized bf16 reference of the canonical weight and scale; a stale or mis-laid-out scale
  shows up as O(1) relative error and raises. On SGLang one local expert of the smallest MXFP8 MoE
  layer is probed the same way (every probe row routed to it, compared with the gated-MLP reference on
  the dequantized `w13` / `w2`; tolerance `VERL_MXFP8_REFIT_CHECK_MOE_TOL`, default = the linear one);
  the expert probe is skipped under expert parallelism and under TP without `reduce_results`, which
  the log states. On vLLM the same expert probe runs on `ModelOptMxFp8FusedMoE` layers by calling the
  weight holder's `forward_modular` / `forward_monolithic` directly (skipped under expert or data
  parallelism and for non-SiLU gates). vLLM 0.24's ModelOpt MXFP8 MoE method processes its weights
  only once per layer and returns early afterwards; verl's patched hook clears that flag on every
  refit so the kernel layout is re-derived from the synced scales. On SGLang MoE layers the loader additionally checks,
  before re-deriving the kernel layout, that the sync wrote every staged expert scale (the staging
  buffer is pre-filled with the UE8M0 NaN code `0xFF`): experts whose HF names miss the sync-side
  rule would otherwise arrive as a scale-less bf16 cast in the fp8 buffer. The kernel-vs-reference
  probe verifies the kernel's *layout*, not that the sync delivered the right scales — the audit
  below covers that. Disable both with `VERL_MXFP8_REFIT_CHECK=0`; the probe tolerance (default
  0.25) is `VERL_MXFP8_REFIT_CHECK_TOL`.
- **Quantized-layer audit** (trainer worker, Megatron engine). "Matched" train/rollout quantization
  presumes both sides quantize the same layers, but training decides implicitly (TE linear modules
  inside `fp8_autocast`) and rollout decides by name blacklist (`ignored_layers`). At the first weight
  sync after a training step, verl reads which decoder layers actually ran fp8 GEMMs (TE's fp8 weight
  workspaces) and compares them, per synced parameter name, with the rollout side. The audit wraps the
  training engine's weight export (`get_per_tensor_param`), which every sync route shares - the
  colocated worker, the checkpoint engine and the server-replica path - and produces its verdict when
  the weight stream ends. What "the rollout
  side" is depends on the engine and every message says which: on vLLM the engine is asked directly
  (`collective_rpc` into the worker, which resolves each HF name onto its live parameter and reports
  its dtype); on SGLang there is no return channel, so the trainer compares against the weight-sync
  rule and the loader's sync-vs-engine dtype check (next bullet) covers the sync-to-engine half. It logs
  each disagreement (e.g. `first_last_layers_bf16` without the matching rollout regex, a router the
  name patterns miss, `lm_head` left in the quantized set, Mixtral's `w1/w2/w3` experts that the
  SGLang sync-time include list does not match). On SGLang the rule evaluated is the sync-time
  include/exclude rule in `verl/utils/fp8_utils.py`; a name it misses is shipped unquantized even
  when the engine built that layer as fp8. Fused expert tensors as `transformers >= 5` saves them
  (`mlp.experts.gate_up_proj`, no `.weight` suffix) are judged too. The signal survives
  `param_offload=True` (verl marks each module before it drops the TE workspace cache on offload);
  when there is no signal at all — `disable_parameter_transpose_cache=True` makes TE skip the cache —
  the audit warns once that it cannot run instead of staying silent. `VERL_QUANT_LAYER_AUDIT=raise`
  turns the report into an error, `=0` disables it.
- **Sync-vs-engine dtype check** (SGLang loader). Before a sync is written into the engine, every
  incoming linear weight's dtype is compared with the dtype of the engine parameter that will receive it
  (HF names are mapped onto SGLang's fused modules: `q_proj` → `qkv_proj`, `gate_proj` → `gate_up_proj`,
  per-expert names → the fused `w13` / `w2` tensors). A bf16 tensor headed for an fp8 parameter, or fp8
  data headed for a bf16 one, is refused by name instead of being cast silently by `load_weights` —
  e.g. Mixtral's `experts.N.w1/w2/w3`, which the sync rule does not match while the engine built them as
  fp8. Disabled with `VERL_MXFP8_REFIT_CHECK=0`.
- **MoE experts on SGLang.** SGLang's MXFP8 MoE method rewrites the expert scales in place at
  load (swizzled on the Triton MoE runner, packed on DeepGEMM), so the refit loader stages them
  back to the canonical `[E, N, K/32]` layout for `load_weights` and re-derives the kernel layout
  afterwards into the storage the CUDA graph captured. This path is covered by CPU tests and has
  not yet been validated on hardware; see the PR description for the MoE validation status.

### MXFP8 Rollout and Train-Inference Consistency

With `quantization: mxfp8`, the rollout engine (SGLang or vLLM) is launched in MXFP8 mode
against the bf16 checkpoint (via a `quantization_config` override, no offline conversion
needed), and weight sync quantizes the bf16 actor weights to MXFP8 on the fly.

For vLLM, the config maps to `ModelOptMxFp8Config` (weight `fp8_e4m3fn` + `uint8` UE8M0
`weight_scale`), refits reuse the same pristine-layout staging cycle as the blockwise FP8
path, and vLLM's Marlin/emulation fallbacks allow serving MXFP8 weights on pre-Blackwell
GPUs (SM80+) — the served weight grid is still produced by TE's quantizer, so
train-inference weight consistency is preserved regardless of the serving kernel.

The staging cycle decides per layer whether a refit must go through stage → load → reprocess
by comparing the live parameters with the checkpoint layout recorded at load. A kernel that
hands back a *rewritten copy with the checkpoint's shape and dtype* is invisible to that
comparison. FlashInfer TRT-LLM's MXFP8 MoE prep (`ModelOptMxFp8FusedMoE` on Blackwell: W13→W31
swap, gate/up row interleave, tile shuffle of weights and scales) is such a kernel, so verl's
patched `replace_parameter` records the rewrite when it sees it and the layer is staged on every
refit. Measured on 1×B200 (vLLM 0.24, Qwen3-MoE tiny, TP1): the MoE expert probe read
rel err 1.739 at the first sync before this record existed and 0.052 with it, same weights
and inputs; the dense probe was 0.026 both times.

**Known issue, bf16 MoE rollout on Blackwell (independent of MXFP8; tracked in verl-project/verl#7978).** With
`quantization` unset, vLLM 0.24 auto-selects the FlashInfer TRT-LLM bf16 MoE backend on SM100,
whose BlockMajorK layout turns `w13_weight` / `w2_weight` into 4-D tensors. verl's bf16 weight
sync feeds `model.load_weights` per-expert 2-D tensors, which the loader can no longer index
(`shard_dim=0 is not a valid data dimension for a 3D tensor`). The fix routes the standard sync
through vLLM's own layerwise reload lifecycle (wengeezhang/verl#4, awaiting hardware validation);
without it, pin the layout-preserving backend:
`+actor_rollout_ref.rollout.engine_kwargs.vllm.moe_backend=triton`. The same-shape staging fix
described above is proposed upstream as verl-project/verl#7986.

The weight-sync quantization deliberately uses **TransformerEngine's `MXFP8Quantizer`** —
the same quantizer the trainer's FP8 GEMMs apply to weights — so the rollout engine serves
exactly the weight grid the training forward pass saw. An independent quantization kernel
can round E8M0 scales differently at block boundaries and reintroduce train-inference
mismatch. Residual mismatch (activation quantization kernels and GEMM implementations still
differ between TE and SGLang) is small; pairing with token-level TIS is recommended,
as with the blockwise FP8 E2E recipe.

Layer skipping follows the same rules as FP8 rollout (`ignored_layers`,
`modules_to_not_convert`, or the `SGLANG_FP8_IGNORED_LAYERS` env var). Layers whose
last weight dim is not a multiple of 32 cannot be MXFP8-quantized and must be excluded
this way — e.g. vision towers of VLMs (`SGLANG_FP8_IGNORED_LAYERS=visual`); weight sync
fails with an actionable error if such a layer is selected. If you enable
`first_last_layers_bf16` on the training side, keep the two sides consistent by excluding
the same layers from rollout quantization, e.g. for a 36-layer model with the first and
last layer in bf16:

```json
{
  "quantization_config": {
    "ignored_layers": ["re:model\\.layers\\.(0|35)\\..*"]
  }
}
```

---

## Citation

For more extensive experiments, ablation studies, and analysis on FP8 reinforcement learning, please refer to our technical report:

```bibtex
@article{qiu2026fp8rl,
  title={FP8-RL: A Practical and Stable Low-Precision Stack for LLM Reinforcement Learning},
  author={Qiu, Zhaopeng and Yu, Shuang and Zhang, Jingqi and Zhang, Shuai and Huang, Xue and Yang, Jingyi and Lai, Junjie},
  journal={arXiv preprint arXiv:2601.18150},
  year={2026},
  url={https://arxiv.org/abs/2601.18150}
}
```
