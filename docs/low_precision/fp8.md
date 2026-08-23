# FP8 RL in verl

Last updated: 08/22/2026

verl supports two FP8 modes for accelerating RL training:

| Mode | Training Precision | Rollout Precision |
|------|-------------------|-------------------|
| **FP8 Rollout Only** | BF16 | FP8 |
| **FP8 End-to-End** | FP8 (Megatron) | FP8 (vLLM) |

> [!TIP]
> For ready-to-run scripts, see the [low-precision recipe directory](https://github.com/verl-project/verl-recipe/low_precision).

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

### MXFP8 Rollout and Train-Inference Consistency

With `quantization: mxfp8`, SGLang is launched in MXFP8 mode against the bf16 checkpoint
(via a `quantization_config` override, no offline conversion needed), and weight sync
quantizes the bf16 actor weights to MXFP8 on the fly.

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
