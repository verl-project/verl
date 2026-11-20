# FP8 RL in verl

Last updated: 11/02/2026

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

The rollout engine starts from a compressed-tensors `quantization_config` with dummy weights, and the trainer quantizes every weight sync to that config with the compressed-tensors compressors (`verl/utils/qat/compressed_tensors.py`). On the vLLM side we monkey patch `process_weights_after_loading` of the compressed-tensors FP8 linear/MoE methods so refits are written back into the live parameters (`verl/utils/vllm/vllm_fp8_utils.py`).

### Support Matrix

- FP8 blockwise quantization for rollout (compressed-tensors preset `FP8_BLOCK`)
  - Used in Deepseek, which is 1x128 quantization for activations and 128x128 quantization for model weights
- Dense models and MoE models
- Async rollout interfaces
- vLLM or SGLang rollout (SGLang validated on Hopper only) with the Megatron training backend; TRT-LLM and FSDP are not supported

### Usage

Enable in config file:

```yaml
rollout:
  quantization:
    linear: FP8_BLOCK  # dense linear layers (attention, dense MLP, shared experts)
    moe: FP8_BLOCK     # routed experts
    ignore: []         # extra "re:" patterns kept in BF16
```

Or via command line:

```bash
actor_rollout_ref.rollout.quantization.linear=FP8_BLOCK \
actor_rollout_ref.rollout.quantization.moe=FP8_BLOCK
```

Routers, `lm_head`, embeddings, vision towers and MTP layers stay in BF16 by default, as do layers whose weight shape is not a multiple of the 128x128 block. To provide a complete compressed-tensors `quantization_config` instead, set `actor_rollout_ref.rollout.quantization.config_file=/path/to/quantization_config.json` (exclusive with `linear` / `moe` / `ignore`).

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
- Environment variable: `NVTE_FP8_BLOCK_SCALING_FP32_SCALES=1`, on **Hopper only**. Do not set it on Blackwell (SM100+): there Transformer Engine runs block-wise scaling through MXFP8 GEMMs, whose E8M0 scales hold only an exponent, so it requires power-of-two block scales and stops at the first FP8 quantization with `Assertion failed: pow2_scale ... requires using power of two scaling factors` when the variable is set (see [#6172](https://github.com/verl-project/verl/issues/6172)). Leaving it unset gives power-of-two scales, the Transformer Engine default.

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
actor_rollout_ref.rollout.quantization:
  linear: FP8_BLOCK
  moe: FP8_BLOCK
```

### FP8 KV Cache (Optional)

The rollout KV cache can also be stored in FP8 by passing vLLM's `kv_cache_dtype` through `engine_kwargs`:

```bash
+actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_dtype=fp8_e4m3
```

- Only the rollout side changes. Training-side core attention stays in BF16, so the extra train/rollout mismatch is corrected by TIS like the rest of the FP8 rollout error.
- The q/k/v scales are uncalibrated (1.0): verl does not export KV cache scales, so the values are cast to FP8 directly. On Hopper, vLLM's FlashAttention-3 backend then runs attention in FP8 and quantizes Q as well.
- Combined with `quantization.linear/moe=FP8_BLOCK`, this needs [#8483](https://github.com/verl-project/verl/pull/8483). Without it, the attention scales keep the random values from the dummy load and the rollout generates garbage.
- Validated with vLLM only.

### Support Matrix

- Megatron training backend (via Megatron-Bridge)
- Verified on Qwen3-30B-A3B and Qwen3-8B (H100); Qwen3-30B-A3B also on H20
- Block-wise FP8 scaling (`fp8_recipe: "blockwise"`)
- Optional FP8 KV cache in vLLM rollout

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

#### Qwen3-30B-A3B MoE Model on H20, with FP8 KV Cache

**Configuration**
- Same recipe as above (DAPO-Math-17k, prompt batch size 128, n=16, max prompt / response length 2K / 20K, token-level TIS with C=2, dynamic sampling), 100 steps
- AIME24 online validation every 10 steps (32 samples per problem, temperature 0.6)
- Megatron + Megatron-Bridge, vLLM 0.29 rollout, Transformer Engine 2.16
- 1\*8*H20 per run, CUDA 13.0
- Three runs: BF16 training + BF16 rollout; FP8 E2E; FP8 E2E + FP8 KV cache (`kv_cache_dtype=fp8_e4m3`, scales 1.0)

| | BF16 | FP8 E2E | FP8 E2E + FP8 KV cache |
|---|---|---|---|
| AIME24 at step 0 / 50 / 100 | 5.4 / 20.1 / 23.9 | 4.4 / 19.4 / 22.5 | 4.6 / 19.0 / 22.0 |
| `rollout_corr/k3_kl`, steps 1–10 / last 20 steps | 0.00127 / 0.00065 | 0.00264 / 0.00131 | 0.00534 / 0.00290 |
| Step time (s) | 396 | 376 | 357 |
| Rollout generation (s) | 182 | 164 | 143 |
| Actor update (s) | 151 | 146 | 147 |
| Peak training memory reserved (GB) | 110 | 127 | 127 |

Times are steady-state means over steps without validation or checkpointing. Generation time includes dynamic-sampling refills.

![Qwen3-30B-A3B on H20, training curves](https://github.com/user-attachments/assets/202130be-49db-47b9-bcc1-ac3a1c5a2745)

![Qwen3-30B-A3B on H20, step time](https://github.com/user-attachments/assets/b97a8ecd-8cb6-47ec-8c00-620493c7e635)

Results and observations:
- All three runs train stably for 100 steps. Accuracy, reward, entropy, response length and grad norm overlap; the AIME24 gap at step 100 is within the evaluation noise of about ±2–3 points.
- FP8 E2E roughly doubles the train/rollout mismatch of BF16, and the FP8 KV cache doubles it again (about 4.5× BF16). It does not grow over training. In a single-step probe, BF16 rollout with an FP8 KV cache alone gives a larger mismatch (0.0050) than FP8 E2E without it (0.0029), so the uncalibrated KV scales are the larger error source.
- The FP8 KV cache cuts rollout generation by another 13% over FP8 E2E and the whole step by 5% (10% versus BF16). Training time and memory do not change. Responses here average about 1K tokens; longer responses spend more time reading the KV cache, so the gain should be larger.

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
