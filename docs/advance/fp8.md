# FP8 rollout for verl

Last updated: 12/4/2025

This document introduces FP8 rollout in verl.


We monkey patch several vLLM functions to enable FP8 rollout for reinforcement learning:

1. **Quantize weights**: Quantize model weights on-the-fly from higher-precision formats to FP8.
2. **Process weights after loading**: For vLLM, we replace the `vllm.model_executor.layers.quantization.fp8.Fp8LinearMethod.process_weights_after_loading` function to handle weight processing after quantization. For SGLang, this patch is not needed as it natively supports loading quantized weights.


## Support Matrix
- FP8 blockwise quantization for rollout
  - Used in Deepseek,
which is 1x128 quantization for activations and 128x128 quantization for model weights
- Dense models and MoE models
- Async rollout interfaces
- vLLM 0.10.x & vLLM 0.11 & SGlang 0.5.5
- FSDP and Megatron training backends

## Experiments and Outcomes
### Qwen3-8B-Base Dense Model

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

### Qwen3-30B-A3B-Base MoE Model

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

Please refer to `recipe/dapo/run_dapo_qwen3_moe_30b_vllm_fp8_rollout.sh`

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

## Usage

FP8 can be enabled in the config file `verl/trainer/config/ppo_megatron_trainer.yaml`:

```
  rollout:
    quantization: "fp8"
```

Or it can be enabled by command line:
- `actor_rollout_ref.rollout.quantization=fp8`

Please refer to `recipe/dapo/run_dapo_qwen3_moe_30b_vllm_fp8_rollout.sh`
