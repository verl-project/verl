# Guide to Using MTP in SFT/RL Training and Inference

**Author**: `https://github.com/meituan-search`

Last updated: 08/11/2026

## 1. Scope of Support

Currently, RL training can be performed on mimo-7B-RL, Qwen-next, and Deepseek series models based on the MTP architecture. SFT also supports NVIDIA Nemotron 3 Super through Megatron-Core's native `HybridModel` MTP implementation. The support rules for training and inference engines are as follows:

- **Training Engine**: Only supports the `mbridge/Megatron-Bridge + megatron` combination; other training engines are not compatible at this time;

- **Inference Engine**: Compatible with all engines, but the model must be in the corresponding engine's compatibility list;

- **Dependency Versions**:

    - NVIDIA Nemotron 3 Super native `HybridModel` MTP requires this exact dependency snapshot:

        - Megatron Bridge commit [`1f12931e2f34ec26f578a4cffe15adc06f71a5a2`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/commit/1f12931e2f34ec26f578a4cffe15adc06f71a5a2);

        - Megatron Core **0.19.0** at commit [`cd4afffa648426a959dc7cb1e24b5ce7d0c3ff54`](https://github.com/NVIDIA/Megatron-LM/commit/cd4afffa648426a959dc7cb1e24b5ce7d0c3ff54). A Git checkout reports this as `0.19.0+cd4afff`. Pin the commit, rather than relying on the `0.19.0` version number alone;

    - mbridge: Apply the patches and review suggestions from PR: [#62](https://github.com/ISEEKYAN/mbridge/pull/62) (Already merged into the main branch);

    - Megatron-Bridge: Apply the patches and review suggestions from PR if you want to try out mimo-7B-RL: [#2387](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2387) (will be merged into the main branch in the future);

    - megatron: Use the latest dev version (commit: [23e092f41ec8bc659020e401ddac9576c1cfed7e](https://github.com/NVIDIA/Megatron-LM/tree/23e092f41ec8bc659020e401ddac9576c1cfed7e)), which supports MTP + CP training methods. If you additionally enable `recompute_granularity=full`, use a dev commit that includes [#3457](https://github.com/NVIDIA/Megatron-LM/pull/3457) (`ffd66a3e6`, 2026-06-03) — the commit pinned above predates it by about six months. #3457 threads `padding_mask` through `MultiTokenPredictionLayer._checkpointed_forward`; without it, `MultiTokenPredictionLayer.forward` passes a keyword the method does not declare and training raises a `TypeError` on the first step. Released `megatron-core` 0.18.0 and 0.18.2 do not carry the fix either (tracked as [#4933](https://github.com/NVIDIA/Megatron-LM/issues/4933)).
    
    - sglang: Use the specified branch: [https://github.com/ArronHZG/sglang/tree/fix_mtp_update_weights_from_tensor](https://github.com/ArronHZG/sglang/tree/fix_mtp_update_weights_from_tensor), [PR](https://github.com/sgl-project/sglang/pull/17870) , which fix the MTP update weights from tensor OOM issue.

## 2. MTP Training Configuration (Core Parameters)

The MTP training process can be flexibly controlled through the following configurations. All configurations are based on the `actor_rollout_ref.model.mtp` prefix:

| Configuration Scenario | Core Parameters                                                                                                                                                                                                                                                                                               | Description                                             |
|------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------|
| Load MTP Parameters Only | `enable=True`                                                                                                                                                                                                                                                                                              | VRAM usage will increase, but the exported parameters include the MTP module and can be directly used for online deployment. This mode is not supported for native Megatron-Core `HybridModel` MTP; set `enable_train=True` or disable MTP.              |
| Full-Parameter MTP Training | `enable=True`<br>`enable_train=True`<br>`mtp_loss_scaling_factor=0.1`                                                                                                                                                                                                                              | MTP Loss will apply to all model parameters                            |
| Detached MTP Auxiliary Training | `enable=True`<br>`enable_train=True`<br>`detach_encoder=True`                                                                                                                                                                                                                                      | Detach the hidden state consumed by MTP so the auxiliary MTP loss updates only the MTP module. The normal SFT loss still updates the base model. |
| MTP Accelerated Rollout | 1. vLLM configuration:<br>`enable=True`<br>`enable_rollout=True`<br>`method="mtp"`<br>`num_speculative_tokens=1`<br>2. SGLang configuration:<br>`enable=True`<br>`enable_rollout=True`<br>`speculative_algorithm="EAGLE"`<br>`speculative_num_steps=2`<br>`speculative_eagle_topk=2`<br>`speculative_num_draft_tokens=4` | Achieve inference acceleration during the Rollout phase based on MTP                      |

For native Megatron-Core `HybridModel` models, `enable=True` requires `enable_train=True`. The load-without-training combination is rejected because this path uses Megatron-Core's native MTP implementation instead of verl's legacy GPT MTP patches.

`detach_encoder=True` applies only to the auxiliary MTP branch. It prevents MTP-loss gradients from flowing back through the base-model hidden states, but it does not freeze the base model: the ordinary SFT language-model loss continues to train it.

## 3. Experimental Results

The experiment was conducted as follows:

* model = mimo-7B-math
* max_response_length = 8k

Experiment chart:

![fully_async_policy_revenue](
https://github.com/ArronHZG/verl-community/blob/main/docs/mimo-7b-mtp.png?raw=true)

The wandb link for the graph: [wandb](https://wandb.ai/hou-zg-meituan/mimo-7b-sft-mtp?nw=nwuserhouzg)

**Scenarios with No Significant Effect**

The following configurations will not have a noticeable impact on training results:

1. The base model does not carry MTP parameters;

2. The base model carries MTP parameters, but the MTP module is not trained;

3. The base model carries MTP parameters and trains MTP, with `mtp_loss_scaling_factor=0`;

4. The base model carries MTP parameters, trains MTP and detaches the encoder, with `mtp_loss_scaling_factor=0.1`.

**Scenarios with Significant Effect**

Only the following configuration will have a noticeable impact on training results:

- The base model carries MTP parameters, MTP Loss applies to all model parameters, and `mtp_loss_scaling_factor=0.1`.

**Recommended Training Method**

It is recommended to adopt the `detach_encoder=True` approach for MTP training. This isolates auxiliary MTP gradients; it does not freeze base-model training from the main loss.

## 4. Performance Notes for MTP in Rollout Inference

Enabling MTP improves the rollout acceptance rate by around 14%. However, on H20 GPUs, overall throughput does not increase and even decreases slightly.

![spec_log](
https://github.com/ArronHZG/verl-community/blob/main/docs/spec_log.png?raw=true)

The effectiveness of MTP-accelerated Rollout is significantly affected by **model size** and **inference hardware**. Key reference information is as follows:

**Hardware Tensor Core Performance**

| Hardware Model | FP16 Performance (TFLOPS) |
|----------------|---------------------------|
| H20  | 148            |
| H800 | 1,671          |
| H200 | 1,979          |

**Measured Performance and Recommendations**

Taking the mimo-7B model deployed separately on H20 hardware using SGLang as an example: After enabling MTP speculative decoding, the Rollout throughput decreases by approximately 50%.

- Current priority recommendation: Do not enable MTP acceleration during the inference phase for now;

- Future planning: Further optimization of the speculative logic in the Rollout phase will be conducted to improve throughput performance.

## 5. SFT training

SFT training with MTP is supported, using the same MTP training configuration as RL training. The relevant prefix is `model.mtp` for SFT and `actor_rollout_ref.model.mtp` for RL.

Example launchers:

- `examples/sft/gsm8k/run_mimo_7b_mtp_megatron.sh` for MiMo-7B;

- `examples/sft/gsm8k/run_nemotron_3_super_megatron.sh` for NVIDIA Nemotron 3 Super with native Megatron-Core `HybridModel` MTP. This launcher uses Megatron Bridge (`engine.use_mbridge=True`, `engine.vanilla_mbridge=False`) and enables MTP training.

**SFT result**

The experiment was conducted using following data:
- model = mimo-7B-math
- dataset = gsm8k

The result: [wandb link](https://wandb.ai/hou-zg-meituan/mimo-7b-sft-mtp?nw=nwuserhouzg)

The presence of mtp layer has limited effect on main loss. However, when MTP layer is detached, the mtp_loss converges to a higher value.
