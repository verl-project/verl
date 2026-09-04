# Guide to Using MTP in SFT/RL Training and Inference

**Author**: `https://github.com/meituan-search`

Last updated: 09/03/2026

## 1. Scope of Support

Currently, RL training can be performed on mimo-7B-RL, Qwen-next, and Deepseek series models based on the MTP architecture. SFT also supports NVIDIA Nemotron 3 Super and Nemotron 3.5 Lightning through Megatron-Core's native `HybridModel` MTP implementation. Nemotron 3.5 Lightning additionally has a GRPO recipe combining vLLM rollout, R3 router replay, and one-token MTP speculation. The support rules for training and inference engines are as follows:

- **Training Engine**: Only supports the `mbridge/Megatron-Bridge + megatron` combination; other training engines are not compatible at this time;

- **Inference Engine**: Generic MTP is available when the model is supported by the selected engine. The Nemotron 3.5 Lightning GRPO recipe currently supports vLLM only;

- **Dependency Versions**:

    - NVIDIA Nemotron 3 Super native `HybridModel` MTP requires this exact dependency snapshot:

        - Megatron Bridge commit [`1f12931e2f34ec26f578a4cffe15adc06f71a5a2`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/commit/1f12931e2f34ec26f578a4cffe15adc06f71a5a2);

        - Megatron Core **0.19.0** at commit [`cd4afffa648426a959dc7cb1e24b5ce7d0c3ff54`](https://github.com/NVIDIA/Megatron-LM/commit/cd4afffa648426a959dc7cb1e24b5ce7d0c3ff54). A Git checkout reports this as `0.19.0+cd4afff`. Pin the commit, rather than relying on the `0.19.0` version number alone;

    - NVIDIA Nemotron 3.5 Lightning uses the reproducible image in `docker/Dockerfile.nemotron_3_5_lightning`, which pins:

        - Megatron Bridge commit [`c93251151adeeadbae3ff2a2bf5ee7a1c34cff01`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/commit/c93251151adeeadbae3ff2a2bf5ee7a1c34cff01) and its Megatron Core submodule at [`cd4afffa648426a959dc7cb1e24b5ce7d0c3ff54`](https://github.com/NVIDIA/Megatron-LM/commit/cd4afffa648426a959dc7cb1e24b5ce7d0c3ff54);

        - Transformers **5.10.4** and vLLM **0.27.1** at [`6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`](https://github.com/vllm-project/vllm/commit/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac).

      The default `main` uv environment currently uses older Megatron Core, Megatron Bridge, and vLLM versions that do not provide this complete native-HybridModel/R3+MTP path. Use the pinned image for this recipe. From the repository root, build it with:

      ```bash
      docker build --build-arg VERL_COMMIT="$(git rev-parse HEAD)" \
        -f docker/Dockerfile.nemotron_3_5_lightning \
        -t verl-nemotron-3-5-lightning .
      ```

      `VERL_COMMIT` records the source revision in the image metadata; the Docker build uses the current build context, so build from a clean checkout of that revision.

      For an immutable Lightning run, download the model and GRPO datasets at the validated revisions before launching:

      ```bash
      hf download nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 \
        --revision d468880b6ad3c6e0d21377ce7242adaea4cc884d \
        --local-dir "$HOME/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"
      hf download BytedTsinghua-SIA/DAPO-Math-17k data/dapo-math-17k.parquet \
        --repo-type dataset --revision 65877096c24ffa7abc4e4fa5edb95cf3413a5674 \
        --local-dir "$HOME/verl/data/DAPO-Math-17k"
      hf download BytedTsinghua-SIA/AIME-2024 data/aime-2024.parquet \
        --repo-type dataset --revision aa49075e24ad594b79fdf0bdcefa735c2181be67 \
        --local-dir "$HOME/verl/data/AIME-2024"
      ```

      Set `MODEL_PATH` to the downloaded model directory. The GRPO launcher defaults match the two dataset paths above. The current `main` `HFModelConfig` does not expose a Hub revision field, so a local snapshot is required for exact model reproduction.

    - MiMo/mbridge: Apply the patches and review suggestions from PR: [#62](https://github.com/ISEEKYAN/mbridge/pull/62) (Already merged into the main branch);

    - MiMo/Megatron-Bridge: Apply the patches and review suggestions from PR if you want to try out mimo-7B-RL: [#2387](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2387) (will be merged into the main branch in the future);

    - Existing MiMo MTP+CP examples: Use Megatron commit [23e092f41ec8bc659020e401ddac9576c1cfed7e](https://github.com/NVIDIA/Megatron-LM/tree/23e092f41ec8bc659020e401ddac9576c1cfed7e). If you additionally enable `recompute_granularity=full`, use a dev commit that includes [#3457](https://github.com/NVIDIA/Megatron-LM/pull/3457) (`ffd66a3e6`, 2026-06-03) — the commit pinned above predates it by about six months. #3457 threads `padding_mask` through `MultiTokenPredictionLayer._checkpointed_forward`; without it, `MultiTokenPredictionLayer.forward` passes a keyword the method does not declare and training raises a `TypeError` on the first step. Released `megatron-core` 0.18.0 and 0.18.2 do not carry the fix either (tracked as [#4933](https://github.com/NVIDIA/Megatron-LM/issues/4933)).
    
    - MiMo/SGLang rollout: Use the specified branch: [https://github.com/ArronHZG/sglang/tree/fix_mtp_update_weights_from_tensor](https://github.com/ArronHZG/sglang/tree/fix_mtp_update_weights_from_tensor), [PR](https://github.com/sgl-project/sglang/pull/17870), which fixes the MTP update-weights-from-tensor OOM issue.

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

When vLLM rollout uses R3 router replay together with MTP speculation, vLLM **0.26.0 or newer** is required so replay excludes draft-model router decisions. The pinned Lightning image uses vLLM 0.27.1. R3 also requires `actor_rollout_ref.rollout.enable_rollout_routing_replay=True` and context parallelism 1 for the validated Lightning topology.

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

- `examples/sft/gsm8k/run_nemotron_3_5_lightning_megatron.sh` for full-parameter NVIDIA Nemotron 3.5 Lightning SFT with native MTP.

**SFT result**

The experiment was conducted using following data:
- model = mimo-7B-math
- dataset = gsm8k

The result: [wandb link](https://wandb.ai/hou-zg-meituan/mimo-7b-sft-mtp?nw=nwuserhouzg)

The presence of mtp layer has limited effect on main loss. However, when MTP layer is detached, the mtp_loss converges to a higher value.

## 6. Nemotron 3.5 Lightning GRPO

Use `examples/grpo_trainer/run_nemotron_3_5_lightning_30b_a3b_megatron.sh` for the 2-node x 8-H100 topology. Its defaults enable R3 router replay and one-token MTP speculation, use raw rollout log probabilities, and disable dynamic batching and router/permutation fusion to keep Megatron and vLLM routing reproducible. These settings completed a four-hour validation on the v0.7 backport; run a GPU smoke test before treating a newly rebased `main` revision as hardware-validated. The recipe supports vLLM only and requires the pinned container described above.
