# Guide to Using MTP in SFT/RL Training and Inference

**Author**: `https://github.com/meituan-search`

Last updated: 01/30/2026

# 1. Scope of Support

Currently, RL training can be performed on mimo-7B-RL, Qwen-next, and Deepseek series models based on the MTP architecture. The support rules for training and inference engines are as follows:

- **Training Engine**: Only supports the `mbridge + megatron` combination; other training engines are not compatible at this time;

- **Inference Engine**: Compatible with all engines, but the model must be in the corresponding engine's compatibility list;

- **Dependency Versions**:

    - mbridge: Use the specified branch: [https://github.com/ArronHZG/mbridge/tree/feature/verl_mtp](https://github.com/ArronHZG/mbridge/tree/feature/verl_mtp) (will be merged into the main branch in the future);

    - megatron: Use the latest dev version (commit: [23e092f41ec8bc659020e401ddac9576c1cfed7e](https://github.com/NVIDIA/Megatron-LM/tree/23e092f41ec8bc659020e401ddac9576c1cfed7e)), which supports MTP + CP training methods.
    
    - sglang: Use the specified branch: [https://github.com/ArronHZG/sglang/tree/fix_mtp_update_weights_from_tensor](https://github.com/ArronHZG/sglang/tree/fix_mtp_update_weights_from_tensor), [PR](https://github.com/sgl-project/sglang/pull/17870) , which fix the MTP update weights from tensor OOM issue.

# 2. MTP Training Configuration (Core Parameters)

The MTP training process can be flexibly controlled through the following configurations. All configurations are based on the `actor_rollout_ref.model.mtp` prefix:

| Configuration Scenario | Core Parameters                                                                                                                                                                                                                                                                                               | Description                                             |
|------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------|
| Load MTP Parameters Only | `enable=True`                                                                                                                                                                                                                                                                                              | VRAM usage will increase, but the exported parameters include the MTP module and can be directly used for online deployment              |
| Full-Parameter MTP Training | `enable=True`<br>`enable_train=True`<br>`mtp_loss_scaling_factor=0.1`                                                                                                                                                                                                                              | MTP Loss will apply to all model parameters                            |
| MTP Parameter-Only Training | `enable=True`<br>`enable_train=True`<br>`detach_encoder=True`                                                                                                                                                                                                                                      | Freeze the Encoder layer, update only MTP module parameters, MTP Loss applies only to MTP parameters |
| MTP Accelerated Rollout | 1. vLLM configuration:<br>`enable=True`<br>`enable_rollout=True`<br>`method="mtp"`<br>`num_speculative_tokens=1`<br>2. SGLang configuration:<br>`enable=True`<br>`enable_rollout=True`<br>`speculative_algorithm="EAGLE"`<br>`speculative_num_steps=2`<br>`speculative_eagle_topk=2`<br>`speculative_num_draft_tokens=4` | Achieve inference acceleration during the Rollout phase based on MTP                      |

# 3. Experimental Results

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

It is recommended to adopt the `detach_encoder=True` approach for MTP training.

# 4. Performance Notes for MTP in Rollout Inference

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

# 5. SFT training

The SFT training with MTP is supported, using the same MTP training configuration as RL training.

An example configuration for running SFT can be found in `examples/sft/gsm8k/run_mimo_megatron_mtp.sh`

**SFT result**

The experiment was conducted using following data:
- model = mimo-7B-math
- dataset = gsm8k

The result: [wandb link](https://wandb.ai/hou-zg-meituan/mimo-7b-sft-mtp?nw=nwuserhouzg)

The presence of mtp layer has limited effect on main loss. However, when MTP layer is detached, the mtp_loss converges to a higher value.

