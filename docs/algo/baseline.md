# Algorithm Baselines

Last updated: 06/18/2025.

## Math related datasets

### GSM8k

Assuming GSM8k/math dataset is preprocessed via:

```bash
python3 examples/data_preprocess/*.py
```

Refer to the table below to reproduce RL training from different pre-trained checkpoints. Below is the performance on the GSM8k dataset if not specified otherwise. More comprehensive benchmark results areavailable in the recipe folder.

| Hardware   | Model                            | Method          | Test score   | Details                                                                                                                                                                                                                       |
| ---------- | -------------------------------- | --------------- | ------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| NVIDIA GPU | google/gemma-2-2b-it             | hf checkpoint   | 23.9         | [Huggingface](https://huggingface.co/google/gemma-2-2b-it#benchmark-results)                                                                                                                                                  |
| NVIDIA GPU | google/gemma-2-2b-it             | SFT             | 52.06        | [command and logs](https://github.com/eric-haibin-lin/verl-data/blob/experiments/gsm8k/gemma-2-2b-it-sft-0.411.log)                                                                                                           |
| NVIDIA GPU | google/gemma-2-2b-it             | SFT + PPO       | 64.02        | [command and logs](https://github.com/eric-haibin-lin/verl-data/blob/experiments/gsm8k/gemma-2-2b-it-ppo-bsz512_4-prompt1024-resp-512-0.640.log), [wandb](https://api.wandb.ai/links/verl-team/h7ux8602)                      |
| NVIDIA GPU | Qwen/Qwen2.5-0.5B-Instruct       | hf checkpoint   | 49.6         | [Qwen blog](https://qwen.ai/blog?id=qwen2.5-llm)                                                                                                                                                                              |
| NVIDIA GPU | Qwen/Qwen2.5-0.5B-Instruct       | PPO             | 56.7         | [command and log](https://github.com/eric-haibin-lin/verl-data/blob/experiments/gsm8k/Qwen2.5-0.5B-bsz256_2-prompt1024-resp512-0.567.log)                                                                                     |
| NVIDIA GPU | Qwen/Qwen2.5-0.5B-Instruct       | PRIME           | 58.7         | [script](https://github.com/verl-project/verl-recipe/blob/main//prime/run_prime_qwen.sh), [wandb](https://api.wandb.ai/links/zefan-wang-thu-tsinghua-university/rxd1btvb)                                                     |
| NVIDIA GPU | Qwen/Qwen2.5-0.5B-Instruct       | GRPO-LoRA       | 54.3         | [command and logs](https://github.com/eric-haibin-lin/verl-data/blob/experiments/gsm8k/Qwen2.5-0.5B-bsz64_2-prompt512-resp1024-lorarank32-score0.543.log)                                                                     |
| NVIDIA GPU | Qwen/Qwen2.5-1.5B-Instruct       | GRPO-LoRA       | 77.9         | [command and logs](https://github.com/eric-haibin-lin/verl-data/blob/experiments/gsm8k/Qwen2.5-1.5B-bsz64_2-prompt512-resp1024-lorarank32-score0.779.log)                                                                     |
| NVIDIA GPU | Qwen/Qwen2.5-3B-Instruct         | GRPO-LoRA       | 86.1         | [command and logs](https://github.com/eric-haibin-lin/verl-data/blob/experiments/gsm8k/Qwen2.5-3B-bsz64_2-prompt512-resp1024-lorarank32-score0.861.log)                                                                       |
| NVIDIA GPU | deepseek-ai/deepseek-llm-7b-chat | PPO (Megatron)  | 69.5 [1]     | [log](https://github.com/eric-haibin-lin/verl-data/blob/experiments/gsm8k/deepseek-llm-7b-chat-megatron-bsz256_4-prompt512-resp512-0.695.log), [wandb](https://wandb.ai/verl-team/verl_megatron_gsm8k_examples/runs/10fetyr3) |
| NVIDIA GPU | Qwen/Qwen2-7B-Instruct           | GRPO            | 89           | [script](https://github.com/volcengine/verl/blob/a65c9157bc0b85b64cd753de19f94e80a11bd871/examples/grpo_trainer/run_qwen2-7b_seq_balance.sh)                                                                                  |
| NVIDIA GPU | Qwen/Qwen2-7B-Instruct           | GRPO (FSDP2)    | 89.8         | [log](https://github.com/eric-haibin-lin/verl-data/blob/experiments/gsm8k/qwen2-7b-fsdp2.log)                                                                                                                                 |
| NVIDIA GPU | Qwen/Qwen2-7B-Instruct           | GRPO (Megatron) | 89.6         | [log](https://github.com/eric-haibin-lin/verl-data/blob/experiments/gsm8k/qwen2-7b_math_megatron.log)                                                                                                                         |
| NVIDIA GPU | Qwen/Qwen2.5-7B-Instruct         | ReMax           | 97           | [script](https://github.com/eric-haibin-lin/verl/blob/main/examples/remax_trainer/run_qwen2.5-3b_seq_balance.sh), [wandb](https://wandb.ai/liziniu1997/verl_remax_example_gsm8k/runs/vxl10pln)                                |
| NVIDIA GPU | Qwen/Qwen2.5-7B-Instruct         | SPPO            | 65.6 (MATH)  | [SPPO script](https://github.com/volcengine/verl-recipe/tree/main/sppo/README.md)                                                                                                                                             |
| NVIDIA GPU | Qwen/Qwen2.5-7B-Instruct         | GRPO-LoRA       | 93.4         | [command and logs](https://github.com/eric-haibin-lin/verl-data/blob/experiments/gsm8k/Qwen2.5-7B-bsz64_8-prompt512-resp1024-lorarank32-score0.934.log)                                                                       |
| NVIDIA GPU | Mixtral-8x22B-Instruct-v0.1      | Instruct model  | 83.7         | [Qwen Blog](https://qwen.ai/blog?id=qwen2.5-llm)                                                                                                                                                                              |
| NVIDIA GPU | Mixtral-8x22B-Instruct-v0.1      | RLOO (Megatron) | 92.3         | [wandb](https://api.wandb.ai/links/ppo_dev/sbuiuf2d)                                                                                                                                                                          |
| NVIDIA GPU | Qwen/Qwen2.5-7B-Instruct         | SPIN            | 92           | [script](https://github.com/volcengine/verl-recipe/tree/main/spin/README.md)                                                                                                                                                  |
| NVIDIA GPU | Qwen/Qwen2-7B-Instruct           | GPG             | 88           | [log](https://github.com/diqiuzhuanzhuan/verldata/blob/main/run_logs/qwen2-7b_math.log), [wandb](https://wandb.ai/diqiuzhuanzhuan/verl_gpg_example_gsm8k_math/runs/ab86c4va)                                                  |
| NVIDIA GPU | Qwen/Qwen2-7B-Instruct           | GPG (Megatron)  | 88           | [log](https://github.com/diqiuzhuanzhuan/verldata/blob/main/run_logs/qwen2-7b_math_megatron.log), [wandb](https://wandb.ai/diqiuzhuanzhuan/verl_gpg_example_gsm8k_math/runs/yy8bheu8)                                         |
| NVIDIA GPU | Qwen/Qwen2.5-VL-7B-Instruct      | GRPO (Megatron) | 65.4 (GEO3k) | [script](https://github.com/volcengine/verl/blob/main/examples/grpo_trainer/run_qwen2_5_vl-7b-megatron.sh), [wandb](https://api.wandb.ai/links/megatron-core-moe-dev/1yngvkek)                                                |
| AMD MI300  | deepseek-ai/deepseek-llm-7b-chat | PPO             | 70.5 [1]     | [log](https://github.com/yushengsu-thu/verl_training_log/blob/main/gsm8k/ppo_run_deepseek7b_llm.log)                                                                                                                          |
| AMD MI300  | deepseek-ai/deepseek-llm-7b-chat | GRPO            | 71.4 [1]     | [log](https://github.com/yushengsu-thu/verl_training_log/blob/main/gsm8k/grpo_run_deepseek7b_llm.log)                                                                                                                         |
| NVIDIA GPU | Qwen/Qwen2.5-14B-Instruct        | GRPO-LoRA       | 94.6         | [command and logs](https://github.com/eric-haibin-lin/verl-data/blob/experiments/gsm8k/Qwen2.5-14B-bsz64_8-prompt512-resp1024-lorarank32-score0.946.log)                                                                      |
| NVIDIA GPU | Qwen/Qwen2.5-32B-Instruct        | GRPO-LoRA       | 95.8         | [command and logs](https://github.com/eric-haibin-lin/verl-data/blob/experiments/gsm8k/Qwen2.5-32B-bsz64_8-prompt512-resp1024-lorarank32-score0.958.log)                                                                      |
| NVIDIA GPU | Qwen/Qwen2.5-72B-Instruct        | GRPO-LoRA       | 96.0         | [command and logs](https://github.com/eric-haibin-lin/verl-data/blob/experiments/gsm8k/Qwen2.5-72B-bs64_8-prompt512-resp1024-lorarank32-score0.960.log)                                                                       |

### DAPO math-17k

- Training DAPO math-17k dataset: https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k
- Testing: AIME'24: https://huggingface.co/datasets/BytedTsinghua-SIA/AIME-2024

Note:

- For Qwen/Qwen2.5-Math-7B, we directly modify the max_position_embeddings to 32768 without observing performance degradation in order to train longer response length.

| Hardware   | Model                      | Method                  | Test score | Details                                                                                                                                                                                             |
| ---------- | -------------------------- | ----------------------- | ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| NVIDIA GPU | Qwen/Qwen2.5-Math-7B (32k) | DAPO                    | 36.3       | [command](https://github.com/verl-project/verl-recipe/blob/main//dapo/test_dapo_7b_math.sh), [logs](https://wandb.ai/verl-org/DAPO%20Reproduction%20on%20verl/runs/ow47vvon?nw=nwusertongyuxuan361) |
| NVIDIA GPU | Qwen/Qwen2.5-7B-Instruct   | DAPO + Code Interpreter | 40.0       | [command](https://github.com/verl-project/verl-recipe/blob/main//retool/run_qwen2_7b_dapo.sh)                                                                                                       |

## Coding related datasets

Below is the result on leetcode if not specified otherwise.

| Hardware   | Model                   | Method | Test score | Details                                                                                                                                                                                |
| ---------- | ----------------------- | ------ | ---------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| NVIDIA GPU | PRIME-RL/Eurus-2-7B-SFT | RPIME  | 36.1       | [script](https://github.com/verl-project/verl-recipe/blob/main//prime/run_prime_qwen_code.sh), [swanlab](https://swanlab.cn/@wangzefan/prime_example/runs/7f541qhspgmy8nmhdlx35/chart) |

### Notes

[1] During evaluation, we have only extracted answers following the format `"####"`. A more flexible answer extraction, longer response length, and better prompt engineering may lead to a higher score.

[2] The default value of `actor_rollout_ref.actor.entropy_coeff` is set to `0.0` since verl 0.3.x on 2025-05-30, which is different from previous versions.
