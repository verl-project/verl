# Recipe: Decoupled Clip and Dynamic Sampling Policy Optimization (DAPO)

Last updated: 06/19/2025.

> Open-Source Algorithm Implementation & Expriement Running: [Yuxuan Tong](https://tongyx361.github.io/), [Guangming Sheng](https://hk.linkedin.com/in/guangming-sheng-b50640211)

🏠 [Homepage](https://dapo-sia.github.io/) | 📝 [Paper@arXiv](https://arxiv.org/abs/2503.14476) | 🤗 [Datasets&Models@HF](https://huggingface.co/collections/BytedTsinghua-SIA/dapo-67d7f1517ee33c8aed059da0) | 🐱 [Code@GitHub](https://github.com/verl-project/verl-recipe/tree/main/dapo/recipe/dapo) | 🐱 [Repo@GitHub](https://github.com/BytedTsinghua-SIA/DAPO)

> We propose the **D**ecoupled Clip and Dynamic s**A**mpling **P**olicy **O**ptimization (DAPO) algorithm. By making our work publicly available, we provide the broader research community and society with practical access to scalable reinforcement learning, enabling all to benefit from these advancements. Our system is based on the awesome [verl](https://github.com/volcengine/verl) framework. Thanks for their great work! Applying DAPO training to Qwen2.5-32B base model proves to outperform the previous state-of-the-art DeepSeek-R1-Zero-Qwen-32B on AIME 2024, achieving **50%** accuracy with **50%** less training steps.
>
> ![dapo-main-result](https://dapo-sia.github.io/static/images/score.png)

## Quickstart

1. Prepare the datasets **on the Ray cluster**:

```bash
bash prepare_dapo_data.sh # This downloads the datasets to ${HOME}/verl/data by default
```

2. Submit the job to the Ray cluster **from any machine**:

```bash
cd verl # Repo root
export RAY_ADDRESS="http://${RAY_IP:-localhost}:8265" # The Ray cluster address to connect to
export WORKING_DIR="${PWD}" # The local directory to package to the Ray cluster
# Set the runtime environment like env vars and pip packages for the Ray cluster in yaml
export RUNTIME_ENV="./recipe/dapo/runtime_env.yaml" # This sets environment variables for the Ray cluster
bash recipe/dapo/run_dapo_qwen2.5_32b.sh # or other scripts
```

## Reproduction Runs

| Setup                                        | AIME 2024 Acc. | Hardware  | Image                                                                | Commit                                                                                       | Environment Variables                                                                                                             | Training Script                                                                                                                                             | Training Record                                                                           |
| -------------------------------------------- | -------------- | --------- | -------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| DAPO                                         | 52%            | 16x8xH800 | `hiyouga/verl:ngc-th2.6.0-cu126-vllm0.8.3-flashinfer0.2.2-cxx11abi0` | [`4f80e4`](https://github.com/volcengine/verl/tree/4f80e465c2ec79ab9c3c30ec74b9745de61d0490) | [runtime_env.yaml](https://github.com/volcengine/verl/blob/4f80e465c2ec79ab9c3c30ec74b9745de61d0490/recipe/dapo/runtime_env.yaml) | [run_dapo_qwen2.5_32b.sh](https://github.com/volcengine/verl/blob/4f80e465c2ec79ab9c3c30ec74b9745de61d0490/recipe/dapo/run_dapo_qwen2.5_32b.sh)             | [W&B](https://wandb.ai/verl-org/DAPO%20Reproduction%20on%20verl/workspace?nw=wmb4qxfht0n) |
| DAPO w/o Dynamic Sampling                    | 50%            | 16x8xH800 | `hiyouga/verl:ngc-th2.6.0-cu126-vllm0.8.3-flashinfer0.2.2-cxx11abi0` | [`4f80e4`](https://github.com/volcengine/verl/tree/4f80e465c2ec79ab9c3c30ec74b9745de61d0490) | [runtime_env.yaml](https://github.com/volcengine/verl/blob/4f80e465c2ec79ab9c3c30ec74b9745de61d0490/recipe/dapo/runtime_env.yaml) | [run_dapo_wo_ds_qwen2.5_32b.sh](https://github.com/volcengine/verl/blob/4f80e465c2ec79ab9c3c30ec74b9745de61d0490/recipe/dapo/run_dapo_wo_ds_qwen2.5_32b.sh) | [W&B](https://wandb.ai/verl-org/DAPO%20Reproduction%20on%20verl/workspace?nw=wmb4qxfht0n) |
| DAPO w/o Token-level Loss & Dynamic Sampling | 44%            | 16x8xH20  | `hiyouga/verl:ngc-th2.5.1-cu120-vllm0.7.4-hotfix`                    | [`4f80e4`](https://github.com/volcengine/verl/tree/4f80e465c2ec79ab9c3c30ec74b9745de61d0490) | [runtime_env.yaml](https://github.com/volcengine/verl/blob/4f80e465c2ec79ab9c3c30ec74b9745de61d0490/recipe/dapo/runtime_env.yaml) | [run_dapo_early_qwen2.5_32b.sh](https://github.com/volcengine/verl/blob/4f80e465c2ec79ab9c3c30ec74b9745de61d0490/recipe/dapo/run_dapo_early_qwen2.5_32b.sh) | [W&B](https://wandb.ai/verl-org/DAPO%20Reproduction%20on%20verl/workspace?nw=wmb4qxfht0n) |

> [!IMPORTANT]
>
> **📢 Call for Contribution!**
>
> Welcome to submit your reproduction runs and setups!

## Configuration

### Separated Clip Epsilons (-> Clip-Higher)

An example configuration:

```yaml
actor_rollout_ref:
  actor:
    clip_ratio_low: 0.2
    clip_ratio_high: 0.28
```

`clip_ratio_low` and `clip_ratio_high` specify the $\varepsilon_{\text {low }}$ and $\varepsilon_{\text {high }}$ in the DAPO objective.

Core relevant code:

```python
pg_losses1 = -advantages * ratio
pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
pg_losses = torch.maximum(pg_losses1, pg_losses2)
```

### Dynamic Sampling (with Group Filtering)

An example configuration:

```yaml
data:
  gen_batch_size: 1536
  train_batch_size: 512
algorithm:
  filter_groups:
    enable: True
    metric: acc # score / seq_reward / seq_final_reward / ...
    max_num_gen_batches: 10 # Non-positive values mean no upper limit
```

Setting `filter_groups.enable` to `True` will filter out groups whose outputs' `metric` are all the same, e.g., for `acc`, groups whose outputs' accuracies are all 1 or 0.

The trainer will repeat sampling with `gen_batch_size` until there are enough qualified groups for `train_batch_size` or reaching the upper limit specified by `max_num_gen_batches`.

Core relevant code:

```python
prompt_bsz = self.config.data.train_batch_size
if num_prompt_in_batch < prompt_bsz:
    print(f'{num_prompt_in_batch=} < {prompt_bsz=}')
    num_gen_batches += 1
    max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches
    if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
        print(f'{num_gen_batches=} < {max_num_gen_batches=}. Keep generating...')
        continue
    else:
        raise ValueError(
            f'{num_gen_batches=} >= {max_num_gen_batches=}. Generated too many. Please check your data.'
        )
else:
    # Align the batch
    traj_bsz = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
    batch = batch[:traj_bsz]
```

### Flexible Loss Aggregation Mode (-> Token-level Loss)

An example configuration:

```yaml
actor_rollout_ref:
  actor:
    loss_agg_mode: "token-mean" # / "seq-mean-token-sum" / "seq-mean-token-mean"
    # NOTE: "token-mean" is the default behavior
```

Setting `loss_agg_mode` to `token-mean` will mean the (policy gradient) loss across all the tokens in all the sequences in a mini-batch.

Core relevant code:

```python
if loss_agg_mode == "token-mean":
    loss = verl_F.masked_mean(loss_mat, loss_mask)
elif loss_agg_mode == "seq-mean-token-sum":
    seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)  # token-sum
    loss = torch.mean(seq_losses)  # seq-mean
elif loss_agg_mode == "seq-mean-token-mean":
    seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1)  # token-mean
    loss = torch.mean(seq_losses)  # seq-mean
else:
    raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")
```

### Overlong Reward Shaping

An example configuration:

```yaml
data:
  max_response_length: 20480 # 16384 + 4096
reward_model:
  overlong_buffer:
    enable: True
    len: 4096
    penalty_factor: 1.0
```

Setting `overlong_buffer.enable` to `True` will penalize the outputs whose lengths are overlong but still within the hard context limit.

Specifically, the penalty increases linearly from `0` to `overlong_buffer.penalty_factor` when the length of the output exceeds the `max_response_length - overlong_buffer.len` by `0` to `overlong_buffer.len` tokens.

Core relevant code:

```python
if self.overlong_buffer_cfg.enable:
    overlong_buffer_len = self.overlong_buffer_cfg.len
    expected_len = self.max_resp_len - overlong_buffer_len
    exceed_len = valid_response_length - expected_len
    overlong_penalty_factor = self.overlong_buffer_cfg.penalty_factor
    overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
    reward += overlong_reward
```

## FAQ

### Where is the "Overlong Filtering" in the paper?

Most experiments in the paper, including the best-performant one, are run without Overlong Filtering because it's somehow overlapping with Overlong Reward Shaping in terms of properly learning from the longest outputs. So we don't implement it here.

### What's the difference between [the `recipe/dapo` directory in the `main` branch](https://github.com/volcengine/verl-recipe/tree/main/dapo) and the [`recipe/dapo` branch](https://github.com/verl-project/verl-recipe/tree/main/dapo/recipe/dapo)?

[The `recipe/dapo` branch](https://github.com/verl-project/verl-recipe/tree/main/dapo/recipe/dapo) is for **as-is reproduction** and thus won't be updated with new features.

[The `recipe/dapo` directory in the `main` branch](https://github.com/volcengine/verl-recipe/tree/main/dapo) works as an example of how to extend the latest `verl` to implement an algorithm recipe, which will be maintained with new features.

### Why can't I produce similar results after modifications?

RL infrastructures nowadays still have inherent unrobustness, on which we are still working hard to improve.

We strongly recommend to only modify one thing at a time.

We also list some known problems here:

1. Enabling CUDA graph (`enforce_eager=False`) might cause model performance degradation, whose cause is still under investigation.
