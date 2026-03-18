# Recipe: CollabLLM 

Last updated: 09/22/2025.

> Open-Source Algorithm Implementation & Expriement Running: [Haiquan Chen](https://github.com/chenhaiq), [Shirley Wu](https://github.com/Wuyxin)

🏠 [Homepage](https://aka.ms/CollabLLM) | 📝 [Paper](https://arxiv.org/pdf/2502.00640) | 🤗 [Datasets & Models](https://huggingface.co/collabllm) | ⭐️ [Original Implementation](https://github.com/Wuyxin/collabllm)

`verl` provides a recipe for the Outstanding Paper at ICML 2025, **"CollabLLM: From Passive Responders to Active Collaborators"**. [CollabLLM](https://aka.ms/CollabLLM) is a unified fine-tuning framework that optimizes LLMs for effective and efficient multiturn collaboration with users.

**Core Idea:** Models are rewarded based on how well their responses enable effective *future* collaboration with users.

Paper Authors: [Shirley Wu](https://cs.stanford.edu/~shirwu/), [Michel Galley](https://www.microsoft.com/en-us/research/people/mgalley/), Baolin Peng, Hao Cheng, Gavin Li, Yao Dou, Weixin Cai, [James Zou](https://www.james-zou.com/), [Jure Leskovec](https://cs.stanford.edu/people/jure/), [Jianfeng Gao](https://www.microsoft.com/en-us/research/people/jfgao/)


---
## Quick Start

### 0. Environment
Make sure the required packages for `verl` are installed. Additionally, install `litellm` and export the required API keys. The API model will be used for user simulators and, optionally, LLM Judges (see the Configuration section below).

### 1. Prepare Your Dataset

First, process your dataset using the provided script (see example commands and usage in `process_dataset.py`):

```bash
python process_dataset.py --dataset <> ... --dataset_type <sft or rl>
```


**Requirements:**
- Input: A Hugging Face multiturn dataset. Existing datasets: `collabllm/collabllm-multiturn-$DATASET`, with `DATASET` in one of [`math-hard(-large)`, `medium(-large)`, `bigcodebench(-large)`] (*-large are the datasets used in the CollabLLM paper)
- Example format: See [collabllm-multiturn-math-hard](https://huggingface.co/datasets/collabllm/collabllm-multiturn-math-hard)
- To generate your own dataset: Use [build_dataset.py](https://github.com/Wuyxin/collabllm/blob/main/scripts/engine/build_dataset.py) from the original CollabLLM repository


### 2. Train Your Model

**(Optional) For Supervised Fine-Tuning (SFT):**
```bash
bash train_sft_collabllm.sh
```

**For Reinforcement Learning (RL):**

```bash
bash train_rl_collabllm.sh
```

The RL script shows an example to train CollabLLM on `math-hard-large`. 

- The config to sample future conversations are in `recipe/collabllm/config/collabllm_interaction_config.yaml`. 
- The Multiturn-aware Reward is aggregated from these three conversational-level rewards:

    ```
    +reward_model.reward_kwargs.metric_weights.accuracy=1 \
    +reward_model.reward_kwargs.metric_weights.interactivity=1 \
    +reward_model.reward_kwargs.metric_weights.token_amount=-0.0001 \
    ```

    You can remove, add, or modify the weights depending on your task. A list of implemented metrics you can already add are under `recipe/collabllm/metrics`. For example, on `medium-large`, you can replace `accuracy` with `bleu_score` via
    ```
    +reward_model.reward_kwargs.metric_weights.bleu_score=1 
    ```
    which will instead apply bleu score on the sampled future conversations. 

## Algorithm

| Step | Name                          | Description                                                                 |
|------|-------------------------------|-----------------------------------------------------------------------------|
| 1    | Model response generation     | The model generates multiple responses for each prompt in a batch.          |
| 2    | Collaborative simulation      | A user simulator (e.g., GPT or Claude) samples `num_repeat_rollouts` (TODO: not currently implemented) conversations for up to `max_user_turns` additional turns. |
| 3    | Compute Multiturn-aware Reward | Customized conversational reward functions are applied to the sampled conversations. Rewards are aggregated, then averaged across rollouts. |
| 4    | Update model                  | The model weights are updated using the computed multiturn-aware rewards.  |

---

## Configuration

The primary configuration is managed through the launch script `train_rl_collabllm.sh` and the YAML file `recipe/collabllm/config/collabllm_interaction_config.yaml`. Key configuration sections:

| Section              | Key Parameters / Notes                                                                 |
|----------------------|-----------------------------------------------------------------------------------------|
| `data`               | Paths to training/validation files, batch sizes, sequence lengths.                      |
| `actor_rollout_ref` (common) | Base model path (used for actor + initial reference), FSDP settings, optimization (LR, scheduler). |
| `actor_rollout_ref` (CollabLLM-specific) | Hyperparameters under `actor_rollout_ref.rollout.multi_turn`: `max_user_turns`, `max_assistant_turns`, `num_repeat_rollouts` (TODO: not currently implemented) . |
| `interaction`        | Defined in `collabllm_interaction_config.yaml`. Specifies user simulator and hyperparameters. Requires exported API keys. |
| `reward_model`       | Manager set to `collabllm` by default. Modify `reward_model.reward_kwargs.metric_weights` for conversational rewards and weights. LLM Judge hyperparameters (e.g., `model`, `temperature`) go under `reward_model.reward_kwargs.llm_judge_kwargs`. |
| `algorithm`          | GRPO-specific hyperparameters such as `actor_rollout_ref.rollout.n`.                    |
| `trainer`            | Distributed training (nodes, GPUs per node), logging (WandB), checkpointing frequency.  |

---

## Key Files

| File Path | Purpose |
|-----------|---------|
| `recipe/collabllm/collabllm_agent_loop.py` | Main logic to sample future conversations, using `CollabLLMInteraction` from `verl/interactions/collabllm_interaction.py`. |
| `verl/workers/reward_manager/collabllm.py` | Computes rewards for future conversations, leveraging `recipe/collabllm/reward_function.py` to apply each metric. |

---

## Acknowledgement

We sincerely thank the `verl` community and advisors for their contributions and guidance!
