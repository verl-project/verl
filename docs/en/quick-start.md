# Quick Start: Your First PPO or GRPO Run in `verl`

This page is the **fastest way to build the right mental model** before diving into the full architecture.

The official quickstart already gives a runnable PPO command on GSM8K. Here we do something different: we explain what each stage means, which files implement it, and how to recognize the system in the logs.

## 1. What makes `verl` feel different?

Many RL training codebases hide the control flow inside a giant distributed training script. `verl` does not. It makes the algorithm loop visible:

1. load a prompt batch
2. generate responses
3. score responses
4. compute advantages
5. update actor and critic

That sounds simple, but the compute under each step can still involve many GPUs, many processes, and different backend engines.

```{mermaid}
flowchart LR
    A[Parquet dataset] --> B[RLHFDataset<br/>+ dataloader]
    B --> C[Driver builds DataProto]
    C --> D[Rollout engine<br/>vLLM / SGLang / HF]
    D --> E[Reward function<br/>or reward model]
    E --> F[Advantage computation<br/>on driver]
    F --> G[Critic update]
    F --> H[Actor update]
    G --> I[Checkpoint / validation / metrics]
    H --> I
```

## 2. The smallest useful starting point

For a first run, use the same path that the official docs recommend:

- preprocess GSM8K-style data into parquet
- use a modest Hugging Face model
- launch `verl.trainer.main_ppo`
- keep placement simple: 1 node, 1 GPU, FSDP-style setup first

Key source references:

- dataset preprocessing examples: `examples/data_preprocess/*`
- official quickstart: `docs/start/quickstart.rst`
- PPO entrypoint: `verl/trainer/main_ppo.py`
- PPO training loop: `verl/trainer/ppo/ray_trainer.py`

## 3. Prepare data

The official quickstart uses:

```bash
python3 examples/data_preprocess/gsm8k.py --local_save_dir ~/data/gsm8k
```

Why parquet first instead of raw JSON every iteration?

- reward-related fields can be standardized early
- reading becomes cheaper and more predictable
- the training loop no longer wastes time on repeated text preprocessing

In plain English, preprocessing is like moving groceries from a giant wholesale bag into labeled storage boxes before opening a restaurant. The kitchen can then cook immediately instead of sorting ingredients every time an order arrives.

## 4. Launch a minimal PPO run

A representative single-GPU command is:

```bash
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
  data.train_files=$HOME/data/gsm8k/train.parquet \
  data.val_files=$HOME/data/gsm8k/test.parquet \
  data.train_batch_size=256 \
  data.max_prompt_length=512 \
  data.max_response_length=512 \
  actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
  critic.model.path=Qwen/Qwen2.5-0.5B-Instruct \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  critic.optim.lr=1e-5 \
  actor_rollout_ref.actor.ppo_mini_batch_size=64 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
  critic.ppo_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
  trainer.logger=console \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.total_epochs=15
```

This is not just "a command that works". It encodes a full system contract:

- `verl.trainer.main_ppo` gives Hydra the top-level entrypoint.
- `actor_rollout_ref.*` configures the actor, rollout engine, and reference policy family.
- `critic.*` configures the value model path and update settings.
- `trainer.*` defines global scheduling and logging behavior.

## 5. What files wake up when this command starts?

```{mermaid}
flowchart TD
    A[`python -m verl.trainer.main_ppo`] --> B[`main_ppo.py`<br/>Hydra + Ray init]
    B --> C[`TaskRunner`<br/>choose workers and pools]
    C --> D[`RayPPOTrainer`<br/>create dataloaders and worker groups]
    D --> E[`fit()`<br/>rollout -> reward -> advantage -> update]
    E --> F[`fsdp_workers.py` or `megatron_workers.py`]
    E --> G[`core_algos.py`]
```

That is why `main_ppo.py` is the best first source file to open: it tells you where the rest of the system fans out.

## 6. How to read the training logs

A typical step prints metrics like:

- `timing/gen`
- `timing/ref`
- `timing/values`
- `timing/update_actor`
- `timing/update_critic`
- `actor/pg_loss`
- `critic/vf_loss`
- `response_length/mean`
- validation scores such as `val/test_score/...`

Here is the practical meaning:

| Metric | What it usually means |
| --- | --- |
| `timing/gen` | How long the rollout engine needed to generate responses. If this is huge, inspect rollout backend choice, tensor parallel size, and memory utilization. |
| `timing/ref` | Cost of reference-policy log-prob evaluation. Useful when KL is enabled or GRPO uses KL loss. |
| `timing/values` | Critic forward-pass cost. In PPO this matters; in GRPO it often disappears because there is no critic. |
| `timing/update_actor` | Time spent in the policy optimization step. Often sensitive to micro-batch and sequence length. |
| `timing/update_critic` | Value-model optimization time. Useful for diagnosing critic bottlenecks. |
| `actor/pg_loss` | Policy-gradient objective. The sign and scale depend on aggregation choices; do not compare blindly across methods. |
| `critic/vf_loss` | Value-function loss. If it explodes, advantage and return estimation may become noisy. |
| `response_length/mean` | A quick proxy for how expensive each rollout is becoming. Long responses stress both memory and latency. |

## 7. Which knobs matter first?

If you are new, tune in this order:

1. **model path** -- choose something you can actually fit.
2. **rollout backend** -- `vllm` is a common default; `sglang` and TRT-LLM are alternatives.
3. **batch and micro-batch sizes** -- these mostly decide memory pressure and throughput.
4. **response length** -- longer outputs multiply rollout cost and update cost.
5. **placement** -- colocated first, split placement later when you understand the resource trade-offs.

## 8. When should you switch to GRPO?

Use PPO when you want a classic actor-critic setup with explicit value estimation.

Use GRPO when:

- you want to avoid training a critic
- you care more about grouped relative rewards than learned value estimation
- your workload naturally samples multiple responses per prompt

In `verl`, the loop stays similar, but the logic changes in two important ways:

- set `algorithm.adv_estimator=grpo`
- set `actor_rollout_ref.rollout.n > 1` so one prompt produces a group of responses

Source anchors:

- algorithm docs: `docs/algo/ppo.md`, `docs/algo/grpo.md`
- GRPO math in code: `verl/trainer/ppo/core_algos.py`

## 9. A short FSDP vs Megatron rule of thumb

| Backend | Use it when | Trade-off |
| --- | --- | --- |
| FSDP | You are prototyping, adding a new Hugging Face model, or want a gentler learning curve. | Easier to extend, but weaker scaling and potentially higher resharding overhead. |
| Megatron | You are chasing large-scale throughput, advanced parallelism, or very large models. | Better scalability, but more parallelism concepts and more backend-specific machinery. |

## 10. Where to go next

- If you want the *system picture*, read [`architecture.md`](./architecture.md).
- If you want the *call path*, read [`source-code-tour.md`](./source-code-tour.md).
- If you want the *math*, read [`math-theory.md`](./math-theory.md).

And keep these official references nearby:

- `docs/start/install.rst`
- `docs/start/quickstart.rst`
- `docs/examples/ppo_code_architecture.rst`
- `examples/ppo_trainer/README.md`
- `examples/grpo_trainer/README.md`
