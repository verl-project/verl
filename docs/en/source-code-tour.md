# Source Code Tour: From `main_ppo.py` to `update_actor()`

This page is a guided walk through the most important code path in the repository.

If you only remember one rule, remember this one:

> Read `verl` **from the controller outward**.

Start from the entrypoint and only then dive into the workers. Otherwise the worker stack feels like a pile of distributed code without a story.

## 1. Entry point: `verl/trainer/main_ppo.py`

This file is the front door for PPO-like training jobs.

What it does at a high level:

1. loads Hydra config
2. auto-selects device settings
3. initializes Ray runtime when needed
4. creates a remote `TaskRunner`
5. asks the task runner to build the training job

The mental model is:

```{mermaid}
sequenceDiagram
    participant U as User command
    participant M as main_ppo.py
    participant R as Ray runtime
    participant T as TaskRunner
    participant P as RayPPOTrainer

    U->>M: python -m verl.trainer.main_ppo
    M->>R: ray.init(...)
    M->>T: create remote TaskRunner
    T->>P: build trainer and worker mapping
    P->>P: init workers and run fit()
```

Why this file matters:

- it tells you where configuration enters the system
- it tells you which part is local and which part is remote
- it is the cleanest proof that the controller is meant to stay lightweight

## 2. `TaskRunner`: turning config into a topology

Inside `main_ppo.py`, `TaskRunner` is the first important control-plane object.

Its job is not to train the model directly. Its job is to decide:

- which worker classes should be used
- which roles exist for this run
- how resource pools should be created
- whether a reward model, critic, or teacher model is needed

This is where the code branches by backend family:

- FSDP / FSDP2 workers
- Megatron workers
- newer engine-worker paths when `use_legacy_worker_impl` is disabled

In other words, `TaskRunner` is the translator between Hydra config and runtime topology.

## 3. `RayPPOTrainer`: the controller's brain

The main trainer lives in:

- `verl/trainer/ppo/ray_trainer.py`

This file is worth reading in three passes.

### Pass 1: constructor and dataloaders

Read the constructor first.

It answers:

- which logical roles are active
- whether reference policy, reward model, teacher model, or critic are enabled
- how dataloaders are constructed
- where checkpointing and logging hooks are prepared

### Pass 2: worker initialization

Then read worker-group setup.

The interesting idea is that each role becomes a **worker group facade**. The controller does not think about "GPU rank 7 on node 2". It thinks about "actor rollout worker group" and "critic worker group".

### Pass 3: `fit()`

Finally read `fit()` carefully. This is the heart of the training loop.

## 4. The `fit()` loop in human language

Here is the loop in the same order the code follows.

```{mermaid}
flowchart TD
    A[Read one batch_dict from dataloader] --> B[Convert to DataProto]
    B --> C[Attach uid and meta_info]
    C --> D[Build generation batch]
    D --> E[Generate sequences]
    E --> F[Repeat / union generated outputs]
    F --> G[Compute reward and reward extras]
    G --> H[Compute or bypass old_log_probs]
    H --> I[Compute reference log-probs if enabled]
    I --> J[Compute values if critic enabled]
    J --> K[Apply KL logic and rollout correction]
    K --> L[Compute advantages on driver]
    L --> M[Update critic]
    L --> N[Update actor]
    M --> O[Validation, metrics, checkpointing]
    N --> O
```

Let us unpack the most important transitions.

### Step A: `batch_dict -> DataProto`

The dataloader returns a regular batch dictionary. The controller immediately wraps it into `DataProto`.

That is a deliberate design decision: once the batch enters the RL pipeline, every stage can now add or merge new fields with a consistent API.

### Step B: generation batch and repeated sampling

`_get_gen_batch(...)` builds the part of the batch needed for rollout.

Then the code may repeat prompts with:

- `rollout.n > 1` for grouped sampling
- especially important in GRPO-style training

### Step C: rollout

The controller calls into `self.async_rollout_manager.generate_sequences(...)`.

This is the point where the code leaves the "nice Python loop" and enters expensive distributed inference.

Possible backends underneath include vLLM, SGLang, or others under `verl/workers/rollout/*`.

### Step D: reward

After generation, the trainer computes reward in two conceptual layers:

1. optional reward-model score
2. rule-based or custom reward extraction

Why split them?

Because `verl` wants reward computation to be composable:

- pure rule-based reward for math/coding tasks
- model-based reward for preference-style settings
- hybrid reward for more complex workflows

### Step E: old log-probs vs rollout log-probs

This part is subtle and very important.

`verl` can operate in two modes:

- **bypass mode**: reuse rollout log-probs as the proximal anchor
- **decoupled mode**: recompute `old_log_probs` separately and use a three-policy interpretation

This is the bridge from ordinary PPO to rollout-correction-aware training.

### Step F: reference policy and critic

If enabled, the trainer separately computes:

- reference log-probs for KL control
- critic values for PPO/GAE

This is why GRPO can be cheaper than PPO: it often removes the critic path entirely.

### Step G: KL, rollout correction, and advantage

The controller then:

- optionally applies KL as an in-reward penalty
- optionally computes rollout-correction weights and metrics
- computes advantages on the driver process

This placement is intentional. Advantage computation is comparatively light-weight and does not need a giant distributed model stack.

### Step H: actor and critic update

Only now does the controller ask the workers to do heavy optimization work.

- `self._update_critic(batch)`
- `self._update_actor(batch)`

These calls fan out into backend-specific worker methods.

## 5. Why `@register` is the hidden hero

To understand why a single call on the controller can launch distributed work correctly, read:

- `verl/single_controller/base/decorator.py`
- `verl/single_controller/base/worker_group.py`
- `docs/single_controller.rst`

The core idea is simple:

- worker methods are decorated with dispatch metadata
- `WorkerGroup` reads that metadata when binding methods
- driver calls become topology-aware distributed RPCs

For example, `generate_sequences(...)` may use a data-parallel split, while Megatron update methods may use topology-aware dispatch that treats PP/TP groups differently.

## 6. `DataProto` deserves its own mental picture

A lot of new readers treat `DataProto` as a technical detail. It is not.

It is the **shared vocabulary** of the whole loop.

```{mermaid}
flowchart LR
    A[Prompts] --> B[DataProto]
    C[Responses] --> B
    D[old_log_probs] --> B
    E[ref_log_prob] --> B
    F[values] --> B
    G[token_level_rewards] --> B
    H[advantages and returns] --> B
    B --> I[actor update]
    B --> J[critic update]
```

If a field is missing or malformed, later stages can fail in confusing ways. That is why learning the evolving contents of `DataProto` is a great debugging skill.

## 7. Backend divergence starts in the worker files

Once you understand the trainer loop, branch into the backend workers.

### FSDP path

Open:

- `verl/workers/fsdp_workers.py`
- `docs/workers/fsdp_workers.rst`

Focus on these methods first:

- `init_model`
- `generate_sequences`
- `compute_ref_log_prob`
- `compute_values`
- `update_actor`
- `update_critic`
- `compute_rm_score`

The FSDP path is conceptually easier because the model definitions stay closer to Hugging Face conventions.

### Megatron path

Open:

- `verl/workers/megatron_workers.py`
- `docs/workers/megatron_workers.rst`

Look for the same logical methods, but pay close attention to different dispatch modes such as Megatron-aware compute paths.

This is where the controller's stable abstraction really pays off: the outer loop barely changes, but the internal parallelism model becomes much richer.

## 8. The source-code reading order I recommend

If you want one coherent path, read files in this exact order:

1. `verl/trainer/main_ppo.py`
2. `verl/trainer/ppo/ray_trainer.py`
3. `verl/protocol.py`
4. `verl/single_controller/base/decorator.py`
5. `verl/single_controller/base/worker_group.py`
6. `verl/single_controller/ray/base.py`
7. `verl/workers/fsdp_workers.py`
8. `verl/workers/megatron_workers.py`
9. `verl/workers/rollout/base.py`
10. one concrete rollout backend under `verl/workers/rollout/*`
11. `verl/trainer/ppo/core_algos.py`
12. reward-related files under `verl/trainer/ppo/reward.py` and `verl/workers/reward_manager/*`

That reading order preserves the story:

- first learn **who orchestrates**
- then learn **how remote calls are bound**
- then learn **what each backend actually computes**
- finally learn **what math those computations implement**

## 9. Where to debug common problems

| Problem | Likely starting point |
| --- | --- |
| generated responses look wrong | rollout backend files and sampling config |
| KL metrics behave strangely | `apply_kl_penalty` in `ray_trainer.py`, `kl_penalty` in `core_algos.py` |
| GRPO grouping is off | `compute_grpo_outcome_advantage` in `core_algos.py`, check `uid` grouping |
| actor update diverges | `update_actor` worker path, policy-loss config, rollout correction config |
| reward is inconsistent | `extract_reward`, reward managers, custom reward function path |
| strange sharding / rank behavior | `decorator.py`, `worker_group.py`, Ray or Megatron dispatch modes |

## 10. Final advice for contributors

When you extend `verl`, resist the temptation to start from the deepest backend file first.

Instead ask three questions in order:

1. **Which stage of the controller loop am I changing?**
2. **What new data fields need to travel inside `DataProto`?**
3. **Which worker method or dispatch mode must implement the heavy compute?**

That habit keeps the architecture readable and prevents accidental coupling between algorithm logic and backend mechanics.

Next, go to [`math-theory.md`](./math-theory.md) to connect these code paths to the core RL formulas.
