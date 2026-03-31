# verl Deep Dive Tutorial Suite

This mini-doc set is a **source-driven companion** to the official `verl` documentation. It is for readers who do not only want to run a command, but also want to answer the harder questions:

- What *actually* happens after `python -m verl.trainer.main_ppo` starts?
- Why does `verl` split control flow from compute flow instead of fusing everything into one distributed script?
- Where do memory pressure, weight resharding, and cross-process communication come from?
- How do PPO, GRPO, KL control, and rollout correction map into real code paths?

The official docs already cover installation, APIs, and many specialized features. This tutorial set focuses on the missing middle layer: **how the system hangs together end to end**.

## Who should read this?

- You can already read Python and distributed training configs.
- You want a readable bridge between papers, config files, and source code.
- You care about practical system questions such as *which process owns what*, *where GPU memory goes*, and *which file to open first when debugging*.

## Recommended reading order

1. [`quick-start.md`](./quick-start.md) -- launch a mental model before touching complex configs.
2. [`architecture.md`](./architecture.md) -- understand HybridFlow, placement, worker roles, and backend swapping.
3. [`source-code-tour.md`](./source-code-tour.md) -- walk the repository from entrypoint to worker RPCs and update kernels.
4. [`math-theory.md`](./math-theory.md) -- decode PPO, GRPO, KL control, and rollout correction with intuitive examples.

```{toctree}
:maxdepth: 1

quick-start
architecture
source-code-tour
math-theory
```

```{mermaid}
flowchart TD
    A[Quick Start<br/>from first command to first metrics] --> B[Architecture<br/>HybridFlow, placement, backend roles]
    B --> C[Source Code Tour<br/>main_ppo.py to update_actor()]
    C --> D[Math Theory<br/>PPO, GRPO, KL, rollout correction]
    B --> E[Official docs and examples<br/>for deeper feature-specific reading]
```

## The big picture in one paragraph

`verl` is an RL post-training system for large language models. Its signature move is **HybridFlow**: the algorithm loop stays in a relatively lightweight **single controller**, while expensive model computation lives in distributed **workers**. That separation lets the controller express RL logic in readable Python, while the workers swap between FSDP, Megatron, vLLM, SGLang, TRT-LLM, or newer engines.

In other words, the controller behaves like a film director: it decides *what happens next*. The workers are the camera crew, lighting crew, and editors: they do the heavy work, often in parallel, often on different GPUs, and often with different internal toolchains.

## Four ideas worth memorizing early

### 1. `DataProto` is the shipping container

`verl/protocol.py` defines `DataProto`, the batch object that carries tensors, non-tensor metadata, and `meta_info` between stages. If you understand `DataProto`, you understand how prompts, responses, rewards, log-probs, advantages, and metrics move through the system.

### 2. `@register` turns worker methods into distributed RPC contracts

In `verl.single_controller`, decorated worker methods declare **how inputs are split**, **which ranks execute**, and **how outputs are collected**. This is why the driver can call `worker_group.generate_sequences(batch)` as if it were local, even when the underlying compute spans many processes.

### 3. roles and resource pools are the placement language

The trainer thinks in logical roles such as actor/rollout, critic, reference policy, and reward model. Ray resource pools decide where those roles live physically. Change the mapping, and you change the topology without rewriting the algorithm loop.

### 4. the algorithm loop is thin by design

`verl/trainer/ppo/ray_trainer.py` is intentionally not where all heavy tensor math lives. The driver orchestrates rollout, reward, KL, advantage, and update. The backend-specific workers execute model code, while `verl/trainer/ppo/core_algos.py` owns the core RL formulas.

## Fast glossary

| Term | Practical meaning in `verl` |
| --- | --- |
| HybridFlow | Split RL control flow from distributed compute flow. |
| Single controller | One Python process that orchestrates rollout, reward, advantage, and updates. |
| WorkerGroup | A distributed facade that makes many remote workers look like one object. |
| ResourcePool | A slice of cluster resources assigned to a role or set of roles. |
| HybridEngine | A colocated actor/rollout setup that reduces transfer and memory overhead. |
| 3D-HybridEngine | Megatron-oriented resharding design that makes training-to-generation transitions cheaper. |
| Rollout correction | Three-policy logic that separates behavior policy, proximal policy, and current policy. |

## Source anchors used throughout this tutorial

- Entry: `verl/trainer/main_ppo.py`
- Main trainer loop: `verl/trainer/ppo/ray_trainer.py`
- Core math: `verl/trainer/ppo/core_algos.py`
- Batch transport: `verl/protocol.py`
- Single-controller mechanics: `verl/single_controller/base/*`, `verl/single_controller/ray/base.py`
- FSDP worker stack: `verl/workers/fsdp_workers.py`
- Megatron worker stack: `verl/workers/megatron_workers.py`
- Rollout adapters: `verl/workers/rollout/*`
- Reward system: `verl/trainer/ppo/reward.py`, `verl/workers/reward_manager/*`, `docs/advance/reward_loop.rst`

## Official docs you should keep open in another tab

- `docs/hybrid_flow.rst`
- `docs/single_controller.rst`
- `docs/examples/ppo_code_architecture.rst`
- `docs/workers/ray_trainer.rst`
- `docs/workers/fsdp_workers.rst`
- `docs/workers/megatron_workers.rst`
- `docs/algo/ppo.md`
- `docs/algo/grpo.md`
- `docs/algo/rollout_corr_math.md`

## Next step

If you want the shortest path from zero to "I know what this repository is doing", continue with [`quick-start.md`](./quick-start.md).
