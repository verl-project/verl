# Architecture Deep Dive: HybridFlow, Placement, and Backend Swapping

This page explains the **system design logic** behind `verl`.

A lot of distributed RLHF code becomes hard to reason about because three things are entangled:

- the *algorithm* (PPO, GRPO, rollout correction, etc.)
- the *runtime topology* (which model lives on which GPUs)
- the *backend engines* (FSDP, Megatron, vLLM, SGLang, TRT-LLM)

`verl` tries to untangle them.

## 1. The industry-scale pain points `verl` is attacking

At LLM scale, RL training is not just "one more optimizer loop". You have to juggle:

1. **control-flow complexity** -- rollout, reward, KL, value estimation, update, validation, checkpointing
2. **heterogeneous engines** -- training often wants FSDP or Megatron, while rollout wants vLLM/SGLang/TRT-LLM
3. **memory duplication** -- actor weights, rollout weights, reference weights, critic weights, optimizer state, and KV cache all fight for the same HBM
4. **cluster topology** -- colocate for speed, split for capacity, or use separate pools for reward/teacher models
5. **algorithm churn** -- research teams constantly switch PPO variants, critic settings, reward pipelines, or off-policy corrections

A system that hard-wires all of that into one monolithic distributed program becomes fast in one configuration and painful everywhere else.

## 2. HybridFlow in plain language

The central move of `verl` is this:

> Keep the RL **control flow** in one readable controller process, and push distributed model **compute flow** into worker groups.

That sounds modest, but it changes the whole engineering surface.

```{mermaid}
flowchart LR
    subgraph Controller[Single controller process]
        A[Read batch]
        B[Call rollout]
        C[Call reward / reference / critic]
        D[Compute advantages]
        E[Call actor and critic updates]
    end

    subgraph Workers[Distributed worker groups on GPUs]
        F[Actor / rollout workers]
        G[Reference policy workers]
        H[Critic workers]
        I[Reward workers]
    end

    A --> B --> C --> D --> E
    B <--> F
    C <--> G
    C <--> H
    C <--> I
```

Think of it as a movie production:

- the **controller** is the director who decides the scene order
- the **workers** are specialized crews that do expensive physical work
- `DataProto` is the production crate moving materials between teams

This is why `docs/hybrid_flow.rst` and `docs/single_controller.rst` are foundational reads.

## 3. The architectural layers that matter most

### Layer A: entrypoints and orchestration

These files answer "what runs next?"

- `verl/trainer/main_ppo.py`
- `verl/trainer/ppo/ray_trainer.py`
- `verl/trainer/ppo/utils.py`

They define the algorithm loop, worker-role mapping, and the high-level schedule.

### Layer B: transport and RPC abstraction

These files answer "how can one Python call fan out to many remote workers?"

- `verl/protocol.py`
- `verl/single_controller/base/decorator.py`
- `verl/single_controller/base/worker.py`
- `verl/single_controller/base/worker_group.py`
- `verl/single_controller/ray/base.py`

### Layer C: backend workers and engines

These files answer "which GPUs do the actual model work?"

- `verl/workers/fsdp_workers.py`
- `verl/workers/megatron_workers.py`
- `verl/workers/engine/base.py`
- `verl/workers/rollout/*`
- `verl/workers/sharding_manager/*`

### Layer D: RL math and reward logic

These files answer "what objective is optimized?"

- `verl/trainer/ppo/core_algos.py`
- `verl/trainer/ppo/reward.py`
- `verl/workers/reward_manager/*`
- `docs/advance/reward_loop.rst`

## 4. Why `DataProto` matters so much

`DataProto` is not "just a batch dict". It is the common transport format between rollout, reward, reference log-prob computation, critic value estimation, and updates.

It carries three categories of payload:

- `batch`: tensor payloads such as token ids, masks, log-probs, values, rewards, and advantages
- `non_tensor_batch`: strings, uids, multimodal side information, and other non-tensor metadata
- `meta_info`: runtime hints and metrics such as timing, global step, temperature, and token statistics

Without a stable transport object, every stage would need custom glue code. With `DataProto`, the controller can treat the whole training step like a reusable dataflow.

## 5. Worker methods are contracts, not just functions

In `verl.single_controller`, a decorated worker method says more than "run this code". It declares:

- how the input should be split
- which ranks should execute
- how the output should be collected back

A driver call like:

```python
output = actor_rollout_wg.generate_sequences(batch)
```

can therefore mean very different low-level behavior depending on the dispatch mode.

```{mermaid}
flowchart TD
    A[Driver calls worker-group method] --> B[@register metadata]
    B --> C[Dispatch fn<br/>split input by DP or Megatron topology]
    C --> D[Execute fn<br/>run on all ranks or selected ranks]
    D --> E[Collect fn<br/>merge outputs into one DataProto]
```

This is one of the most elegant ideas in the repository: it preserves a readable algorithm loop while still exposing topology-aware execution.

## 6. Roles, resource pools, and placement

The trainer thinks in **logical roles**:

- actor/rollout
- reference policy
- critic
- reward model
- sometimes teacher models or reward-loop workers

A **resource pool** is the physical home for one or more roles.

```{mermaid}
flowchart LR
    subgraph LogicalRoles[Logical roles]
        A[Actor/Rollout]
        B[Reference]
        C[Critic]
        D[Reward]
    end

    subgraph PhysicalPools[Physical resource pools]
        P1[global_pool<br/>node x GPU slice]
        P2[reward_pool<br/>optional separate pool]
    end

    A --> P1
    B --> P1
    C --> P1
    D --> P1
    D -. optional separate placement .-> P2
```

This decoupling is the reason `verl` can support both:

- **colocated placement** -- faster communication, lower process overhead
- **split placement** -- more flexible capacity planning, sometimes easier on memory

If you change the mapping between roles and pools, you change the topology without rewriting the PPO loop.

## 7. FSDP vs Megatron: same loop, different machinery

The controller-level PPO loop stays almost the same. The worker-level implementation changes dramatically.

| Dimension | FSDP path | Megatron path |
| --- | --- | --- |
| Main worker file | `verl/workers/fsdp_workers.py` | `verl/workers/megatron_workers.py` |
| Best use case | prototyping, fast extension, Hugging Face model compatibility | large-scale training, richer parallelism, high-end throughput |
| Model parallelism | lighter-weight, easier mental model | TP / PP / DP / CP / EP aware |
| Resharding story | simpler but can be costlier | more complex, but better scaling and stronger HybridEngine support |
| Research ergonomics | better | lower |

The key architectural point is not "which backend is better". It is that the **control plane stays stable while the compute plane changes**.

## 8. Why 3D-HybridEngine matters

When training and generation share the same or overlapping hardware, a naive design can duplicate too much state:

- actor weights for training
- rollout weights for generation
- KV cache for decoding
- optimizer state and activations

That is how GPU memory disappears.

A useful intuition is:

- **naive peak memory** behaves like `actor weights + rollout weights + KV cache + training extras`
- **HybridEngine-style peak memory** tries to behave more like `shared/resharded weights + transition overhead + KV cache`

This is *not* an exact formula, but it explains the design target.

### A toy memory example

Imagine one GPU budget is 80 GB.

- actor weights occupy 28 GB
- rollout weights occupy another 28 GB
- KV cache needs 18 GB
- training-only extra state needs 10 GB

A naive copy-everything design wants `28 + 28 + 18 + 10 = 84 GB`, which already fails.

If the system can avoid persistent duplication and instead reshard/share weights across phases, the peak might feel closer to:

- shared actor/rollout footprint: 28 GB
- transition overhead: 6 GB
- KV cache: 18 GB
- training extra state: 10 GB

Now the peak is about `62 GB`, which leaves headroom instead of crashing.

Again, the real memory picture depends on backend, parallelism, batch size, and offload settings. But this cartoon explains why `verl` treats training-to-generation transition as a first-class system problem.

## 9. The full training step as architecture, not just as math

```{mermaid}
flowchart TD
    A[Load prompt batch] --> B[Repeat by rollout.n if needed]
    B --> C[Generate responses with rollout engine]
    C --> D[Optionally recompute old log-probs]
    D --> E[Optionally compute reference log-probs]
    E --> F[Optionally compute critic values]
    F --> G[Extract rule-based or model-based reward]
    G --> H[Apply KL penalty or KL loss logic]
    H --> I[Compute advantages]
    I --> J[Update critic]
    I --> K[Update actor]
    J --> L[Checkpoint / validation / logging]
    K --> L
```

Notice what is heavy and what is light:

- heavy: rollout, log-prob recomputation, value estimation, actor update, critic update
- light: orchestration, reward composition, advantage bookkeeping, metrics assembly

That is exactly why the controller stays separate from the workers.

## 10. What to read when you hit a real bottleneck

| Symptom | Start reading here |
| --- | --- |
| rollout is slow | `verl/workers/rollout/*`, `docs/workers/sglang_worker.rst`, rollout-related config docs |
| memory blows up during rollout | `docs/workers/megatron_workers.rst`, `docs/workers/fsdp_workers.rst`, `verl/workers/sharding_manager/*` |
| batch topology is confusing | `docs/single_controller.rst`, `verl/single_controller/base/decorator.py` |
| rewards are hard to extend | `docs/advance/reward_loop.rst`, `verl/workers/reward_manager/*` |
| math seems off | `verl/trainer/ppo/core_algos.py`, `docs/algo/ppo.md`, `docs/algo/grpo.md`, `docs/algo/rollout_corr_math.md` |

## 11. Practical takeaway

The most important architectural sentence in this repository is:

> **`verl` keeps the algorithm loop stable while letting placement, parallelism, and backend engines vary underneath.**

That is why it scales from "single-GPU research prototype" to "very large distributed RL training job" without forcing you to rewrite the core control logic every time.

Next, read [`source-code-tour.md`](./source-code-tour.md) to see this architecture unfold file by file.
