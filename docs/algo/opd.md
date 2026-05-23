# On-Policy Distillation (OPD)

**Author:** [Jacob Helwig](https://jacobhelwig.github.io/)

Last updated: 05/14/2026.

## Background

### Summary

1. OPD distills knowledge from teacher model(s) into a student model on states sampled from the student policy.
2. Compared with SFT or standard KD, OPD reduces exposure bias by aligning training-time states with inference-time states.
3. Compared with RLVR, OPD provides dense, continuous, token-level supervision rather than sparse outcome-level rewards.

### Knowledge Distillation

Knowledge distillation (KD) transfers behavior from a teacher model to a student model. In mathematical reasoning, for example, standard KD samples reasoning traces and final answers from the teacher, then trains the student with a next-token prediction objective against the teacher distribution.

This can introduce exposure bias. During training, the student observes states sampled from the teacher. At inference time, however, states are sampled from the student. Unless the teacher and student induce the same state distribution, the student may not learn how the teacher would act in the states the student actually visits.

For example, the student may prefer algebraic proofs while the teacher prefers geometric proofs. Standard KD primarily distills the teacher's behavior along geometric-proof trajectories, even if the student continues to generate algebraic-proof trajectories at inference time.

### On-Policy RL

RLVR has no train/inference state mismatch by construction: rollouts are sampled from the student policy, so the states the student trains on are exactly the states it would visit at inference time. If a rollout produces a correct final answer, the policy is updated to increase the likelihood of the sampled solution.

This aligns training and inference states, but the reward is sparse and outcome-based. A rollout typically contributes a binary success signal at the sequence level rather than dense token-level feedback.

### On-Policy Distillation

On-policy distillation (OPD) [1,2,3] combines the state alignment of on-policy RL with the dense supervision of KD. The student samples rollouts from its own policy. Given each student-generated state, the teacher provides next-token log-probabilities, and the student is trained to match the teacher distribution at those states.

Intuitively, the teacher provides guidance conditioned on the trajectory the student actually chose. If the student follows an algebraic proof path, the teacher supplies supervision for what it would do from that algebraic state.

Formally, let $x \sim p_{\mathrm{data}}$ be a prompt, $y \sim \pi_{\theta}(\cdot \mid x)$ be a student rollout, and $s_t = (x, y_{<t})$ be the state at token $t$. OPD minimizes

$$
\mathcal{L}_{\mathrm{OPD}}(\theta)
=
\mathbb{E}_{x \sim p_{\mathrm{data}},\, y \sim \pi_{\theta}(\cdot \mid x)}
\left[
\frac{1}{|y|}
\sum_{t=1}^{|y|}
D\!\left(
\pi_{\theta}(\cdot \mid s_t),
\nu(\cdot \mid s_t),
y_t
\right)
\right],
$$

where $\pi_{\theta}$ is the student policy, $\nu$ is the teacher policy, and $D_t$ is either a distribution-level divergence or a sampled-token estimator of a divergence.

In practice, the sampled rollout is treated as fixed during the student update. The distinction between supervised OPD and policy-gradient OPD is how the per-token distillation signal is applied.

### Loss Variants

We implement two OPD variants.

#### GKD OPD

GKD OPD [1] directly minimizes a KL divergence between the teacher and student distributions at student-induced states. For forward KL,

$$
D\!\left(
\pi_{\theta}(\cdot \mid s_t),
\nu(\cdot \mid s_t),
y_t
\right)
=
\sum_{v \in V}
\nu(v \mid s_t)
\log
\frac{
\nu(v \mid s_t)
}{
\pi_{\theta}(v \mid s_t)
}.
$$

The distillation loss is optimized by direct backpropagation through the student probabilities. This uses the full distributional signal available from the teacher.

#### PG OPD

PG OPD [3] treats an unbiased estimator of the reverse KL as a reward and applies a policy-gradient update. The reverse KL is given by:

$$
{\mathrm{KL}}\!\left(\pi_{\theta}(\cdot|s_t) \,\Vert \nu(\cdot| s_t)\right)
=
\mathbb{E}_{y_t \sim \pi_{\theta}(\cdot \mid s_t)}
\left[
\log \pi_{\theta}(y_t \mid s_t)
-
\log \nu(y_t \mid s_t)
\right].
$$

Since tokens are sampled from the student policy $\pi_\theta$, a Monte Carlo estimator can be used to approximate this divergence. PG OPD uses a single-sample estimator such as the k1 estimator [4] given by 

$$
D\!\left(
\pi_{\theta}(\cdot \mid s_t),
\nu(\cdot \mid s_t),
y_t
\right)
=
\operatorname{sg}\!\left(
\log \pi_{\theta}(y_t \mid s_t)
-
\log \nu(y_t \mid s_t)
\right),
\quad
y_t \sim \pi_{\theta}(\cdot \mid s_t).
$$

PG OPD then uses the negative reverse KL estimator as a reward given by

$$
r_t
=
\operatorname{sg}\!\left(
\log \nu(y_t \mid s_t)
-
\log \pi_{\theta}(y_t \mid s_t)
\right).
$$

The stop-gradient is required because the reward is used inside a policy-gradient objective. Without it, differentiating through the estimator with respect to student parameters would zero the teacher's logprob, leading to a loss that is independent of the teacher signal. 

### Multi-Teacher OPD

Multi-teacher OPD (MOPD) extends OPD to multiple domain-specialized teachers [5,6,7,8]. This is useful when distilling knowledge to a student across multiple domains. In each domain, such as math, coding, or instruction following, different teachers specialized to the domain can be used.

A base model can be trained or adapted independently on each domain, producing one expert teacher per domain. The student is then trained on a mixture of domains. For each example, the routing key selects the corresponding teacher, and the student matches that teacher's log-probabilities on student-induced states.

MOPD consolidates multiple specialized policies into a single student model while preserving the on-policy state alignment of OPD.

### Bibliography

[1] Agarwal, Rishabh, et al. "On-policy distillation of language models: Learning from self-generated mistakes." *International Conference on Learning Representations*, 2024.

[2] Yang, An, et al. "Qwen3 Technical Report." arXiv preprint arXiv:2505.09388, 2025.

[3] Lu, Kevin and Thinking Machines Lab. "On-Policy Distillation." *Thinking Machines Lab: Connectionism*, Oct. 2025.

[4] DeepSeek-AI. "DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence." 2026.

[5] Xiao, Bangjun, et al. "Mimo-v2-flash Technical Report." arXiv preprint arXiv:2601, 2026.

[6] Zeng, Aohan, et al. "GLM-5: From Vibe Coding to Agentic Engineering." arXiv preprint arXiv:2602.15763, 2026.

[7] Yang, Zhuolin, et al. "Nemotron-Cascade 2: Post-Training LLMs with Cascade RL and Multi-Domain On-Policy Distillation." arXiv preprint arXiv:2603.19220, 2026.

[8] DeepSeek-AI. "DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence." 2026.

## Configuration Parameters

OPD parameters live under three namespaces:

- `distillation.*` — top-level switches and the teacher resource pool ([`DistillationConfig`](../../verl/workers/config/distillation.py))
- `distillation.teacher_models.<name>.*` — one entry per teacher ([`DistillationTeacherModelConfig`](../../verl/workers/config/distillation.py))
- `distillation.distillation_loss.*` — loss-mode and aggregation settings ([`DistillationLossConfig`](../../verl/workers/config/distillation.py))

Defaults below are the YAML defaults from
[`verl/trainer/config/distillation/distillation.yaml`](../../verl/trainer/config/distillation/distillation.yaml).

---

### `distillation.enabled` (bool)

Whether on-policy distillation is enabled. Default: `false`.

When `true`, `main_ppo` allocates a separate teacher resource pool and spins up
one or more teacher inference servers; the actor loss switches from `ppo_loss`
to `distillation_ppo_loss`.

### `distillation.n_gpus_per_node` (int)

Number of GPUs per node in the teacher resource pool. Default: `8`.

### `distillation.nnodes` (int)

Number of nodes in the teacher resource pool. Default: `0` (effectively
disables the pool — must be set to `≥ 1` when `enabled=True`).

**Constraint:** the total teacher pool size (`n_gpus_per_node × nnodes`) must
exactly equal the sum of `(num_replicas × per_replica_world_size)` across all
configured teachers, or `DistillationConfig.__post_init__` raises.

### `distillation.teacher_key` (str)

Field on each sample's data proto used to route the sample to the right
teacher in multi-teacher setups. Default: `"data_source"`.

- **Single-teacher**: ignored (everything goes to the sole teacher).
- **Multi-teacher**: the value of `sample[teacher_key]` must match the `key`
  of one of the configured teachers, or
  `AsyncTeacherLLMServerManager._resolve_teacher_key` raises.

### `distillation.teacher_models` (dict)

Map of teacher entries. Each value is a `DistillationTeacherModelConfig`.

The single-teacher entry is named `teacher_model` by convention. **Pitfall:**
when adding more named teachers, the `teacher_model` entry is silently popped
— so do **not** keep `teacher_model` as one entry alongside other named
teachers. Either rely on it alone, or rename it (e.g. `teacher_model1`) and
add the others.

```bash
# WRONG: teacher_model is popped, only teacher_model2 is used
distillation.teacher_models.teacher_model.key=openai/gsm8k
distillation.teacher_models.teacher_model.model_path=Qwen/Qwen3-4B
+distillation.teacher_models.teacher_model2.key=hiyouga/geometry3k
+distillation.teacher_models.teacher_model2.model_path=Qwen/Qwen3-VL-4B-Instruct

# RIGHT: rename the first teacher
+distillation.teacher_models.teacher_model1.key=openai/gsm8k
+distillation.teacher_models.teacher_model1.model_path=Qwen/Qwen3-4B
+distillation.teacher_models.teacher_model2.key=hiyouga/geometry3k
+distillation.teacher_models.teacher_model2.model_path=Qwen/Qwen3-VL-4B-Instruct
```

---

### `distillation.teacher_models.<name>.key` (str)

Identifier used to route samples to this teacher in multi-teacher mode. Must
match the value of `sample[distillation.teacher_key]`. Default: `null`
(required for multi-teacher; auto-set to `"default"` for single-teacher).

### `distillation.teacher_models.<name>.model_path` (str)

Local path or Hugging Face model id for the teacher. **Required.**

The teacher must share the student's tokenizer/vocab — typically satisfied by
picking a teacher in the same model family (e.g. `Qwen3-32B` teacher for a
`Qwen3-8B` student).

### `distillation.teacher_models.<name>.num_replicas` (int)

Number of inference replicas of this teacher to launch. Default: `0`.

Each replica occupies
`per_replica_world_size = inference.tensor_model_parallel_size * inference.data_parallel_size * inference.pipeline_model_parallel_size`
GPUs, so the teacher's total footprint is `num_replicas × per_replica_world_size`.

For a **single teacher**, you may leave this at `0`: `_resolve_teacher_models`
auto-fills it as `pool_size // per_replica_world_size`.

### `distillation.teacher_models.<name>.inference.*`

Inference-engine config for this teacher; see [`RolloutConfig`](../../verl/workers/config/rollout.py). Same shape as
`actor_rollout_ref.rollout.*`. Notable defaults inherited from the YAML:

- `inference.name` — e.g. `vllm` or `sglang`.
- `inference.tensor_model_parallel_size` — default `2`.
- `inference.gpu_memory_utilization` — default `0.5`.
- `inference.max_model_len` — must accommodate `student_prompt_length +
  student_response_length + 1`; otherwise
  `validate_and_prepare_for_distillation` raises.
- `inference.engine_kwargs.vllm.max_logprobs` — auto-bumped to `≥
  distillation.distillation_loss.topk` whenever the active loss mode requires
  top-k.

`validate_and_prepare_for_distillation` rewrites
`inference.prompt_length := prompt_length + response_length` and
`inference.response_length := 1`, since the teacher only scores the
(prompt + response) prefix and emits one dummy token.

---

### `distillation.distillation_loss.loss_mode` (str)

Distillation divergence to use. Default: `"k3"`.

Two registered families:

- **Top-k** (`forward_kl_topk`): forward KL using the teacher's top-k logits.
- **Single-sample KL estimators** (`kl`, `k1`, `abs`, `mse`, `k2`,
  `low_var_kl`, `k3`): per-token Monte Carlo estimators of reverse KL
  computed from the student's `log_probs` and the teacher's single
  `log_prob` at the sampled token.

### `distillation.distillation_loss.topk` (int, optional)

`k` for top-k distillation losses. Default: `32`.

Only used when `loss_mode` requires top-$k$ (e.g. `forward_kl_topk`). Drives both
the teacher's `prompt_logprobs` request size and (for vLLM) the engine's
`max_logprobs` cap.

### `distillation.distillation_loss.use_task_rewards` (bool)

Whether to add the standard PPO/GRPO task-reward loss on top of the
distillation loss. Default: `true`.

- `true`: final loss is `policy_loss + distillation_loss_coef × distill_loss`.
- `false`: the PPO term is zeroed and only the distillation loss contributes.

Orthogonal to `use_policy_gradient` (which controls how the *distillation
signal itself* is applied).

### `distillation.distillation_loss.distillation_loss_coef` (float)

Coefficient on the distillation loss when combined with task rewards.
Default: `1.0`. Only takes effect when `use_task_rewards=true`.

### `distillation.distillation_loss.loss_max_clamp` (float, optional)

Per-token clamp on the distillation loss to `[-clamp, +clamp]`. Default:
`null` (no clamp).

### `distillation.distillation_loss.log_prob_min_clamp` (float, optional)

Lower clamp on log probabilities used inside divergence computations, to
prevent `log q − log p` from blowing up when `p` or `q` are near zero.
Default: `null`.

### `distillation.distillation_loss.use_policy_gradient` (bool)

How the distillation signal is applied. `true` corresponds to PG OPD, `false` to GKD OPD. Default: `false`.


**Validation:**

- `use_policy_gradient=False` + `loss_mode="k1"` $\to$ `ValueError`. The k1 loss
  has no gradient through the teacher logprob, so backpropagating it directly
  is meaningless.
- `use_policy_gradient=True` + `loss_mode="forward_kl_topk"` $\to$ warning. The
  PG update only moves $\nabla_\theta\log\pi_\theta(y_t|s_t)$ for the sampled token $y_t$, so the top-$k$
  distributional signal is largely unused.

### `distillation.distillation_loss.policy_loss_mode` (str)

Name of the policy loss to use when `use_policy_gradient=True`. Default:
`"vanilla"`. **Currently only `"vanilla"` is supported**; anything else raises
`NotImplementedError`.

### `distillation.distillation_loss.clip_ratio` (float)

PPO clip ratio used by the policy-gradient update when
`use_policy_gradient=True`. Default: `0.2`.

### `distillation.distillation_loss.clip_ratio_low` (float)

Lower bound of the PPO clip range. Default: `0.2`.

### `distillation.distillation_loss.clip_ratio_high` (float)

Upper bound of the PPO clip range. Default: `0.2`.

### `distillation.distillation_loss.global_batch_info` / `loss_settings`

Internal fields populated at runtime — **do not set from the user side.**
`loss_settings` is auto-populated from `loss_mode` via
`get_distillation_loss_settings`; `global_batch_info` is filled by the actor
worker before the loss runs.

## Usage

Example scripts are available in `examples/on_policy_distillation_trainer`. This section shows how to configure different OPD recipes.

### Quick start

For single-teacher OPD, first enable distillation, allocate a teacher resource pool, and specify the teacher model and inference server settings:

```yaml
distillation:
   enabled: true

   n_gpus_per_node: 2
   nnodes: 1

   teacher_models:
      teacher_model:
         model_path: Qwen/Qwen3-32B
         inference:
            name: vllm
            gpu_memory_utilization: 0.8
```

The teacher must share the student's tokenizer and vocabulary. This is usually true for models from the same family, such as a `Qwen3-8B` student and a `Qwen3-32B` teacher. 

In most OPD runs, disable the standard PPO/GRPO reference-policy KL. Otherwise, the student is simultaneously regularized toward the reference policy and distilled from the teacher:

```yaml
actor_rollout_ref:
   actor:
      use_kl_loss: false
algorithm:
   use_kl_in_reward: false
```

### GKD OPD

For efficiency, the current implementation of GKD OPD uses a top-$k$ approximation to forward KL using the top-$k$ teacher logits and the forward KL:

$$
\mathcal{L}_{\mathrm{GKD}}^{(k)}(s_t)
=
\sum_{v \in \operatorname{TopK}(\nu(\cdot \mid s_t))}
\nu(v \mid s_t)
\bigl[
\log \nu(v \mid s_t)
-
\log \pi_\theta(v \mid s_t)
\bigr].
$$

The reason GKD OPD is implemented only over the teacher top-$k$ logits is because current inference servers return log-probabilities for the sampled token and the teacher top-$k$ tokens, but do not support gathering log-probabilities at arbitrary token IDs. Therefore, the implementation supports teacher-top-$k$ forward KL, but not student-top-$k$ reverse KL.

To use GKD OPD, set `loss_mode=forward_kl_topk`, choose `topk`, and disable policy-gradient distillation:

```yaml
distillation:
   distillation_loss:
      loss_mode: forward_kl_topk
      topk: 128
      use_policy_gradient: false
```

Do not use `forward_kl_topk` with `use_policy_gradient=true`. The top-$k$ loss contains distributional information for many teacher-preferred tokens, but a policy-gradient update only acts through the sampled token:

$$
\nabla_\theta \mathcal{L}_{\mathrm{PG}}
\propto
- A_t \nabla_\theta \log \pi_\theta(y_t \mid s_t).
$$

Thus, the update cannot directly assign credit to the non-sampled top-$k$ tokens. This discards most of the distributional signal and can produce misleading updates. For example, if the student already matches the teacher on the sampled token but overestimates other teacher-top-$k$ tokens, the forward KL is still positive; using it as a policy-gradient reward would incorrectly push on the sampled token.


### PG OPD

PG OPD treats the negative reverse-KL estimate as a reward and applies a policy-gradient update. To use PG OPD with the `k1` estimator, set `loss_mode=k1`, enable policy-gradient distillation, and configure the PPO clipping range:

```yaml
distillation:
   distillation_loss:
      loss_mode: k1
      use_policy_gradient: true
      policy_loss_mode: vanilla
      clip_ratio_low: 0.2
      clip_ratio_high: 0.28
```

Currently, only `policy_loss_mode=vanilla` is supported. Other policy-loss modes, such as `dppo_tv`, require additional parameters and are not implemented for OPD.

### Task rewards

OPD can be optimized alone or combined with the standard PPO/GRPO task-reward loss.

When `use_task_rewards=true`, the final loss is

$$
\mathcal{L}
=
\mathcal{L}_{\mathrm{policy}}
+
\lambda_{\mathrm{distill}}
\mathcal{L}_{\mathrm{distill}},
$$

where $\mathcal{L}_{\mathrm{policy}}$ is the PPO/GRPO task-reward loss, $\mathcal{L}_{\mathrm{distill}}$ is the distillation loss, and $\lambda_{\mathrm{distill}}$ is set by `distillation_loss_coef`.

To combine task rewards with distillation:

```yaml
distillation:
   distillation_loss:
      use_task_rewards: true
      distillation_loss_coef: 1.5
```

When `use_task_rewards=false`, the PPO/GRPO task-reward loss is zeroed and the update optimizes only the distillation loss.

### Multi-teacher OPD

Multiple teachers can be configured by adding one entry under `distillation.teacher_models` per teacher. Each teacher has a routing `key`, model path, replica count, and inference configuration.

```yaml
distillation:
   n_gpus_per_node: 8
   nnodes: 2
   teacher_key: data_source

   teacher_models:
      gsm8k:
         key: "openai/gsm8k"
         model_path: Qwen/Qwen3-32B
         num_replicas: 2
         inference:
            name: vllm
            tensor_model_parallel_size: 2
            gpu_memory_utilization: 0.6

      geo3k:
         key: "hiyouga/geometry3k"
         model_path: Qwen/Qwen3-VL-32B-Instruct
         num_replicas: 3
         inference:
            name: vllm
            tensor_model_parallel_size: 4
            gpu_memory_utilization: 0.8

data:
   shuffle: true
   reward_fn_key: data_source
```

In this example, the teacher pool has `8 * 2 = 16` GPUs. Assuming `data_parallel_size=1` and `pipeline_model_parallel_size=1`, the teacher footprints are:

$$
\text{gsm8k}: 2 \text{ replicas} \times 2 \text{ GPUs} = 4 \text{ GPUs}
$$

$$
\text{geo3k}: 3 \text{ replicas} \times 4 \text{ GPUs} = 12 \text{ GPUs}
$$

so the total teacher footprint is $4 + 12 = 16$ GPUs, matching the resource pool.

Teacher replicas are assigned by linearly splitting the teacher resource pool into contiguous GPU bundles. Each individual replica must occupy the expected number of nodes implied by its `per_replica_world_size`:

```python
per_replica_world_size = tensor_model_parallel_size * data_parallel_size * pipeline_model_parallel_size
```

With `n_gpus_per_node=8`, the example above aligns cleanly:

```text
node 0: [gsm8k replica 0: 2 GPUs] [gsm8k replica 1: 2 GPUs] [geo3k replica 0: 4 GPUs]
node 1: [geo3k replica 1: 4 GPUs] [geo3k replica 2: 4 GPUs]
```

No replica crosses a node boundary unless its `per_replica_world_size` requires multiple nodes.

A similar-looking configuration can fail if a replica's GPU bundle does not fall entirely within a single node. For example, with the same `n_gpus_per_node=8`, `nnodes=2` (pool size 16) but two teachers configured as

- `a`: `tensor_model_parallel_size=3`, `num_replicas=2` $\to$ 6 GPUs total
- `b`: `tensor_model_parallel_size=5`, `num_replicas=2` $\to$ 10 GPUs total

the pool size still matches (`6 + 10 = 16`), but the linear bundle layout becomes

```text
node 0: [a_0: 0,1,2] [a_1: 3,4,5] [b_0: 6,7,...
node 1:                                     ...,8,9,10] [b_1: 11,12,13,14,15]
```

Replica `b_0` spans bundles `[6, 11)` — straddling node 0 (bundles 6, 7) and node 1 (bundles 8, 9, 10). A 5-GPU replica with `n_gpus_per_node=8` is expected to fit on a single node (`ceil(5 / 8) = 1`), so `_validate_replica_node_alignment` raises. To fix it, adjust `num_replicas` and the per-teacher inference parallelism.

#### Teacher routing

The `teacher_key` controls routing. It must name a top-level field on each sample's `non_tensor_batch`. `data_source` is one such field, set by the dataset loader. In the example above, `teacher_key=data_source`, so samples with `data_source="openai/gsm8k"` are routed to the `gsm8k` teacher, and samples with `data_source="hiyouga/geometry3k"` are routed to the `geo3k` teacher.

When routing by data source, enable data shuffling. Without shuffling, a dataset created by concatenating other datasets may activate only one teacher for long contiguous stretches. For example, if GSM8K examples are followed by Geo3K examples, then training will use only the GSM8K teacher for the first portion of the epoch and only the Geo3K teacher for the remaining portion.

## Metrics

OPD logs metrics under `actor/distillation/*`.

### Core metrics

- `actor/distillation/loss`  
  Unscaled distillation loss. When `use_task_rewards=true`, compare this with `actor/pg_loss` to choose `distillation_loss_coef`.

- `actor/distillation/abs_loss`  
  Absolute value of the distillation loss. This is mainly useful for signed estimators such as `k1`, where divergences can be negative.

- `actor/distillation/loss_min` / `actor/distillation/loss_max`  
  Minimum and maximum per-token distillation loss in the batch. 

### Top-$k$ metrics

These metrics are logged for top-$k$ loss modes such as `forward_kl_topk`.

- `actor/distillation/student_mass`  
  Average student probability mass assigned to the teacher top-$k$ tokens.

- `actor/distillation/teacher_mass`  
  Average teacher probability mass assigned to its own top-$k$ tokens.

- `actor/distillation/student_mass_min` / `actor/distillation/student_mass_max`  
  Minimum and maximum student mass on the teacher top-$k$ tokens within the batch.

- `actor/distillation/teacher_mass_min` / `actor/distillation/teacher_mass_max`  
  Minimum and maximum teacher mass on the teacher top-$k$ tokens within the batch.

`teacher_mass` indicates how much of the teacher distribution is covered by the selected top-$k$. Low `teacher_mass` means the top-$k$ approximation is truncating substantial teacher probability mass. This can happen either by selecting too small $k$, or due to unstable optimization leading to the student generating low probability sequences.

`student_mass` indicates how much probability the student assigns to the teacher-preferred tokens. 

## Debugging

A useful technique for debugging modifications and additions to the distillation pipeline is to set the student to be the same model as the teacher. The loss should be approximately zero (not exact, due to differences between train/inference engines). 

## Architecture

OPD has two components:

1. **Teacher logprob computation** — runs on a dedicated teacher resource pool as part of the agent loop.
2. **Student optimization** — runs on the train workers, the same actor workers
   that handle PPO/GRPO updates.

### Teacher logprob computation

Teacher logprob computation is interleaved with rollouts inside the **Agent
Loop**. Each sample's teacher call fires as soon as its rollout finishes (no batch-wide barrier) so teacher work overlaps with the still-running
rollouts on other samples.

1. **Input.** `AgentLoopManager.generate_sequences(prompts: DataProto)` receives
   a batch of prompts.

2. **Chunking across workers.** The manager splits the batch evenly across its
   `AgentLoopWorker` actors, then dispatches each
   chunk via `worker.generate_sequences.remote(chunk)`.

3. **Per-sample fan-out inside a worker.** Inside
   `AgentLoopWorker.generate_sequences`, each sample in the chunk is launched as
   its own asyncio task running `_run_agent_loop`. The agent loop runs on the
   rollout GPUs and produces a rollout (prompt + response token ids).

4. **Postprocess hook.** `_run_agent_loop` calls
   `self._agent_loop_postprocess(...)`. This is where teacher logprob
   computation is triggered, per sample, as soon as that sample's rollout is
   ready.

5. **Worker-side teacher dispatch.** `_agent_loop_postprocess` calls
   `self._compute_teacher_logprobs(...)`, which extracts the routing value from
   the sample's non-tensor fields using `sample_kwargs[self.teacher_key]`
   (default `teacher_key="data_source"`), then calls
   `self.teacher_server_manager.compute_teacher_logprobs_single(...)`.

6. **Teacher selection.**
   `AsyncTeacherLLMServerManager.compute_teacher_logprobs_single` resolves the
   teacher via `_resolve_teacher_key`:

   - **Single-teacher**: routing key is ignored; the sole configured teacher
     is used.
   - **Multi-teacher**: `routing_key` must match a configured teacher in
     `distillation.teacher_models`; otherwise an error is raised.

   The resolved key indexes into the per-teacher `LLMServerClient` dict to pick
   the right client.

7. **Sampling params for logprob computation.** The manager builds
   sampling params with `max_tokens=1` plus `prompt_logprobs=topk` (or `0`),
   so the teacher computes logprobs for the (prompt + response) sequence rather than
   generating new tokens. `topk` is set to `distillation.distillation_loss.topk`
   when the loss mode requires top-$k$ (e.g. `forward_kl_topk`); otherwise `0`
   (single-sample logprob only).

8. **Server-side load balancing.** The manager calls `client.generate(...)`,
   which acquires a backing server through the shared `GlobalRequestLoadBalancer`
   actor.

9. **Backend execution.** With the vLLM backend the selected server is a
   `vLLMHttpServer` actor; its `generate` method runs the forward pass and
   returns a `TokenOutput` containing `prompt_ids` and `prompt_logprobs` for
   the full (prompt + response) sequence. The SGLang backend has an analogous
   server class.

10. **Return path.** `compute_teacher_logprobs_single` packs the response into
    two tensors `teacher_ids` and `teacher_logprobs`, each of shape `(S, 1 or K)`
    where `S` is the sequence length and `K` is `topk` (or `1`). These are
    stashed on the rollout output and later concatenated into the per-batch
    `DataProto` for the student training step.


### Student training

Using the `DataProto` produced by the Agent Loop (rollouts + teacher logprobs in
`teacher_logprobs`), the student step proceeds as follows.

1. **Train entry.** `TrainingWorker.train_batch` invokes
   `self.engine.train_batch(data, loss_function=self.loss_fn)`. When
   distillation is enabled, `self.loss_fn` is bound to `distillation_ppo_loss`
   at worker init; otherwise it is the standard `ppo_loss`.

2. **Forward pass and (optional) inline top-k loss.** The training engine's
   forward step runs the model forward and, for top-$k$ loss modes
   (`distillation_use_topk=True`), invokes `distillation_ppo_loss` **as a
   logits processor** while the full logits tensor is still in memory; this is
   the `student_logits is not None` branch of `distillation_ppo_loss`. The
   logits-processor branch dispatches to `compute_forward_kl_topk`, which has a
   separate implementation per training engine (FSDP and Megatron). Per-token
   `distillation_losses`, `student_mass`, and `teacher_mass` tensors are
   written back into `model_output` so the full logits can be freed before the
   final loss step.

3. **Final loss.** After the forward, the engine calls the loss function with
   `model_output` (full logits already freed); this is the
   `student_logits is None` branch of `distillation_ppo_loss`, where:

   1. **Per-token distillation loss** is produced by `distillation_loss(...)`,
      which dispatches via `get_distillation_loss_fn(loss_mode)` to one of the registered distillation losses.

   2. **Optional clamp.** If `loss_max_clamp` is set, per-token losses are
      clamped to `[-clamp, +clamp]` (k1 in particular can be negative).

   3. **Aggregation mode** — controlled by `use_policy_gradient`:

      - `False` (GKD OPD): straight backprop on `distillation_losses`.
      - `True` (PG OPD): treat `-distillation_losses` as
        advantages and run PPO-style clipped importance sampling.

   4. **Combine with task rewards.** A standard PPO policy loss is computed
      from the rollout's task rewards via `ppo_loss(...)`. If
      `use_task_rewards=False` it is zeroed; otherwise the final loss is
      `policy_loss + distillation_loss_coef * distill_loss`.

The returned scalar loss is what `engine.train_batch` backpropagates.

## Files

### **Core Implementation**

- `verl/experimental/teacher_loop/teacher_model.py` — `MultiTeacherModelManager` and `TeacherModelManager`; spin up teacher inference replicas on the dedicated teacher resource pool and expose per-teacher `LLMServerClient` factories
- `verl/experimental/teacher_loop/teacher_manager.py` — `AsyncTeacherLLMServerManager`; routes per-sample teacher calls (single- or multi-teacher) and builds teacher sampling params
- `verl/experimental/agent_loop/agent_loop.py` — `AgentLoopWorker._compute_teacher_logprobs`; per-sample teacher dispatch from `_agent_loop_postprocess`, packs `teacher_logprobs` into the rollout output
- `verl/trainer/distillation/fsdp/losses.py` — FSDP backend `compute_forward_kl_topk`
- `verl/trainer/distillation/megatron/losses.py` — Megatron backend `compute_forward_kl_topk`
- `verl/workers/engine_workers.py` — `ActorRolloutRefWorker.init_model`; binds `distillation_ppo_loss` as the actor's `loss_fn` when distillation is enabled
- `verl/workers/engine/{fsdp,megatron}/transformer_impl.py` — training-engine forward steps; invoke `distillation_ppo_loss` first as a logits processor (top-$k$ modes) and again as the final loss
- `verl/trainer/main_ppo.py` — `is_distillation_enabled` gate; allocates the dedicated `teacher_pool` resource pool
- `verl/trainer/ppo/ray_trainer.py` — constructs `MultiTeacherModelManager` and hands its `get_client()` dict to `AgentLoopWorker(... teacher_client=...)`
- `verl/workers/rollout/llm_server.py` — `LLMServerClient` and `GlobalRequestLoadBalancer` used for both student rollout and teacher logprob computation

### **Configuration Files**

- `verl/trainer/config/distillation/distillation.yaml` — YAML defaults for the `distillation.*` config tree
- `verl/workers/config/distillation.py` — dataclass schema (`DistillationConfig`, `DistillationLossConfig`, `DistillationTeacherModelConfig`)

### **Documentation**

- `docs/algo/opd.md` — this document

### **Example Scripts**

- `examples/on_policy_distillation_trainer/README.md` — script index
- `examples/on_policy_distillation_trainer/run_qwen3_8b_fsdp.sh` — text, vLLM rollout, FSDP student, single teacher
- `examples/on_policy_distillation_trainer/run_qwen3_8b_megatron.sh` — text, vLLM rollout, Megatron student, single teacher
- `examples/on_policy_distillation_trainer/run_qwen3_vl_8b_fsdp.sh` — VL student/teacher, vLLM rollout, FSDP student
- `examples/on_policy_distillation_trainer/run_qwen3_8b_mopd_fsdp.sh` — multi-teacher (gsm8k text + geo3k VL), routed by `data_source`

### **Tests**

- `tests/workers/test_distillation_topk_symmetry_on_cpu.py` — top-k loss symmetry checks
- `tests/utils/test_special_megatron_kl_loss_tp.py` — Megatron KL loss under tensor parallelism
- `tests/special_e2e/run_fully_async_policy_opd.sh` — end-to-end OPD with the fully-async rollouter