# On-Policy Distillation

**Author:** [Jacob Helwig](https://jacobhelwig.github.io/)

Last updated: 05/14/2026.

---

> **Contents**
>
> - **Background** - Introduction to On-Policy Distillation
> - **Configuration Parameters** - Reference for the `distillation.*` config tree
> - **Usage** - Recipes for single-teacher, multi-teacher, GKD, PG, and task-reward OPD
> - **Metrics** - Logged metrics and how to interpret them
> - **Debugging** - Sanity-check tips
> - **Architecture** - Implementation and control flow of OPD
> - **Files** - Where each piece of OPD lives in the repo

---

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

RLVR avoids this state-distribution mismatch by sampling rollouts from the student policy. If a rollout produces a correct final answer, the policy is updated to increase the likelihood of the sampled solution.

This aligns training and inference states, but the reward is sparse and outcome-based. A rollout typically contributes a binary success signal at the sequence level rather than dense token-level feedback.

### On-Policy Distillation

On-policy distillation (OPD) combines the state alignment of on-policy RL with the dense supervision of KD. The student samples rollouts from its own policy. Given each student-generated state, the teacher provides next-token log-probabilities, and the student is trained to match the teacher distribution at those states.

Intuitively, the teacher provides guidance conditioned on the trajectory the student actually chose. If the student follows an algebraic proof path, the teacher supplies supervision for what it would do from that algebraic state.

Formally, let \(x \sim p_{\mathrm{data}}\) be a prompt, \(y \sim \pi_{\theta}(\cdot \mid x)\) be a student rollout, and \(s_t = (x, y_{<t})\) be the state at token \(t\). OPD minimizes

\[
\mathcal{L}_{\mathrm{OPD}}(\theta)
=
\mathbb{E}_{x \sim p_{\mathrm{data}},\, y \sim \pi_{\theta}(\cdot \mid x)}
\left[
\frac{1}{|y|}
\sum_{t=1}^{|y|}
D_t\!\left(
\pi_{\theta}(\cdot \mid s_t),
\nu(\cdot \mid s_t),
y_t
\right)
\right],
\]

where \(\pi_{\theta}\) is the student policy, \(\nu\) is the teacher policy, and \(D_t\) is either a distribution-level divergence or a sampled-token estimator of a divergence.

In practice, the sampled rollout is treated as fixed during the student update. The distinction between supervised OPD and policy-gradient OPD is how the per-token distillation signal is applied.

### Loss Variants

We implement two OPD variants.

#### GKD OPD

GKD OPD directly minimizes a KL divergence between the teacher and student distributions at student-induced states. For forward KL,

\[
D_{\mathrm{KL}}\!\left(\nu \,\|\, \pi_{\theta}\right)(s_t)
=
\sum_{v \in V}
\nu(v \mid s_t)
\log
\frac{
\nu(v \mid s_t)
}{
\pi_{\theta}(v \mid s_t)
}.
\]

The distillation loss is optimized by direct backpropagation through the student probabilities. This uses the full distributional signal available from the teacher.

#### PG OPD

PG OPD treats the distillation signal as a reward and applies a policy-gradient update. Since tokens are sampled from the student policy, the sampled-token estimator corresponds to reverse KL:

\[
D_{\mathrm{KL}}\!\left(\pi_{\theta} \,\|\, \nu\right)(s_t)
=
\mathbb{E}_{y_t \sim \pi_{\theta}(\cdot \mid s_t)}
\left[
\log \pi_{\theta}(y_t \mid s_t)
-
\log \nu(y_t \mid s_t)
\right].
\]

The per-token Monte Carlo estimator is

\[
\widehat{D}_{\mathrm{KL}}\!\left(\pi_{\theta} \,\|\, \nu\right)(s_t, y_t)
=
\operatorname{sg}\!\left(
\log \pi_{\theta}(y_t \mid s_t)
-
\log \nu(y_t \mid s_t)
\right),
\quad
y_t \sim \pi_{\theta}(\cdot \mid s_t).
\]

Equivalently, maximizing negative reverse KL uses the reward

\[
r_t
=
\operatorname{sg}\!\left(
\log \nu(y_t \mid s_t)
-
\log \pi_{\theta}(y_t \mid s_t)
\right).
\]

The stop-gradient is required because the reward is used inside a policy-gradient objective. Without it, differentiating through the estimator would not produce the intended score-function update. This estimator is valid for reverse KL because samples are drawn from the student distribution; estimating forward KL would require samples from the teacher distribution.

### Multi-Teacher OPD

Multi-teacher OPD (MOPD) extends OPD to multiple domain-specialized teachers. This is useful when different teachers are specialized for different data domains, such as math, coding, or instruction following.

A base model can be trained or adapted independently on each domain, producing one expert teacher per domain. The student is then trained on a mixture of domains. For each example, the routing key selects the corresponding teacher, and the student matches that teacher's log-probabilities on student-induced states.

MOPD consolidates multiple specialized policies into a single student model while preserving the on-policy state alignment of OPD.

### Bibliography

[1] Agarwal, Rishabh, et al. "On-policy distillation of language models: Learning from self-generated mistakes." *International Conference on Learning Representations*, 2024.

[2] Yang, An, et al. "Qwen3 Technical Report." arXiv preprint arXiv:2505.09388, 2025.

[3] Lu, Kevin and Thinking Machines Lab. "On-Policy Distillation." *Thinking Machines Lab: Connectionism*, Oct. 2025.

[4] Xiao, Bangjun, et al. "Mimo-v2-flash Technical Report." arXiv preprint arXiv:2601, 2026.

[5] Zeng, Aohan, et al. "GLM-5: From Vibe Coding to Agentic Engineering." arXiv preprint arXiv:2602.15763, 2026.

[6] Yang, Zhuolin, et al. "Nemotron-Cascade 2: Post-Training LLMs with Cascade RL and Multi-Domain On-Policy Distillation." arXiv preprint arXiv:2603.19220, 2026.

[7] DeepSeek-AI. "DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence." 2026.

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
`Qwen3-8B` student). Different LM-head padding is fine.

### `distillation.teacher_models.<name>.num_replicas` (int)

Number of inference replicas of this teacher to launch. Default: `0`.

Each replica occupies
`per_replica_world_size = inference.tensor_model_parallel_size * inference.data_parallel_size * inference.pipeline_model_parallel_size`
GPUs, so the teacher's total footprint is `num_replicas × per_replica_world_size`.

For a **single teacher**, you may leave this at `0`: `_resolve_teacher_models`
auto-fills it as `pool_size // per_replica_world_size`.

### `distillation.teacher_models.<name>.inference.*` ([`RolloutConfig`](../../verl/workers/config/rollout.py))

Inference-engine config for this teacher (vLLM/SGLang). Same shape as
`actor_rollout_ref.rollout.*`. Notable defaults inherited from the YAML:

- `inference.name` — e.g. `vllm` or `sglang`.
- `inference.tensor_model_parallel_size` — default `2`.
- `inference.gpu_memory_utilization` — default `0.5`.
- `inference.max_model_len` — must accommodate `student_prompt_length +
  student_response_length + 1`; otherwise
  `validate_and_prepare_for_distillation` raises.
- `inference.engine_kwargs.vllm.max_logprobs` — auto-bumped to `≥
  distillation.distillation_loss.topk` whenever the active loss mode requires
  top-k. (No-op for SGLang; `top_logprobs_num` is per-request there.)

`validate_and_prepare_for_distillation` rewrites
`inference.prompt_length := prompt_length + response_length` and
`inference.response_length := 1`, since the teacher only scores the
(prompt + response) prefix and emits one dummy token.

---

### `distillation.distillation_loss.loss_mode` (str)

Distillation divergence to use. Default: `"k3"`.

Two registered families:

- **Top-k** (`forward_kl_topk`): forward KL using the teacher's top-k logits.
  Computed as a logits processor inline with the student's forward pass so the
  full vocab logits don't need to leave the GPU.
- **Single-sample KL estimators** (`kl`, `k1`, `abs`, `mse`, `k2`,
  `low_var_kl`, `k3`): per-token Monte Carlo estimators of reverse KL
  computed from the student's `log_probs` and the teacher's single
  `log_prob` at the sampled token.

### `distillation.distillation_loss.topk` (int, optional)

`k` for top-k distillation losses. Default: `32`.

Only used when `loss_mode` requires top-k (e.g. `forward_kl_topk`). Drives both
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

Useful for `k1`, which can be negative, and to defang occasional
exploding-token outliers. Example scripts override to `10.0`.

### `distillation.distillation_loss.log_prob_min_clamp` (float, optional)

Lower clamp on log probabilities used inside divergence computations, to
prevent `log q − log p` from blowing up when `p` or `q` are near zero.
Default: `null`. Example scripts override to `-10.0`.

### `distillation.distillation_loss.use_policy_gradient` (bool)

How the distillation signal is applied. Default: `false`.

- `false` (supervised, [arxiv:2306.13649](https://arxiv.org/abs/2306.13649)):
  per-token distillation loss is aggregated over the response mask and
  backpropagated directly. Recommended with `loss_mode=k3` or
  `forward_kl_topk`.
- `true` (on-policy distillation,
  [Thinking Machines blog](https://thinkingmachines.ai/blog/on-policy-distillation/)):
  treat `−distillation_loss` as the advantage and run a PPO-style clipped
  importance-sampling update against `data["old_log_probs"]`. Recommended
  with `loss_mode=k1`.

**Validation:**

- `use_policy_gradient=False` + `loss_mode="k1"` → `ValueError`. The k1 loss
  has no gradient through the teacher logprob, so backpropagating it directly
  is meaningless.
- `use_policy_gradient=True` + `loss_mode="forward_kl_topk"` → warning. The
  PG update only moves `∇log π(a)` for the sampled token, so the top-k
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

We have example scripts in the directory `examples/on_policy_distillation_trainer`. Now we show some basics for adapting a script to OPD.

## Usage

Example scripts are available in `examples/on_policy_distillation_trainer`. This section shows the main configuration changes needed to adapt an existing PPO/GRPO script to OPD.

### Quick start

For single-teacher OPD, first enable distillation and allocate a teacher resource pool:

```yaml
distillation:
   enabled: true
   n_gpus_per_node: 2
   nnodes: 1
```

Then specify the teacher model and inference-server settings:

```yaml
distillation:
   teacher_models:
      teacher_model:
         model_path: Qwen/Qwen3-32B
         inference:
            name: vllm
            gpu_memory_utilization: 0.8
```

The teacher must share the student's tokenizer and vocabulary. This is usually true for models from the same family, such as a `Qwen3-8B` student and a `Qwen3-32B` teacher. Different LM-head padding is allowed, but the vocabularies must be compatible.

In most OPD runs, disable the standard PPO/GRPO reference-policy KL. Otherwise, the student is simultaneously regularized toward the reference policy and distilled from the teacher:

```yaml
actor_rollout_ref:
   actor:
      use_kl_loss: false
algorithm:
   use_kl_in_reward: false
```

### GKD OPD

GKD OPD uses a top-\(k\) approximation to forward KL from the teacher to the student:

\[
D_{\mathrm{KL}}^{(k)}(\nu \,\|\, \pi_\theta)(s_t)
=
\sum_{v \in \operatorname{TopK}(\nu(\cdot \mid s_t))}
\tilde{\nu}(v \mid s_t)
\log
\frac{
\tilde{\nu}(v \mid s_t)
}{
\tilde{\pi}_\theta(v \mid s_t)
},
\]

where \(\nu\) is the teacher policy, \(\pi_\theta\) is the student policy, and the distributions are renormalized over the teacher top-\(k\) tokens.

As of May 14, 2026, GKD OPD is implemented only over the teacher top-\(k\) logits. The current teacher server returns log-probabilities for the sampled token and the teacher top-\(k\) tokens, but does not support gathering log-probabilities at arbitrary token IDs. Therefore, the implementation supports teacher-top-\(k\) forward KL, but not student-top-\(k\) reverse KL.

To use GKD OPD, set `loss_mode=forward_kl_topk`, choose `topk`, and disable policy-gradient distillation:

```yaml
distillation:
   distillation_loss:
      loss_mode: forward_kl_topk
      topk: 128
      use_policy_gradient: false
```

Do not use `forward_kl_topk` with `use_policy_gradient=true`. The top-\(k\) loss contains distributional information for many teacher-preferred tokens, but a policy-gradient update only acts through the sampled token:

\[
\nabla_\theta \mathcal{L}_{\mathrm{PG}}
\propto
- A_t \nabla_\theta \log \pi_\theta(y_t \mid s_t).
\]

Thus, the update cannot directly assign credit to the non-sampled top-\(k\) tokens. This discards most of the distributional signal and can produce misleading updates. For example, if the student already matches the teacher on the sampled token but overestimates other teacher-top-\(k\) tokens, the forward KL is still positive; using it as a policy-gradient reward would incorrectly push on the sampled token.


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

The `k1` estimator is valid for reverse KL because sampled tokens are drawn from the student policy:

\[
D_{\mathrm{KL}}(\pi_\theta \,\|\, \nu)(s_t)
=
\mathbb{E}_{y_t \sim \pi_\theta(\cdot \mid s_t)}
\left[
\log \pi_\theta(y_t \mid s_t)
-
\log \nu(y_t \mid s_t)
\right].
\]

Thus, a single student-sampled token gives the estimator

\[
\widehat{D}_{\mathrm{KL}}(\pi_\theta \,\|\, \nu)(s_t, y_t)
=
\log \pi_\theta(y_t \mid s_t)
-
\log \nu(y_t \mid s_t).
\]

Estimating forward KL would require samples from the teacher distribution:

\[
D_{\mathrm{KL}}(\nu \,\|\, \pi_\theta)(s_t)
=
\mathbb{E}_{y_t \sim \nu(\cdot \mid s_t)}
\left[
\log \nu(y_t \mid s_t)
-
\log \pi_\theta(y_t \mid s_t)
\right],
\]

which is closer to standard off-policy KD.

### Task rewards

OPD can be optimized alone or combined with the standard PPO/GRPO task-reward loss.

When `use_task_rewards=true`, the final loss is

\[
\mathcal{L}
=
\mathcal{L}_{\mathrm{policy}}
+
\lambda_{\mathrm{distill}}
\mathcal{L}_{\mathrm{distill}},
\]

where \(\mathcal{L}_{\mathrm{policy}}\) is the PPO/GRPO task-reward loss, \(\mathcal{L}_{\mathrm{distill}}\) is the distillation loss, and \(\lambda_{\mathrm{distill}}\) is set by `distillation_loss_coef`.

To combine task rewards with distillation:

```yaml
distillation:
   distillation_loss:
      use_task_rewards: true
      distillation_loss_coef: 1.5
```

When `use_task_rewards=false`, the PPO/GRPO task-reward loss is zeroed and the model optimizes only the distillation loss.

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

In this example, the teacher pool has

\[
8 \times 2 = 16
\]

GPUs. Assuming `data_parallel_size=1` and `pipeline_model_parallel_size=1`, the teacher footprints are:

\[
\text{gsm8k}: 2 \text{ replicas} \times 2 \text{ GPUs} = 4 \text{ GPUs}
\]

\[
\text{geo3k}: 3 \text{ replicas} \times 4 \text{ GPUs} = 12 \text{ GPUs}
\]

so the total teacher footprint is \(4 + 12 = 16\) GPUs, matching the resource pool.

Teacher replicas are assigned by linearly splitting the teacher resource pool into contiguous GPU bundles. Each individual replica must occupy the expected number of nodes implied by its `per_replica_world_size`:

\[
\texttt{per\_replica\_world\_size}
=
\texttt{tensor\_model\_parallel\_size}
\times
\texttt{data\_parallel\_size}
\times
\texttt{pipeline\_model\_parallel\_size}.
\]

With `n_gpus_per_node=8`, the example above aligns cleanly:

```text
node 0: [gsm8k replica 0: 2 GPUs] [gsm8k replica 1: 2 GPUs] [geo3k replica 0: 4 GPUs]
node 1: [geo3k replica 1: 4 GPUs] [geo3k replica 2: 4 GPUs]
```

No replica crosses a node boundary unless its `per_replica_world_size` requires multiple nodes.

A similar-looking configuration can fail if the replica sizes do not align with node boundaries. For example, if `gsm8k.tensor_model_parallel_size` were changed from `2` to `3`, then `gsm8k` replicas would occupy bundles `[0, 3)`, `[3, 6)`, `[6, 9)`, and so on. The replica covering `[6, 9)` would straddle node 0 and node 1, even though a 3-GPU replica is expected to fit within one 8-GPU node. In that case, validation raises and asks you to reorder teachers or adjust `num_replicas` / inference parallelism.

#### Teacher routing

The `teacher_key` controls routing. It must refer to a field in each sample's `extra_info`. In the example above, `teacher_key=data_source`, so samples with `data_source="openai/gsm8k"` are routed to the `gsm8k` teacher, and samples with `data_source="hiyouga/geometry3k"` are routed to the `geo3k` teacher.

When routing by data source, enable data shuffling. Without shuffling, a concatenated dataset may activate only one teacher for long contiguous stretches. For example, if GSM8K examples are followed by Geo3K examples, then training will use only the GSM8K teacher for the first portion of the epoch and only the Geo3K teacher for the remaining portion.

## Metrics

OPD logs metrics under `actor/distillation/*`.

### Core metrics

- `actor/distillation/loss`  
  Unscaled distillation loss. When `use_task_rewards=true`, compare this with `actor/pg_loss` to choose `distillation_loss_coef`.

- `actor/distillation/abs_loss`  
  Absolute value of the distillation loss. This is mainly useful for signed estimators such as `k1`, where the mean loss can be near zero even when individual token-level values are large.

- `actor/distillation/loss_min` / `actor/distillation/loss_max`  
  Minimum and maximum per-token distillation loss in the batch. Use these to detect outlier tokens or numerical instability.

### Top-\(k\) metrics

These metrics are logged for top-\(k\) loss modes such as `forward_kl_topk`.

- `actor/distillation/student_mass`  
  Average student probability mass assigned to the teacher top-\(k\) tokens.

- `actor/distillation/teacher_mass`  
  Average teacher probability mass assigned to its own top-\(k\) tokens.

- `actor/distillation/student_mass_min` / `actor/distillation/student_mass_max`  
  Minimum and maximum student mass on the teacher top-\(k\) tokens within the batch.

- `actor/distillation/teacher_mass_min` / `actor/distillation/teacher_mass_max`  
  Minimum and maximum teacher mass on the teacher top-\(k\) tokens within the batch.

`teacher_mass` indicates how much of the teacher distribution is covered by the selected top-\(k\). Low `teacher_mass` means the top-\(k\) approximation is truncating substantial teacher probability mass; increase `topk` if memory and runtime allow.

`student_mass` indicates how much probability the student assigns to the teacher-preferred tokens. During successful distillation, `student_mass` should generally move toward `teacher_mass`. A sharp drop in `student_mass`, especially with rising `loss`, can indicate instability or a token-alignment issue.

## Debugging

A useful technique for debugging modifications and additions to the distillation pipeline is to set the student to be the same model as the teacher. The loss should be approximately zero (not exact, since due to differences between train/inference engines). 

## Architecture

OPD has two components, mirroring RL:

1. **Teacher logprob computation** — runs on a dedicated teacher resource pool
   (`distillation.n_gpus_per_node × distillation.nnodes`, allocated in
   [`verl/trainer/main_ppo.py`](../../verl/trainer/main_ppo.py)).
2. **Student optimization** — runs on the train workers, the same actor workers
   that handle PPO/GRPO updates.

### Teacher logprob computation

Teacher logprob computation is interleaved with rollouts inside the **Agent
Loop**. Each sample's teacher call fires as soon as its rollout finishes — there
is no batch-wide barrier — so teacher work overlaps with the still-running
rollouts on other samples.

#### Step-by-step

1. **Input.** `AgentLoopManager.generate_sequences(prompts: DataProto)` receives
   a batch of prompts
   ([`verl/experimental/agent_loop/agent_loop.py`](../../verl/experimental/agent_loop/agent_loop.py)).

2. **Chunking across workers.** The manager splits the batch evenly across its
   `AgentLoopWorker` actors:
   `chunks = prompts.chunk(len(self.agent_loop_workers))`, then dispatches each
   chunk via `worker.generate_sequences.remote(chunk)`
   ([`agent_loop.py`](../../verl/experimental/agent_loop/agent_loop.py)).

3. **Per-sample fan-out inside a worker.** Inside
   `AgentLoopWorker.generate_sequences`, each sample in the chunk is launched as
   its own asyncio task:
   `asyncio.create_task(self._run_agent_loop(...))`
   ([`agent_loop.py`](../../verl/experimental/agent_loop/agent_loop.py)).
   The agent loop runs on the rollout GPUs and produces a rollout (prompt +
   response token ids).

4. **Postprocess hook.** `_run_agent_loop` calls
   `self._agent_loop_postprocess(output, …)`
   ([`agent_loop.py`](../../verl/experimental/agent_loop/agent_loop.py)).
   This is where teacher logprob computation is triggered, per sample, as soon
   as that sample's rollout is ready.

5. **Worker-side teacher dispatch.** `_agent_loop_postprocess` calls
   `self._compute_teacher_logprobs(output, prompt_ids=…, response_ids=…, …)`
   ([`agent_loop.py`](../../verl/experimental/agent_loop/agent_loop.py)).
   This method extracts the routing value from the sample's non-tensor fields
   using
   `sample_kwargs[self.teacher_key]` (default `teacher_key = "data_source"`),
   then calls
   `self.teacher_server_manager.compute_teacher_logprobs_single(...)`.

6. **Teacher selection.**
   `AsyncTeacherLLMServerManager.compute_teacher_logprobs_single`
   ([`verl/experimental/teacher_loop/teacher_manager.py`](../../verl/experimental/teacher_loop/teacher_manager.py))
   resolves the teacher via `_resolve_teacher_key`:

   - **Single-teacher**: routing key is ignored; the sole configured teacher is
     used.
   - **Multi-teacher**: `routing_key` must match a configured teacher in
     `distillation.teacher_models`; otherwise an error is raised.

   The resolved key indexes into `self.teacher_client: dict[str, LLMServerClient]`
   to pick the right client.

7. **Sampling params for scoring (not generation).** The manager builds sampling
   params via `_get_teacher_sampling_params`
   ([`teacher_manager.py`](../../verl/experimental/teacher_loop/teacher_manager.py)):
   `max_tokens=1` plus `prompt_logprobs=topk` (or `0`) — the teacher *scores* the
   (prompt + response) sequence rather than generating new tokens. `topk` is
   set to `distillation.distillation_loss.topk` when the loss mode requires
   top-k (e.g. `forward_kl_topk`); otherwise `0` (single-sample logprob only).

8. **Server-side load balancing.** The manager calls `client.generate(...)`.
   Inside `LLMServerClient.generate`
   ([`verl/workers/rollout/llm_server.py`](../../verl/workers/rollout/llm_server.py)),
   the client acquires a backing server through the shared
   `GlobalRequestLoadBalancer` actor:

   - **Sticky session**: if the `request_id` was seen before and the previously
     chosen server is still in the pool, route to it (preserves vLLM prefix
     cache hits across multi-turn).
   - **Else least-loaded**: pick the server with the fewest in-flight requests.

9. **Backend execution.** With the vLLM backend, the selected server is a
   `vLLMHttpServer` actor
   ([`verl/workers/rollout/vllm_rollout/vllm_async_server.py`](../../verl/workers/rollout/vllm_rollout/vllm_async_server.py)).
   Its `generate` method runs the forward pass and returns a `TokenOutput`
   containing `prompt_ids`
   and `prompt_logprobs` for the full (prompt + response) sequence. The SGLang
   backend has an analogous server class.

10. **Return path.** `compute_teacher_logprobs_single` packs the response into
    two tensors of shape `(S, 1 or K)` — `teacher_ids` and `teacher_logprobs`,
    where `S` is the sequence length and `K = topk` (or `1`). These are stashed
    in `output.extra_fields["teacher_ids"]` / `["teacher_logprobs"]` and later
    concatenated into the per-batch `DataProto` in `_postprocess` for the
    student optimization step.


### Student Optimization

Using the `DataProto` produced by the Agent Loop (rollouts + teacher logprobs in
`teacher_ids` / `teacher_logprobs`), the student step proceeds as follows.

#### Step-by-step

1. **Train entry.** `TrainingWorker.train_batch`
   ([`verl/workers/engine_workers.py`](../../verl/workers/engine_workers.py))
   invokes `self.engine.train_batch(data, loss_function=self.loss_fn)`. When
   distillation is enabled, `self.loss_fn` is bound to
   `distillation_ppo_loss` at worker init
   (`partial(distillation_ppo_loss, config=actor_config, distillation_config=…)`);
   otherwise it is the standard `ppo_loss`.

2. **Forward pass and (optional) inline top-k loss.**
   `FSDPEngineWithLMHead.forward_step`
   ([`verl/workers/engine/fsdp/transformer_impl.py`](../../verl/workers/engine/fsdp/transformer_impl.py))
   runs the model forward, then calls `prepare_model_outputs(...,
   logits_processor_func=loss_function)`. If the active loss mode requires
   top-k (`distillation_use_topk=True`), `prepare_model_outputs` invokes
   `distillation_ppo_loss(student_logits=…, data=…)` **as a logits processor**
   while the full logits tensor is still in memory. This is the
   `student_logits is not None` branch in `distillation_ppo_loss`
   ([`verl/trainer/distillation/losses.py`](../../verl/trainer/distillation/losses.py)),
   which dispatches to a backend-specific `compute_forward_kl_topk` (FSDP /
   Megatron). Per-token `distillation_losses`, `student_mass`, and
   `teacher_mass` tensors are written back into `model_output` so the full
   logits can be freed before the final loss step.

3. **Final loss.** `forward_step` then calls `loss_function(model_output=…,
   data=…, dp_group=…)` — this is the `student_logits is None` branch of
   `distillation_ppo_loss`, where:

   1. **Per-token distillation loss** is produced by `distillation_loss(...)`,
      which dispatches via `get_distillation_loss_fn(loss_mode)` to one of
      two registered families
      ([`losses.py`](../../verl/trainer/distillation/losses.py)):

      - **Top-k** (`forward_kl_topk`, `use_topk=True`): reads the pre-computed
        per-token tensors from `model_output` (populated by the logits
        processor in step 2) and logs `student_mass` / `teacher_mass`
        diagnostics. Negative divergences (a top-k truncation artifact) are
        clamped to 0.
      - **Single-sample KL estimators** (`kl`, `k1`, `abs`, `mse`, `k2`,
        `low_var_kl`, `k3`, `use_estimator=True`): compares the student's
        per-token `log_probs` (from the forward pass) directly against the
        teacher's single log-prob in `data["teacher_logprobs"]` via
        `kl_penalty`. No logits-processor pass is needed.

   2. **Optional clamp.** If `loss_max_clamp` is set, per-token losses are
      clamped to `[-clamp, +clamp]` (k1 in particular can be negative).

   3. **Aggregation mode** — controlled by `use_policy_gradient`:

      - `False` (supervised): aggregate per-token losses via `agg_loss` over
        the response mask — straight backprop, as in
        [arxiv 2306.13649](https://arxiv.org/abs/2306.13649).
      - `True` (on-policy distillation): treat `-distillation_losses` as
        advantages and run PPO-style clipped importance sampling against
        `data["old_log_probs"]`, as in
        [Thinking Machines' on-policy distillation post](https://thinkingmachines.ai/blog/on-policy-distillation/).

   4. **Combine with task rewards.** A standard PPO policy loss is computed
      from the rollout's task rewards via `ppo_loss(...)`. If
      `use_task_rewards=False` it is zeroed; otherwise the final loss is
      `policy_loss + distillation_loss_coef * distill_loss`.

The returned scalar loss is what `engine.train_batch` backpropagates.

## Files

### **Core Implementation**

- `verl/experimental/teacher_loop/teacher_model.py` — `MultiTeacherModelManager` and `TeacherModelManager`; spin up teacher inference replicas on the dedicated teacher resource pool and expose per-teacher `LLMServerClient` factories
- `verl/experimental/teacher_loop/teacher_manager.py` — `AsyncTeacherLLMServerManager`; routes per-sample teacher calls (single- or multi-teacher) and builds scoring sampling params (`max_tokens=1`, `prompt_logprobs=topk`)
- `verl/experimental/agent_loop/agent_loop.py` — `AgentLoopWorker._compute_teacher_logprobs`; per-sample teacher dispatch from `_agent_loop_postprocess`, packs `teacher_ids` / `teacher_logprobs` into the rollout output
- `verl/trainer/distillation/losses.py` — `distillation_ppo_loss`, `distillation_loss`, loss registry, top-k vs. estimator dispatch, policy-gradient vs. supervised aggregation, task-reward combination
- `verl/trainer/distillation/fsdp/losses.py` — FSDP backend `compute_forward_kl_topk`
- `verl/trainer/distillation/megatron/losses.py` — Megatron backend `compute_forward_kl_topk`
- `verl/workers/engine_workers.py` — `ActorRolloutRefWorker.init_model`; binds `distillation_ppo_loss` as the actor's `loss_fn` when distillation is enabled
- `verl/workers/engine/fsdp/transformer_impl.py` — `forward_step` / `prepare_model_outputs`; invokes `distillation_ppo_loss` first as a logits processor (top-k modes) and again as the final loss
- `verl/trainer/main_ppo.py` — `is_distillation_enabled` gate; allocates the dedicated `teacher_pool` resource pool
- `verl/trainer/ppo/ray_trainer.py` — constructs `MultiTeacherModelManager` and hands its `get_client()` dict to `AgentLoopWorker(... teacher_client=…)`
- `verl/workers/rollout/llm_server.py` — `LLMServerClient` and `GlobalRequestLoadBalancer` (sticky-session + least-loaded) used for both student rollout and teacher scoring

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
- `examples/on_policy_distillation_trainer/run_qwen3_mopd_gsm8k_geo3k.sh` — multi-teacher (one per dataset), routed by `data_source`

### **Tests**

- `tests/workers/test_distillation_topk_symmetry_on_cpu.py` — top-k loss symmetry checks
- `tests/utils/test_special_megatron_kl_loss_tp.py` — Megatron KL loss under tensor parallelism
- `tests/special_e2e/run_fully_async_policy_opd.sh` — end-to-end OPD with the fully-async rollouter