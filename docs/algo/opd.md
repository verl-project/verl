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

OPD configuration lives under three namespaces:

- `distillation.*` — top-level switches and teacher resource-pool settings ([`DistillationConfig`](../../verl/workers/config/distillation.py))
- `distillation.teacher_models.<name>.*` — per-teacher settings ([`DistillationTeacherModelConfig`](../../verl/workers/config/distillation.py))
- `distillation.distillation_loss.*` — loss-mode, aggregation, and clipping settings ([`DistillationLossConfig`](../../verl/workers/config/distillation.py))

Defaults below come from [`verl/trainer/config/distillation/distillation.yaml`](../../verl/trainer/config/distillation/distillation.yaml).

---

### Top-level settings

#### `distillation.enabled` (bool)

Default: `false`.

Enables on-policy distillation. When `true`, `main_ppo` allocates a teacher resource pool, starts one or more teacher inference servers, and uses `distillation_ppo_loss` instead of the standard `ppo_loss`.

#### `distillation.n_gpus_per_node` (int)

Default: `8`.

Number of GPUs per node in the teacher resource pool.

#### `distillation.nnodes` (int)

Default: `0`.

Number of nodes in the teacher resource pool. Set this to at least `1` when `distillation.enabled=true`.

The total teacher pool size must equal the total GPU footprint of all teacher replicas:

\[
\texttt{n\_gpus\_per\_node}
\times
\texttt{nnodes}
=
\sum_i
\texttt{num\_replicas}_i
\times
\texttt{per\_replica\_world\_size}_i .
\]

Otherwise, `DistillationConfig.__post_init__` raises.

#### `distillation.teacher_key` (str)

Default: `"data_source"`.

Sample field used to route examples to teachers in multi-teacher setups.

- In single-teacher mode, this field is ignored.
- In multi-teacher mode, `sample[distillation.teacher_key]` must match the `key` of one configured teacher.

If no matching teacher exists, `AsyncTeacherLLMServerManager._resolve_teacher_key` raises.

#### `distillation.teacher_models` (dict)

Map from teacher names to `DistillationTeacherModelConfig` entries.

For a single teacher, the default entry is conventionally named `teacher_model`. In multi-teacher configs, do not keep the default `teacher_model` entry alongside additional named teachers: it is removed during config resolution. Rename it and add all teachers explicitly.

```bash
# Wrong: teacher_model is removed, so only teacher_model2 is used.
distillation.teacher_models.teacher_model.key=openai/gsm8k
distillation.teacher_models.teacher_model.model_path=Qwen/Qwen3-4B
+distillation.teacher_models.teacher_model2.key=hiyouga/geometry3k
+distillation.teacher_models.teacher_model2.model_path=Qwen/Qwen3-VL-4B-Instruct

# Correct: explicitly name every teacher.
+distillation.teacher_models.teacher_model1.key=openai/gsm8k
+distillation.teacher_models.teacher_model1.model_path=Qwen/Qwen3-4B
+distillation.teacher_models.teacher_model2.key=hiyouga/geometry3k
+distillation.teacher_models.teacher_model2.model_path=Qwen/Qwen3-VL-4B-Instruct
```

---

### Teacher-model settings

#### `distillation.teacher_models.<name>.key` (str)

Default: `null`.

Routing key for this teacher. In multi-teacher mode, this must match the value of `sample[distillation.teacher_key]`.

For single-teacher configs, this is automatically set to `"default"` if unset.

#### `distillation.teacher_models.<name>.model_path` (str)

Required.

Local path or Hugging Face model ID for the teacher.

The teacher must share the student's tokenizer and vocabulary. This is usually satisfied by using models from the same family, such as a `Qwen3-32B` teacher with a `Qwen3-8B` student. Different LM-head padding is allowed.

#### `distillation.teacher_models.<name>.num_replicas` (int)

Default: `0`.

Number of inference replicas to launch for this teacher.

Each replica uses

\[
\texttt{per\_replica\_world\_size}
=
\texttt{tensor\_model\_parallel\_size}
\times
\texttt{data\_parallel\_size}
\times
\texttt{pipeline\_model\_parallel\_size}.
\]

The teacher's total GPU footprint is therefore

\[
\texttt{num\_replicas}
\times
\texttt{per\_replica\_world\_size}.
\]

For a single teacher, `num_replicas=0` is allowed: `_resolve_teacher_models` fills it as `pool_size // per_replica_world_size`.

For multi-teacher configs, set `num_replicas` explicitly so that the total footprint matches the teacher resource pool size.

#### `distillation.teacher_models.<name>.inference.*` ([`RolloutConfig`](../../verl/workers/config/rollout.py))

Teacher inference-server configuration. This has the same shape as `actor_rollout_ref.rollout.*`.

Common fields:

- `inference.name` — inference backend, such as `vllm` or `sglang`
- `inference.tensor_model_parallel_size` — default: `2`
- `inference.gpu_memory_utilization` — default: `0.5`
- `inference.max_model_len` — must be at least `student_prompt_length + student_response_length + 1`
- `inference.engine_kwargs.vllm.max_logprobs` — automatically raised to at least `distillation.distillation_loss.topk` when the active loss requires top-\(k\) teacher log-probabilities

For SGLang, `max_logprobs` is not used in the same way; top-logprob count is supplied per request.

During validation, `validate_and_prepare_for_distillation` rewrites the teacher rollout lengths:

```text
inference.prompt_length   := student_prompt_length + student_response_length
inference.response_length := 1
```

The teacher scores the full student-generated prefix and emits one dummy token.

---

### Distillation-loss settings

#### `distillation.distillation_loss.loss_mode` (str)

Default: `"k3"`.

Selects the distillation loss or estimator.

Supported families:

- **Top-\(k\) distributional loss**
  - `forward_kl_topk`
  - Uses the teacher's top-\(k\) logits to approximate forward KL.
  - Computed inline during the student forward pass so full-vocabulary logits do not need to leave the GPU.

- **Sampled-token estimators / penalties**
  - `kl`, `k1`, `abs`, `mse`, `k2`, `low_var_kl`, `k3`
  - Computed from the student's sampled-token `log_probs` and the teacher's log-probability for the same sampled token.

#### `distillation.distillation_loss.topk` (int, optional)

Default: `32`.

Number of teacher top-logprobs to request for top-\(k\) loss modes such as `forward_kl_topk`.

This controls both:

- the teacher `prompt_logprobs` request size
- the vLLM `max_logprobs` cap

It is ignored by sampled-token loss modes.

#### `distillation.distillation_loss.use_task_rewards` (bool)

Default: `true`.

Controls whether the standard PPO/GRPO task-reward loss is combined with the distillation loss.

- `true`: use task rewards and distillation:

  \[
  \mathcal{L}
  =
  \mathcal{L}_{\mathrm{policy}}
  +
  \lambda_{\mathrm{distill}}
  \mathcal{L}_{\mathrm{distill}} .
  \]

- `false`: zero the PPO/GRPO task-reward loss and optimize only the distillation loss.

This is independent of `use_policy_gradient`, which controls how the distillation signal itself is applied.

#### `distillation.distillation_loss.distillation_loss_coef` (float)

Default: `1.0`.

Coefficient \(\lambda_{\mathrm{distill}}\) applied to the distillation loss when `use_task_rewards=true`.

#### `distillation.distillation_loss.loss_max_clamp` (float, optional)

Default: `null`.

If set, clamps each per-token distillation loss to

\[
[-\texttt{loss\_max\_clamp},\ \texttt{loss\_max\_clamp}].
\]

This is useful for limiting outlier tokens. It is especially relevant for `k1`, which can be negative.

#### `distillation.distillation_loss.log_prob_min_clamp` (float, optional)

Default: `null`.

If set, lower-clamps log-probabilities before divergence computation. This prevents large values from near-zero probabilities in terms such as `log q - log p`.

Example scripts commonly set this to `-10.0`.

#### `distillation.distillation_loss.use_policy_gradient` (bool)

Default: `false`.

Controls how the per-token distillation signal is applied.

- `false`: supervised distillation.
  - Aggregate the per-token distillation loss over the response mask.
  - Backpropagate through the resulting loss directly.
  - Recommended for `forward_kl_topk` and `k3`.

- `true`: policy-gradient distillation.
  - Treat `-distillation_loss` as an advantage.
  - Apply a PPO-style clipped importance-sampling update against `data["old_log_probs"]`.
  - Recommended for `k1`.

Validation behavior:

- `use_policy_gradient=false` with `loss_mode="k1"` raises `ValueError`, because direct backpropagation through the `k1` sampled-token estimator is not meaningful.
- `use_policy_gradient=true` with `loss_mode="forward_kl_topk"` emits a warning, because the policy-gradient update only acts through the sampled token and does not use most of the top-\(k\) distributional signal.

#### `distillation.distillation_loss.policy_loss_mode` (str)

Default: `"vanilla"`.

Policy-loss variant used when `use_policy_gradient=true`.

Currently, only `"vanilla"` is supported. Other values raise `NotImplementedError`.

#### `distillation.distillation_loss.clip_ratio` (float)

Default: `0.2`.

PPO clip ratio used by the policy-gradient distillation update when `use_policy_gradient=true`.

#### `distillation.distillation_loss.clip_ratio_low` (float)

Default: `0.2`.

Lower PPO clip bound used when `use_policy_gradient=true`.

#### `distillation.distillation_loss.clip_ratio_high` (float)

Default: `0.2`.

Upper PPO clip bound used when `use_policy_gradient=true`.

#### `distillation.distillation_loss.global_batch_info` / `loss_settings`

Internal runtime fields. Do not set these manually.

- `loss_settings` is populated from `loss_mode` by `get_distillation_loss_settings`.
- `global_batch_info` is populated by the actor worker before loss computation.

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

Multiple teachers can be specified as

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

**Note**: TODO: add a note about teacher GPU placement. follow the function below to highlight how placement is done. rm the function code after this TODO is satisfied. Relate the explanation to the configuration snippet above by explaining why the snippet works and telling how it could be modified not to work anymore

```
    def _validate_replica_node_alignment(self, replica_pools, per_replica_world_size, gpus_per_node):
        """Verify that each replica occupies the expected number of nodes.

        `per_replica_world_size` (W below) is the GPU count of a *single* inference
        replica — the product of the replica's inference-time parallelism
        (tensor_model_parallel_size * data_parallel_size * pipeline_model_parallel_size).
        It is not the teacher's total GPU footprint (`num_replicas * W`).

        `split_resource_pool` walks bundles linearly and is oblivious to node
        boundaries, so a replica's sub-pool can end up touching more nodes than W
        implies when W does not divide the node layout cleanly.

        Example (P = n_gpus_per_node = 4, two teachers with W=3 and W=4):

                node 0                  node 1
                [0 1 2 3]               [4 5 6 7]         ← bundle idx

            teacher A (W=3):
                [A A A .]               [. . . .]          expected span 1, observed 1  ✓
            teacher B (W=4):
                [. . . B]               [B B B .]          expected span 1, observed 2  ✗

        Teacher B's one replica (W=4) is expected to stay on a single node, but the
        linear split dropped it on bundles 3-6 — straddling nodes 0 and 1.
        """
        key = self.teacher_model_config.key
        P = gpus_per_node
        W = per_replica_world_size
        expected_span = (W + P - 1) // P
        for i, sub_pool in enumerate(replica_pools):
            start = sub_pool.start_bundle_index
            first_node = start // P
            last_node = (start + W - 1) // P
            observed_span = last_node - first_node + 1
            if observed_span != expected_span:
                raise ValueError(
                    f"Teacher {key!r} replica {i} sub-pool bundles [{start}, {start + W}) "
                    f"span {observed_span} node(s) but per_replica_world_size {W} with "
                    f"n_gpus_per_node {P} expects {expected_span}. Reorder teachers or "
                    f"adjust num_replicas / inference parallelism so each replica sub-pool "
                    f"aligns to node boundaries."
                )
```

**Note**: The `teacher_key` is used to route examples to teachers and can be any string that is in the `extra_info` of each example. If examples are routed based on data source, i.e., `teacher_key == data_source`, make sure to shuffle the data. Otherwise, only one teacher will be active. For example, if the data is GSM8k concatenated with Geo3k, the first 8/11~73% of training will only use the GSM8k teacher, and the remaining 27% will use the Geo3k teacher.



## Metrics

- `actor/distillation/abs_loss`: absolute value of distillation loss. Useful for k1 estimator, which can be negative.
- `actor/distillation/loss_{min,max}`: min/max distillation loss in a batch.
- `actor/distillation/loss`: Unscaled distillation loss. Compare the magnitude to `actor/pg_loss` when `use_task_rewards=True` to determine the value of `distillation_loss_coef`.
- `actor/distillation/{student,teacher}_mass`: average sum of probabilities for the student/ teacher in the teacher top-\(k\) when using a top-\(k\) loss. Decreasing can indicate instability.
- `actor/distillation/{student,teacher}_mass_{min,max}`: min/max sum of probabilities for the student/teacher in the teacher top-\(k\) when using a top-\(k\) loss. Decreasing can indicate instability.

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