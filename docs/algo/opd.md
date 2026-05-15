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

1. OPD distills knowledge from teacher model(s) into a student model while aligning training-time states with inference-time states.
2. Compared with SFT or vanilla KD, OPD is on-policy: the student is trained on states induced by its own rollouts.
3. Compared with RLVR, OPD provides dense, continuous, token-level supervision rather than sparse outcome-level rewards.

### Knowledge Distillation

Knowledge distillation (KD) transfers behavior from a teacher model to a student model. For example, in mathematical reasoning, conventional KD samples reasoning traces and solutions from the teacher, then trains the student with a next-token prediction objective to match the teacher distribution.

This can introduce exposure bias. During training, the student only observes states sampled from the teacher. At inference time, however, states are sampled from the student. Unless the student and teacher induce identical state distributions, the student may not learn how the teacher would act in the states the student actually visits.

For example, the student may prefer algebraic proofs while the teacher prefers geometric proofs. Standard KD primarily distills the teacher's behavior along geometric-proof trajectories, even though the student may continue to generate algebraic-proof trajectories at inference time.

### On-Policy RL

RLVR addresses this mismatch by sampling rollouts from the student policy. If a rollout produces a correct final answer, the likelihood of the sampled solution is increased.

This aligns the training and inference distributions, but the reward is sparse and outcome-based. A rollout typically contributes only a binary success signal, applied at the sequence level rather than as dense token-level feedback.

### On-Policy Distillation

On-policy distillation (OPD) combines these advantages. The student samples rollouts from its own policy. Given each student-generated state, the teacher provides next-token log-probabilities. The student is then trained to match the teacher distribution at those student-induced states.

Thus, OPD aligns training and inference states while preserving the dense, continuous supervision of KD. Intuitively, the teacher provides guidance conditioned on the trajectory the student actually chose. If the student follows an algebraic proof path, the teacher supplies supervision for what it would do from that algebraic state.

Formally, let \(x \sim p_{\mathrm{data}}\) be a prompt, \(y \sim \pi_{\theta}(\cdot \mid x)\) be a student rollout, and \(s_t = (x, y_{<t})\) be the state at token \(t\). OPD minimizes

\[
\mathcal{L}_{\mathrm{OPD}}(\theta)
=
\mathbb{E}_{x \sim p_{\mathrm{data}},\, y \sim \pi_{\theta}(\cdot \mid x)}
\left[
\frac{1}{|y|}
\sum_{t=1}^{|y|}
D\!\left(
\pi_{\theta}(\cdot \mid s_t),
\nu(\cdot \mid s_t);
y_t
\right)
\right],
\]

where \(\pi_{\theta}\) is the student policy, \(\nu\) is the teacher policy, and \(D\) is either a divergence or a Monte Carlo estimator of a divergence.

### Choice of Divergence

We implement two OPD variants, following [1] and [3].

#### GKD OPD

Following [1], GKD OPD minimizes a full-vocabulary KL divergence between the teacher and student distributions. For forward KL,

\[
D_{\mathrm{KL}}\!\left(\nu \,\|\, \pi_{\theta}\right)
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

The distillation loss is optimized directly by backpropagating through this divergence.

#### PG OPD

Following [3], PG OPD treats a negative reverse-KL estimate as a reward and optimizes it with a policy-gradient objective. Since tokens are sampled from the student policy, the per-token Monte Carlo estimator of reverse KL is

\[
\widehat{D}_{\mathrm{KL}}\!\left(\pi_{\theta} \,\|\, \nu\right)
=
\operatorname{sg}\!\left(
\log \pi_{\theta}(y_t \mid s_t)
-
\log \nu(y_t \mid s_t)
\right),
\quad
y_t \sim \pi_{\theta}(\cdot \mid s_t).
\]

Equivalently, the reward for maximizing negative reverse KL is

\[
r_t
=
\operatorname{sg}\!\left(
\log \nu(y_t \mid s_t)
-
\log \pi_{\theta}(y_t \mid s_t)
\right).
\]

**Note**: When using the k1 estimator, the stop-gradient is required so that the reward is treated as fixed under the policy-gradient update; otherwise, the loss will not depend on the teacher logprob due to differentiation with respect to student parameters.


### Multi-Teacher OPD

Multi-teacher OPD (MOPD) extends OPD to multiple domain-specialized teachers. This approach has recently been used as a post-training approach in several models [4,5,6,7].

A base model is first trained or adapted independently across domains such as math, coding, and instruction following, producing one expert teacher per domain. A student is then trained on a mixture of data from those domains. For each example, the student matches the log-probabilities of the corresponding domain expert on the student-induced states.

This allows the student to consolidate multiple specialized policies into a single model while preserving the on-policy alignment benefits of OPD.

### Bibliography

[1] Agarwal, Rishabh, et al. "On-policy distillation of language models: Learning from self-generated mistakes." International Conference on Learning Representations. Vol. 2024. 2024.

[2] Yang, An, et al. "Qwen3 technical report." arXiv preprint arXiv:2505.09388 (2025).

[3] Lu, Kevin and Thinking Machines Lab, "On-Policy Distillation", Thinking Machines Lab: Connectionism, Oct 2025.

[4] Xiao, Bangjun, et al. "Mimo-v2-flash technical report." arXiv preprint arXiv:2601

[5] Zeng, Aohan, et al. "Glm-5: from vibe coding to agentic engineering." arXiv preprint arXiv:2602.15763 (2026).

[6] Yang, Zhuolin, et al. "Nemotron-Cascade 2: Post-Training LLMs with Cascade RL and Multi-Domain On-Policy Distillation." arXiv preprint arXiv:2603.19220 (2026).

[7] DeepSeek-AI. "DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence." (2026) 


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

### Quick start

This shows the minimal setup for single teacher OPD. Enable distillation and specify resources for the teacher servers:

```yaml
distillation:
   enabled: true
   n_gpus_per_node: 2
   nnodes: 1
```

Specify the teacher model name and server settings. 

**NOTE**: the teacher model must have the same vocab as the student model, e.g., Qwen3-8B student and Qwen3-32B teacher. This is usually true if models are in the same family as each other. It can be verified by comparing tokenizers. Note that the model heads might have slightly different output dimensions due to padding, although this is not an issue. 

```yaml
distillation:
   teacher_models:
      teacher_model:
         model_path: Qwen/Qwen3-32B
         inference:
            name: vllm
            gpu_memory_utilization: 0.8
```

Note that the reference policy in GRPO/PPO applies a reverse KL distillation loss to the student policy to distill it from the reference policy. In most cases, this should be disabled by ensuring that

```yaml
actor_rollout_ref:
   actor:
      use_kl_loss: false
algorithm: 
   use_kl_in_reward: false
```

### GKD OPD

As of May 14, 2026, the implementation only supports computing of the GKD OPD loss over the teacher top-\(k\) logits. Thus, the implemented objective is a top-\(k\) approximation to forward KL. Earlier implementations attempted reverse KL using the student top-\(k\) logits, but this was unstable for Qwen2.5-0.5B. Additionally, the current implementation does not allow for computing the student top-\(k\) logits because the teacher server does not allow for gathering at specified token IDs, only the sampled token and the top-\(k\) tokens.  

To use GKD OPD, set the loss mode, the top-\(k\) value, and disable policy gradient. 

```yaml
distillation:
   distillation_loss: 
      loss_mode: forward_kl_topk
      topk: 128
      use_policy_gradient: false
```

**Note**: It is also important to not use policy gradient, since policy gradient only directly influences increases/decreases the logprob of the sampled token to match the teacher logprob, whereas the top-\(k\) loss includes signal for at least \(k-1\) other tokens. Using policy gradient is therefore not only computationally wasteful, but also adds noise to the reward. For example, consider the case where the student has perfectly matched the logprob of the sampled token relative to the teacher, but for all other tokens in the top-\(k\), it has overestimated. The forward KL is therefore positive, so policy gradient will decrease the logprob of the sampled token, despite already matching.


**TODO: add a math eqn summarizing this**

Put another way (rm this note after editting):

```python
        if self.use_policy_gradient and self.loss_mode == "forward_kl_topk":
            print(
                "WARNING: forward_kl_topk is most effective as a supervised distillation loss "
                "(use_policy_gradient=False). With policy gradient, the update uses only the sampled"
                " token's logprob ∇logπ(a), so the top-k distributional signal (how non-sampled logits "
                "should move) is largely unused."
            )

```


### PG OPD

To use PG OPD with k1 estimator, set the loss mode, enable policy gradient, and coonfigure clipping:

```yaml
distillation:
   distillation_loss: 
      loss_mode: k1
      use_policy_gradient: true
      policy_loss_mode: vanilla
      clip_ratio_low: 0.2
      clip_ratio_high: 0.28
```

**Note**: currently only `policy_loss_mode=vanilla` is supported. Other loss modes such as `dppo_tv` require additional parameters, such as `clip_ratio_c`.

**Note**: Computing the distillation loss using KL estimator is valid for reverse KL because samples are drawn from the student distribution. Estimating forward KL would instead require samples from the teacher distribution, which is just conventional (off-policy) KD. 

### Task rewards

Task rewards can be optimized at the same time as distillation loss by adding together as:

TODO: make the eqn better. 

$$
\mathcal L = \mathcal L_{task} + \lambda \mathcal L_{distillation},
$$

where $\lambda$ is the parameter controlling the strength of the distillation loss.

To simultaneously optimize task rewards (e.g., RLVR rewards) with a distillation loss, enable task rewards and set coefficient for the distillation loss:

```yaml
distillation:
   distillation_loss:
      use_task_rewards: true
      distillation_loss_coef: 1.5
```

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

**Note**: TODO: add a note about teacher GPU placement

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