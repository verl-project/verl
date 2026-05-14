# On-Policy Distillation

**Author:** [Jacob Helwig](https://jacobhelwig.github.io/)

Last updated: 05/14/2026.

---

> **Contents**
>
> - **Background** - Introduction to On-Policy Distillation
> - **Architecture** - Implementation and control flow of OPD 
> - **Usage** - Configuring OPD runs

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

**Implementation note:** as of May 14, 2026, the implementation only supports computing this loss over the teacher top-\(k\) logits. Thus, the implemented objective is a top-\(k\) approximation to forward KL. Earlier implementations attempted reverse KL using the student top-\(k\) logits, but this was unstable for Qwen2.5-0.5B.

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

This estimator is valid for reverse KL because samples are drawn from the student distribution. Estimating forward KL would instead require samples from the teacher distribution. The stop-gradient is required so that the reward is treated as fixed under the policy-gradient update; otherwise, the optimization no longer corresponds to the intended score-function estimator.

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