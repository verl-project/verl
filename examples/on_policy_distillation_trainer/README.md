# On-Policy Distillation

This trainer jointly trains a student model with policy-gradient on-policy rollouts and a distillation loss against a frozen teacher model served by a separate Ray cluster. Compared to pure SFT from teacher generations, on-policy distillation typically closes more of the teacher/student gap at the same compute budget.

## Canonical Scripts

| Script                          | Teachers | Modality   | Infer | Train    | Platform |
|---------------------------------|----------|------------|-------|----------|----------|
| `run_qwen3_8b_fsdp.sh`          | single   | text       | vLLM  | FSDP     | NVIDIA   |
| `run_qwen3_8b_megatron.sh`      | single   | text       | vLLM  | Megatron | NVIDIA   |
| `run_qwen3_vl_8b_fsdp.sh`       | single   | VL         | vLLM  | FSDP     | NVIDIA   |
| `run_qwen3_8b_mopd_fsdp.sh`     | multi    | text + VL  | vLLM  | FSDP     | NVIDIA   |

Override `STUDENT_MODEL` and `TEACHER_MODEL` via env vars to swap model pairs in
the single-teacher scripts. The MOPD script exposes per-teacher overrides.

## Key Flags

- `distillation.enabled=True`
- `distillation.teacher_models.teacher_model.model_path=<HF path>` (single-teacher)
- `+distillation.teacher_models.<name>.{key,model_path,num_replicas,inference.*}` (multi-teacher)
- `distillation.distillation_loss.loss_mode={k1, k3, forward_kl_topk, forward_kl_topk_tail, ...}`
- `distillation.distillation_loss.use_policy_gradient=True|False`
- `distillation.distillation_loss.topk=64`
- `distillation.rollout_source_selector.{path,name,kwargs}` (optional user function; absent by default)

## Teacher-generated responses

The experimental `verl.experimental.teacher_rollout` feature uses the native synchronous V1 trainer to reuse the frozen vLLM teacher for response generation, then score the complete response with that teacher for supervised soft-target distillation. A user function selects `"student"` or `"teacher"` for each training submission from its prompt batch, global step, and rollout seed. The trainer imposes no balancing algorithm. Without a selector, generation remains student-only; validation always uses the student.

The built-in `select_source` ([source](../../verl/experimental/teacher_rollout.py)) implements fixed-probability batch mixing:

```bash
DISTILLATION_LOSS_MODE=forward_kl_topk USE_POLICY_GRADIENT=False \
  bash examples/on_policy_distillation_trainer/run_qwen3_8b_fsdp.sh \
  distillation.rollout_source_selector.path=pkg://verl.experimental.teacher_rollout \
  distillation.rollout_source_selector.name=select_source \
  +distillation.rollout_source_selector.kwargs.teacher_probability=1.0 \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.calculate_log_probs=False \
  distillation.distillation_loss.loss_max_clamp=null \
  distillation.distillation_loss.log_prob_min_clamp=null
```

Change `teacher_probability` to `0.5` for batch mixing, or supply your own function via `path` and `name`. It receives keyword arguments `batch`, `global_step`, `seed`, and configured `kwargs`, and must return one source for the whole submission. Policies own deterministic decisions and any checkpoint state; this initial interface does not support per-row decisions or token-level interleaving.

Initial support is text-only single-turn generation with one teacher, vLLM, and supervised `forward_kl_topk` at temperature `1.0`. Student and teacher must share tokenizer, token IDs, vocabulary, and EOS semantics. Teacher generation uses the student's sampling controls and full response budget, followed by a separate teacher scoring pass; it therefore adds teacher decoding cost. Policy-gradient/task-reward combinations, rollout logprobs, custom agent loops/replay samplers, refilling failed groups, multimodal inputs, and async training are rejected when a selector is configured. Rollout correction, auxiliary KL/entropy, group filtering, router replay, and speculative decoding are not validated with teacher-generated rollouts; do not combine them with a selector. See [the OPD guide](../../docs/algo/opd.md#experimental-teacher-generated-training-rollouts) for the full contract. The launcher's model and GPU sizes are examples, not a CPU smoke test.
