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

## Hidden-state transfer for full-vocabulary KL

Set the following overrides for a vLLM teacher and an unfused Megatron or FSDP actor.
Each trajectory uses the teacher identified by `distillation.teacher_key`.
The teacher reads the full token sequence and sends the last-layer rows needed to
predict the selected response tokens through TransferQueue SimpleStorage.

```text
distillation.teacher_execution.mode=dedicated_hidden_state
distillation.distillation_loss.loss_mode=forward_kl_hidden_state
distillation.distillation_loss.use_policy_gradient=False
distillation.distillation_loss.use_task_rewards=False
actor_rollout_ref.model.use_fused_kernels=False
```

Use `dedicated_hidden_state` for resident teachers of different hidden sizes with
a common vocabulary. For compatible checkpoints, `weight_swapped_hidden_state`
loads one teacher at a time into a shared pool. Set
`distillation.teacher_execution.transfer_signal=full_logits` in dedicated mode
to compare BF16 vocabulary-logit transfer on the same target positions.

The teacher transfers one prediction per token selected by the final response
and loss masks. A custom agent can set `AgentLoopOutput.loss_mask` to one
binary entry per response token and clear any positions it wants to exclude.
By default, all eligible response tokens contribute. Rollout rejection can
further narrow the selection. KL and its token count use the same selection.

Prompts and tool or environment replies are context. For a selected token at
sequence position `t`, the teacher sends hidden row `t - 1`. For example, the
last tool-reply row transfers when it predicts the first supervised token of
the next assistant turn. The rest of that tool reply adds no hidden rows to TQ.

On multi-turn Qwen3.5 (`qwen3_coder`) and Hermes rollouts,
`distillation.teacher_execution.skip_tool_call_tokens=True` also excludes
assistant-generated call delimiters and arguments as targets.
