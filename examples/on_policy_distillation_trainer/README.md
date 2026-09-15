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

## Tail-aware top-k KL oracle

`benchmark_tail_kl.py` compares the legacy truncated teacher-top-k objective
and `forward_kl_topk_tail` against a full-vocabulary forward-KL oracle. It
reports value error, gradient cosine similarity, and the fraction of negative
truncated per-token losses without downloading a model:

```bash
uv run --extra cpu python examples/on_policy_distillation_trainer/benchmark_tail_kl.py \
  --tokens 64 --vocab-size 8192 --topk 8 16 32 64
```

To run an existing trainer script with the tail-aware objective, disable its
legacy per-entry log-probability clamp explicitly:

```bash
DISTILLATION_LOSS_MODE=forward_kl_topk_tail USE_POLICY_GRADIENT=False \
  bash examples/on_policy_distillation_trainer/run_qwen3_8b_fsdp.sh \
  distillation.distillation_loss.log_prob_min_clamp=null
```
