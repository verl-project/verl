# On-Policy Distillation

This trainer jointly trains a student model with policy-gradient on-policy rollouts and a distillation loss against a frozen teacher model served by a separate Ray cluster. Compared to pure SFT from teacher generations, on-policy distillation typically closes more of the teacher/student gap at the same compute budget.

## Canonical Scripts

| Script                              | Teachers                         | Modality   | Infer | Train    | Platform |
|-------------------------------------|----------------------------------|------------|-------|----------|----------|
| `run_qwen3_8b_fsdp.sh`              | single                           | text       | vLLM  | FSDP     | NVIDIA   |
| `run_qwen3_8b_megatron.sh`          | single                           | text       | vLLM  | Megatron | NVIDIA   |
| `run_qwen3_vl_8b_fsdp.sh`           | single                           | VL         | vLLM  | FSDP     | NVIDIA   |
| `run_qwen3_8b_mopd_fsdp.sh`         | multi (resident)                 | text + VL  | vLLM  | FSDP     | NVIDIA   |
| `run_qwen3_8b_mopd_on_demand.sh`    | multi (shared fleet / occupancy) | text + VL  | vLLM  | FSDP     | NVIDIA   |

Override `STUDENT_MODEL` and `TEACHER_MODEL` via env vars to swap model pairs in
the single-teacher scripts. The MOPD scripts expose per-teacher overrides.

`run_qwen3_8b_mopd_on_demand.sh` is the same gsm8k + geo3k recipe as
`run_qwen3_8b_mopd_fsdp.sh`, but sets `distillation.teacher_serving=on_demand`
so both teachers share one replica fleet. Teacher cards stay fixed when you add
teachers; a replica is donated to the next unstarted routing key only after its
current teacher finishes the step.

## Key Flags

- `distillation.enabled=True`
- `distillation.teacher_serving={resident, on_demand}` (default `resident`)
- `distillation.teacher_models.teacher_model.model_path=<HF path>` (single-teacher)
- `+distillation.teacher_models.<name>.{key,model_path,num_replicas,inference.*}` (multi-teacher)
- `distillation.distillation_loss.loss_mode={k1, k3, forward_kl_topk, forward_kl_topk_tail, ...}`
- `distillation.distillation_loss.use_policy_gradient=True|False`
- `distillation.distillation_loss.topk=64`
