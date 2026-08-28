# On-Policy Distillation

This trainer jointly trains a student model with policy-gradient on-policy rollouts and a distillation loss against a frozen teacher model served by a separate Ray cluster. Compared to pure SFT from teacher generations, on-policy distillation typically closes more of the teacher/student gap at the same compute budget.

## Canonical Scripts

| Script                              | Teachers | Modality  | Infer | Train    | Platform   |
| ----------------------------------- | -------- | --------- | ----- | -------- | ---------- |
| `run_qwen3_8b_fsdp.sh`              | single   | text      | vLLM  | FSDP     | NVIDIA     |
| `run_qwen3_8b_megatron.sh`          | single   | text      | vLLM  | Megatron | NVIDIA     |
| `run_qwen3_vl_8b_fsdp.sh`           | single   | VL        | vLLM  | FSDP     | NVIDIA     |
| `run_qwen3_8b_mopd_fsdp.sh`         | multi    | text + VL | vLLM  | FSDP     | NVIDIA     |
| `run_qwen2_5_0_5b_megatron.sh`      | single   | text      | vLLM  | Megatron | GPU / NPU  |
| `run_qwen3_vl_megatron.sh`          | single   | VL        | vLLM  | Megatron | GPU / NPU  |
| `run_qwen3_vl_2b_megatron.sh`       | single   | VL        | vLLM  | Megatron | GPU / NPU  |

Override `STUDENT_MODEL` and `TEACHER_MODEL` via env vars to swap model pairs in
the single-teacher scripts. The MOPD script exposes per-teacher overrides.
`DEVICE` is auto-detected from `torch_npu`; set `DEVICE=gpu` or `DEVICE=npu`
only to override detection. `run_qwen3_vl_2b_megatron.sh` is the official
Geo3K calibration pair: a Qwen3-VL-2B student with a Qwen3-VL-4B teacher,
2048-token responses, top-k 64, and no task reward. Hardware-specific resource
and memory defaults are selected inside the scripts.

See the [Ascend Megatron + vLLM OPD guide](../../docs/ascend_tutorial/model_support/examples/opd_megatron_vllm_ascend.md)
for dependency compatibility, four-NPU placement, validation, and tuning.

## Key Flags

- `distillation.enabled=True`
- `distillation.teacher_models.teacher_model.model_path=<HF path>` (single-teacher)
- `+distillation.teacher_models.<name>.{key,model_path,num_replicas,inference.*}` (multi-teacher)
- `distillation.distillation_loss.loss_mode={k1, k3, forward_kl_topk, ...}`
- `distillation.distillation_loss.use_policy_gradient=True|False`
- `distillation.distillation_loss.topk=64`
