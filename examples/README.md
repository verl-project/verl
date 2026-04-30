# verl Examples

This directory hosts curated, minimal-dependency examples that drive
`verl.trainer.main_ppo` with the current Hydra API. Algorithm-specific
extensions, research baselines, and non-trivial entry points live under
`recipe/`; prefer that directory if you need a custom loss or reward beyond
what these examples show.

## Conventions

All run scripts follow the same shape:

1. Canonical filename:

   ```
   run_<model>_<train-backend>.sh
   ```

   - `<model>`: a single canonical size per model family. E.g.
     `qwen3_8b`, `qwen3_30b_a3b`, `qwen3_235b_a22b`, `qwen3_vl_8b`,
     `deepseek_v3`, `mimo_7b`, `nemotron_nano_v3`.
   - `<train-backend>`: one of `fsdp`, `fsdp2`, `megatron`, `mindspeed`,
     `automodel`, or `veomni`. **Must be the last underscore-separated
     token before `.sh`**.

   Nothing follows `<train-backend>`. Per-example *features* — including
   the inference backend (`vllm`/`sglang`/`trtllm`), the platform
   (`DEVICE=gpu|npu`), the GPU machine type (`MACHINE=gb200`/`b200`/
   `blackwell`), Liger kernel, LoRA, FP8 quantization, sequence parallel
   size, server vs sync rollout, etc. — do **not** show up in filenames.
   They are exposed as env-var toggles inside the one canonical script.
   Do not add `_npu`, `_amd`, `_vllm`, `_sglang`, `_trtllm`, or `_fp8`
   script variants. For example, `sft/gsm8k/run_qwen2_5_0_5b_fsdp.sh`
   covers plain SFT and its `USE_LIGER=1`, `SP_SIZE=2`, `USE_PEFT=1`
   variants via env vars; `grpo_trainer/run_qwen3_8b_fsdp.sh` covers vLLM,
   SGLang, and TRT-LLM rollouts, CUDA/NPU platforms, and `MACHINE=gb200`
   (Blackwell) via toggles.

   This naming rule is enforced by the `check-example-naming` pre-commit
   hook (see `tests/special_sanity/check_example_naming.py`).

2. Every script exposes its important knobs in a user-adjustable region near
   the top. Derived defaults and device/backend-specific details belong below
   the "no user adjustment needed below" / "derived defaults" boundary.
   Use uppercase env vars for user-facing knobs, e.g.

   ```bash
   # ---- user-adjustable ----
   DEVICE=${DEVICE:-gpu}
   INFER_BACKEND=${INFER_BACKEND:-vllm}
   MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-8B}
   NNODES=${NNODES:-1}
   NDEVICES_PER_NODE=${NDEVICES_PER_NODE:-}
   TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-1024}
   ROLLOUT_TP=${ROLLOUT_TP:-2}
   ROLLOUT_N=${ROLLOUT_N:-5}
   PROJECT_NAME=${PROJECT_NAME:-verl_grpo_gsm8k_math}
   EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_8b_grpo_vllm_fsdp}
   # ---- end user-adjustable ----
   ...
   ```

   Override anything you care about on the command line:

   ```bash
   DEVICE=npu MODEL_PATH=/my/local/qwen3-8b NDEVICES_PER_NODE=4 bash examples/grpo_trainer/run_qwen3_8b_fsdp.sh
   ```

   GPU and NPU paths should share the same `PROJECT_NAME` /
   `EXPERIMENT_NAME` form. Do not append `_npu` to project or experiment
   names just because `DEVICE=npu` is selected.

3. Defaults (unless a directory explicitly documents otherwise):

   - `data.train_files` + `data.val_files` = GSM8K + MATH for text LLMs
     (`geo3k` for vision, `dapo-math-17k` / `aime-2024` for scale-demo 235B /
     671B scripts).
   - `actor_rollout_ref.actor.use_dynamic_bsz=True`
   - `trainer.balance_batch=True`
   - `trainer.logger=["console","wandb"]`.

4. No deprecated Hydra knobs:

   - `ppo_megatron_trainer.yaml` → use `actor_rollout_ref.actor.model_engine=megatron`.
   - `actor_rollout_ref.rollout.mode=async` → removed; async rollout is no
     longer selected this way in example scripts.
   - `actor_rollout_ref.hybrid_engine=True` → removed; the trainer now
     enforces the supported hybrid-engine path internally.
   - `ppo_micro_batch_size` / `log_prob_micro_batch_size` → use the
     `_per_gpu` suffix.
   - `data.val_batch_size` → removed.
   - Top-level `reward_model.*` → use `reward_model.reward_model.*` /
     `reward.reward_model.*` as applicable.
   - `actor.ulysses_sequence_parallel_size` → use
     `actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size`.

## Directory layout

### Algorithm trainers

Each directory holds a canonical recipe for one training algorithm. Adding a
new algorithm? If it needs its own trainer entry point or reward code, put it
under `recipe/` instead.

| Dir                                | Algorithm                      | `algorithm.adv_estimator` / `policy_loss.loss_mode` |
|------------------------------------|--------------------------------|-----------------------------------------------------|
| `ppo_trainer/`                     | PPO (actor + critic)           | `adv_estimator=gae`                                 |
| `grpo_trainer/`                    | GRPO                           | `adv_estimator=grpo`                                |
| `rloo_trainer/`                    | RLOO                           | `adv_estimator=rloo`                                |
| `remax_trainer/`                   | ReMax                          | `adv_estimator=remax`                               |
| `reinforce_plus_plus_trainer/`     | REINFORCE++ / baseline         | `adv_estimator=reinforce_plus_plus[_baseline]`      |
| `cispo_trainer/`                   | CISPO                          | `loss_mode=cispo`                                   |
| `dppo_trainer/`                    | DPPO (TV / KL variants)        | `loss_mode=dppo_tv \| dppo_kl`                      |
| `gdpo_trainer/`                    | GDPO                           | `adv_estimator=gdpo`                                |
| `gmpo_trainer/`                    | GMPO                           | `loss_mode=geo_mean`                                |
| `gpg_trainer/`                     | GPG                            | `adv_estimator=gpg`, `loss_mode=gpg`                |
| `gspo_trainer/`                    | GSPO                           | `loss_mode=gspo`                                    |
| `sapo_trainer/`                    | SAPO                           | `loss_mode=sapo`                                    |
| `otb_trainer/`                     | OTB                            | `adv_estimator=optimal_token_baseline`              |
| `mtp_trainer/`                     | DAPO + MTP (MiMo-7B)           | `adv_estimator=grpo`, MTP flags                     |
| `on_policy_distillation_trainer/`  | on-policy distillation         | GRPO + distillation loss                            |
| `flowgrpo_trainer/`                | Flow-GRPO (diffusion)          | image-gen specific                                  |

### Feature / infra

| Dir                  | Purpose                                                                                  |
|----------------------|------------------------------------------------------------------------------------------|
| `tuning/`            | LoRA (`tuning/lora/`) and scaling demos (`tuning/scaling/`).                             |
| `profile/`           | NPU profiler / torch-memory profiler runs.                                               |
| `sft/`               | Supervised fine-tuning examples.                                                         |
| `generation/`        | Rollout-only inference launches.                                                         |
| `sglang_multiturn/`  | SGLang multi-turn rollout examples.                                                      |
| `vllm_omni/`         | vLLM omni backend examples.                                                              |
| `data_preprocess/`   | Scripts that produce the `$HOME/data/<dataset>/*.parquet` layout the run scripts expect. |
| `prefix_grouper/`    | Prefix-grouped rollout examples.                                                         |
| `rollout_correction/`| Rollout correction examples.                                                             |
| `router_replay/`     | Router replay examples.                                                                  |
| `tutorial/`          | Tutorials and cluster launchers (`ray/`, `slurm/`, `skypilot/`, `agent_loop_get_started/`). |

### Where are the algorithm research variants?

`recipe/` — e.g. `recipe/dapo`, `recipe/prime`, `recipe/retool`,
`recipe/r1`, `recipe/spin`, `recipe/gvpo`, `recipe/flowrl`, ...
They ship their own trainer entry points and reward code.
