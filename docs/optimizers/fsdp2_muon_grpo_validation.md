# Real GRPO and submission validation — 2026-11-09

Last updated: 11/10/2026

The native FSDP2 implementation completed **Muon 30 updates, AdamW 30 updates,
and a fresh Muon process restoring step 15 and continuing through step 30**.
These are real `verl.trainer.main_ppo` runs using TaskRunnerV1/PPOTrainerSync,
GSM8K rule rewards, vLLM rollout, actor backward/clipping/update and weight sync.
No RL algorithm code was changed. This is bounded integration evidence, not a
claim of convergence, optimizer superiority or support for every FSDP2 model.

## Reproducible inputs and environment

- GPU experiment base commit: `689fa1a2a1a35fbfee6cdd8cabdec0c15688ccaf`;
  the implementation was uncommitted on `codex/fsdp2-muon-grpo-validation`
  when these runs were collected. See the submission integration record below.
- Hardware: single node, two A100-SXM4-80GB, approximately 96 GiB host RAM,
  Ubuntu 24.04, driver 595.84; the instance exposes PHB topology, not NVLink.
- Python 3.12.3; torch `2.13.0+cu130`; vLLM 0.32.0; Transformers 5.16.1;
  Ray 2.55.1; Triton 3.7.1; Hydra 1.3.3; TensorDict 0.10.0; TransferQueue 0.1.11.
- Dion 0.1.0, MIT, pinned commit
  `7692479288f928a3d61d1cc58137b4dbe4f71000`. No Megatron installation was required.
- Model: `Qwen/Qwen2.5-0.5B-Instruct`, revision
  `7ae557604adf67be50417f59c2c2f167def9a775`.
- Data: `openai/gsm8k`, revision `740312add88f781978c0658806c59bc2815b9866`.
  All 7473 training examples; validation uses 64 test examples selected by
  `shuffle(seed=42).select(range(64))`, not the complete 1319-example test set.
  Preprocessing uses the repository's `extract_solution` and the instruction
  `Let's think step by step and output the final answer after "####".`

The existing uv environment was expanded with the checkout's frozen FSDP/vLLM
dependencies. Downloading the locked FlashAttention wheel returned HTTP 504,
including a retry through the local proxy. Both arms therefore explicitly use
actor SDPA and `use_remove_padding=False`; the environment was installed with:

```bash
uv sync --frozen --extra fsdp --extra vllm --inexact \
  --no-install-project --no-install-package flash-attn
uv pip install --no-deps -r requirements-fsdp2-muon.txt
```

This records the tested SDPA environment; it does not validate FlashAttention.
CUDA compilation of the optimizer remained enabled. Model/data revision manifests,
package freezes, effective Hydra configs and SHA256 hashes of the executed source
are archived with the results.

## Matched configuration and observations

Both arms use seed 42, eight prompts per global batch, four responses per prompt,
one PPO epoch, actor microbatch 1, prompt/response limits 512/512, numerical LR
`1e-6`, constant schedule with total horizon 30, weight decay `0.01`, FSDP2 size 2,
rollout TP 1, no KL/reference, and no CPU offload. Stored parameters are FP32 with
the engine's BF16 computation policy. Dion uses its documented BF16 NS math.
Checkpoints and 64-example validation run at updates 15 and 30.

The effective configs differ only in optimizer name, experiment name and output
directories. For every step, the global multiset of 32 prompt/ground-truth pairs
matches between arms. Length balancing can change their order. This same-numeric-LR
comparison does **not** calibrate Muon and AdamW update magnitudes.

| Observed metric | Muon, steps 1–30 | AdamW, steps 1–30 | Muon resume, steps 16–30 |
| --- | ---: | ---: | ---: |
| Completed updates / exit code | 30 / 0 | 30 / 0 | 15 / 0 |
| Steps with nonzero actor gradient norm | 11 | 19 | 2 |
| Mean training reward | 0.015625 | 0.043750 | 0.004167 |
| Last 10 updates' mean training reward | 0.009375 | 0.084375 | 0.006250 |
| Mean actor-update time, seconds | 7.4809 | 6.9195 | 7.7614 |
| Mean logged step time, seconds | 18.9313 | 18.1296 | 19.9619 |
| Maximum actor-reported allocated memory, GiB | 6.0380 | 5.2951 | 6.0402 |

These are single-seed observations, including warmup/compilation effects, with
short outputs and a small validation subset. Actor memory is the logged allocator
metric, not total GPU/rollout memory. This run **does not show a Muon reward,
speed or memory advantage**. No confidence interval or convergence claim is made.
Zero-gradient steps occur when GRPO groups have equal rewards; an exit code alone
would not establish a policy-gradient update. All logged numerical metrics were
checked finite, and every recorded update has a positive weight-sync duration.

The actual model grouping is 168 Muon tensors / 357826560 elements and 122 AdamW
tensors / 136206208 elements. Every Muon and fallback parameter shard changed
from initialization on both ranks by checkpoints 15 and 30. All saved model and
optimizer tensors are finite. The 291 model state-dict entries include the tied
output-head alias; they are not 291 independently optimized parameters.

### Post-run reward-format audit

The logged `acc/mean@1` uses the repository's strict GSM8K scorer: it searches
the last 300 output characters for `#### <number>` and compares the extracted
number with the reference. It measures answer correctness **and format adherence**,
not format-independent mathematical accuracy. Recomputing this scorer from every
saved training/validation output reproduces the recorded rewards exactly.

At update 30, Muon has **0/64 parseable strict answers**, while AdamW has 15/64,
of which 10 match the reference. Thus Muon's zero validation score includes a
complete failure to meet the scorer's answer format. Some outputs contain the
correct number in a boxed or prose answer, but receive zero strict reward.
As a separate diagnostic, the scorer's `flexible` last-number extraction matches
the reference on 33/64 Muon and 28/64 AdamW outputs. This is not a replacement
acceptance metric or proof of mathematical reasoning correctness; the original
strict scores remain 0/64 and 10/64.

Across 960 training responses per arm, Muon has 39 parseable strict answers and
15 rewarded responses; AdamW has 115 and 42 respectively. This reproduces mean
rewards `15/960 = 0.015625` and `42/960 = 0.04375` and exposes sparse reward signals.
The runs lack an initial-model validation baseline, optimizer-specific update-scale
calibration and repeated seeds. Consequently they establish integration execution,
but do not establish training-quality parity or explain causally why the optimizers
learn the answer format differently. The next quality experiment should preregister
the scorer/settings and include the initial checkpoint on the same validation set.

Audit script/result: `logs/fsdp2_muon_grpo_20261109/audit_reward_format.py` and
`reward_format_audit.json`; original archived outputs are unchanged.

## Real checkpoint restoration

The continuous Muon process saved step 15 and finished step 30 before exiting.
A new Ray/trainer process reconstructed the model and optimizer, loaded that
step-15 checkpoint using the real `FSDPCheckpointManager`, and ran steps 16–30.
The opt-in [restore probe](../../tests/special_e2e/fsdp2_muon/restore_probe.py)
asserted **exact** post-load equality on each rank for 291 model tensor entries,
534 optimizer tensors and the complete scheduler state (`last_epoch=15`).
The probe changes no training state. Both ranks produced `exact_restore=true`.
The restored global batches at steps 16–30 match uninterrupted training, and
the final scheduler epoch is 30 with finite model/optimizer state.

The resumed stochastic rollout trajectory is **not bitwise identical** to the
continuous run. At step 30, the maximum weight difference across both ranks is
`1.2265518e-5`; the maximum optimizer tensor-state difference is `0.07471506`.
These are reported differences, not errors accepted using an enlarged tolerance.
This experiment proves exact checkpoint loading and continued training, not
deterministic vLLM replay. The separate synthetic checkpoint test covers exact
continuous10 versus save5/exit/rebuild/load/another5 with controlled inputs.

## Failure found by the real training run

The first Muon attempt exited 1 after the initial actor update: Dion's scheduler
returned a CUDA LR tensor, and the CPU-only Ray driver failed to deserialize it.
`FSDPEngine.lr_scheduler_step()` now converts only the **reported metric** to a
host number. Optimizer LR tensors, scheduler state, update frequency, clipping
and formulas remain unchanged. A regression first failed for real Dion while
AdamW passed; both pass after the fix. The synthetic GPU checkpoint loop also
calls this real engine method and asserts a finite host float.

The failed first training log is retained separately from the successful rerun.
A later read-only inspection invocation exited 2 because torchrun interpreted
`--run` as an abbreviated launcher option; inserting `--` before the script path
fixed the invocation. It did not require rerunning or modifying training.

## Repository checks and CI

New Python files now contain the complete Apache-2.0 header with truthful 2026
individual attribution. The license checker gained the 2026 individual prefix;
the prior 2025 prefix remains valid and missing/unknown headers remain rejected.
Other full-hook findings (device API usage, documentation timestamps, example
backend suffix, and three Ruff union checks) were corrected.

The GPU workflow explicitly installs pinned optional Dion with `--no-deps`, runs
the standalone all-cases/both-precisions suite on two visible GPUs, preserves
failure exits through `pipefail`, sets timeouts and uploads diagnostics even on
failure. The existing generic `tests/special*` exclusion no longer hides this
suite because the new step invokes it directly. GitHub Actions execution is
**NOT RUN**: no push or PR update was authorized or performed.

| Final check | Exit | Result |
| --- | ---: | --- |
| Complete pre-commit | 0 | 14/14 hooks passed; no hook rewrites. |
| Related CPU regression | 0 | 79 passed, 0 failed, 5 skipped. |
| Standalone GPU suite, final source/environment | 0 | 10/10 worker phases passed, 0 failed, 0 skipped. |
| Muon30 / AdamW30 / Muon resume15–30 | 0 each | All expected updates recorded; finite metrics and nonzero gradients. |
| GRPO batch/restore assertions and checkpoint inspections | 0 | Both ranks pass; final scheduler epoch 30. |

The five CPU skips are two pre-existing real-Megatron cases (dependency absent)
and three CUDA wrappers on the local Mac. No new real-Dion CPU case skipped.
CPU eager math uses `TORCH_COMPILE_DISABLE=1` because this Mac lacks usable C++
standard-library headers for Inductor. GPU compilation was enabled. Constructor
stub/missing-import tests remain distinct from real-Dion numerical tests.

```bash
TORCH_COMPILE_DISABLE=1 .venv/bin/python -m pytest \
  tests/workers/config/test_fsdp2_muon_on_cpu.py \
  tests/utils/test_fsdp2_muon_layout_on_cpu.py \
  tests/workers/config/test_optim_config_on_cpu.py \
  tests/utils/megatron/test_muon_optim_wiring_on_cpu.py \
  tests/utils/megatron/test_muon_optim_realcfg_on_cpu.py \
  tests/workers/test_fsdp_gradient_accumulation_sync_on_cpu.py \
  tests/special_distributed/test_muon_fsdp2_acceptance.py \
  tests/utils/test_muon_grpo_restore_probe_on_cpu.py -q -ra

# Prepared GPU environment, final source; a new output directory is required.
python tests/special_distributed/run_muon_fsdp2_acceptance.py \
  --case all --precision both --dynamo-recompile-limit 64 --timeout 900 \
  --output /root/verl-muon-grpo-results/final-gpu-acceptance

pre-commit run --all-files --show-diff-on-failure
```

The pre-commit command ran in a disposable full source snapshot containing all
1348 tracked and proposed new source entries, with SHA256 metadata. This includes
untracked files without staging or committing anything in the user's checkout.
The original index remains untouched. Earlier failing hooks and their fixes are
retained in the local evidence directory.

The final GPU suite repeats smoke, five-step optimizer/training alignment,
continuous10, save5 and resume5 in both precisions. Optimizer-level alignment and
FP32 training match exactly. BF16 training's maximum weight/state errors are
`1.6573817e-5` / `1.8626451e-9`, with maximum relative update L2 error `0.009568745`;
the declared tolerances were not changed. Both synthetic checkpoint paths match
exactly after restart, including optimizer groups/state and scheduler. The final
driver manifest contains ten zero exits and its assertion-completion marker.

## Submission integration with upstream main

The submission branch incorporates upstream main
`a911bca25d6adb26f5bef2dc57e7ba62e99b7ade`, seven commits after the GPU experiment
base. None of those commits overlaps the 26 proposed implementation, test and
documentation paths. At submission commit `14fd349501b45e91919fe09dd34bb3b2b7222b44`,
the executed Muon implementation and GPU test source remained byte-identical.
One upstream change masks empty responses in rollout metrics;
its CPU regression was included in the post-integration check:

```bash
TORCH_COMPILE_DISABLE=1 .venv/bin/python -m pytest \
  tests/workers/config/test_fsdp2_muon_on_cpu.py \
  tests/utils/test_fsdp2_muon_layout_on_cpu.py \
  tests/workers/config/test_optim_config_on_cpu.py \
  tests/utils/megatron/test_muon_optim_wiring_on_cpu.py \
  tests/utils/megatron/test_muon_optim_realcfg_on_cpu.py \
  tests/workers/test_fsdp_gradient_accumulation_sync_on_cpu.py \
  tests/special_distributed/test_muon_fsdp2_acceptance.py \
  tests/utils/test_muon_grpo_restore_probe_on_cpu.py \
  tests/trainer/ppo/test_rollout_corr.py -q -ra
```

Result: **99 passed, 0 failed, 5 skipped; exit 0**. The skip reasons are unchanged.
Complete pre-commit also passed all 14 hooks on the integrated source snapshot,
with no hook changes. GPU and real GRPO evidence above belongs to the explicitly
recorded experiment base, not a rerun after main integration. Submission evidence
and source hashes are retained locally in `logs/fsdp2_muon_submission_20261109/`.

### Review follow-up: TRL value-head grouping

The output-head exclusions now include `v_head`, keeping TRL's
`v_head.summary` on AdamW even when shared with an ordinary Linear alias.
Two new CPU cases failed before this correction and passed afterward. Repeating
the CPU command above produced **101 passed, 0 failed, 5 skipped; exit 0**.
The initial sandbox run could not initialize loopback communication for three
Gloo tests; the same command passed with local network access enabled.
Function descriptions were also added to the optimizer and acceptance helpers.
The numerical update and communication algorithms are unchanged. Review evidence
is retained locally in `logs/fsdp2_muon_review_20261109/`.

## Reproduce the real training experiment

Activate the prepared GPU environment and provide the pinned local model and
preprocessed data paths. The launcher requires new result directories and has a
2400-second per-run timeout. From the repository root:

```bash
export MODEL_PATH=/models/Qwen2.5-0.5B-Instruct
export TRAIN_FILE=/data/gsm8k/train.parquet
export VAL_FILE=/data/gsm8k/test.parquet
OPTIMIZER=Muon RUN_ROOT=/results/muon30 \
  bash tests/special_e2e/fsdp2_muon/run_gsm8k.sh
OPTIMIZER=AdamW RUN_ROOT=/results/adamw30 \
  bash tests/special_e2e/fsdp2_muon/run_gsm8k.sh
VERL_MUON_VERIFY_RESTORE=1 OPTIMIZER=Muon \
  RESUME_FROM=/results/muon30/checkpoints/global_step_15 \
  RUN_ROOT=/results/muon_resume15 \
  bash tests/special_e2e/fsdp2_muon/run_gsm8k.sh \
  actor_rollout_ref.model.external_lib=tests.special_e2e.fsdp2_muon.restore_probe
```

## Validation scope and limitations

Validation covers FP32/BF16 synthetic FSDP2 tests and 30-update GSM8K GRPO runs
with Qwen2.5-0.5B-Instruct on a single node with two A100-SXM4-80GB GPUs,
including checkpoint restoration in a fresh process.

The implementation supports dense full-parameter native FSDP2 on a single 1D
shard mesh, FP32 parameter/state storage, FP32/BF16 computation and same-world-size
checkpoint restoration. FSDP1, HSDP, TP training, MoE, LoRA, NPU and CPU offload
remain outside the implemented scope. Full-matrix shape-batch gathering and
replicated orthogonalization can add substantial communication and temporary
memory cost.
