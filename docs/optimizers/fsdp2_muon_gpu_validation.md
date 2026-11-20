# GPU validation — 2026-11-09

Last updated: 11/09/2026

**The small synthetic-model GPU acceptance passed on the user's two A100s after
correcting the matrix communication path.** This does not establish real GRPO,
Qwen3.5-4B compatibility, convergence, throughput or memory efficiency.
The subsequent [GRPO and submission-check record](fsdp2_muon_grpo_validation.md)
documents the later training experiment and rerun against the final source.

## Hardware and exact source

- User-provided single-node instance: 2 x NVIDIA A100-SXM4-80GB, 81920 MiB each.
- Ubuntu 24.04, Python 3.12.3, driver 595.84, PyTorch `2.13.0+cu130`, CUDA runtime 13.0,
  Triton 3.7.1, NCCL 2.29.7. Approximately 96 GiB host RAM.
- The visible GPU topology is **PHB**, with peer access reported OK; NVLink was
  not visible to this instance. These tests are correctness tests, not a bandwidth benchmark.
- Local branch/HEAD stayed `main` / `689fa1a2a1a35fbfee6cdd8cabdec0c15688ccaf`.
  The working-tree implementation was uploaded, including untracked source files.
  No merge, commit, push, PR or community message was performed.
- Microsoft Dion Muon commit `7692479288f928a3d61d1cc58137b4dbe4f71000`, package
  0.1.0, MIT. VCS commit verified from installed `direct_url.json`.
- Repository on GPU: `/root/verl-muon-20261109`; isolated `.venv` managed with uv.
  This is a focused test environment, not the full vLLM/Megatron training stack.

## Failure found and correction

The initial direct Dion all-to-all integration passed single-rank smoke and
checkpoint restart, but **failed** the preregistered two-rank numerical budgets.
At optimizer step 1, momentum and AdamW state matched exactly, while the four
narrow matrices differed. Maximum absolute weight error was `2.4859607e-4`;
per-matrix relative update errors ranged from approximately 3.77% to 6.84%.
The FP32 training comparison failed at step 1; BF16 training passed step 1 but
failed at step 2. Later steps of these initial failing runs were **NOT RUN**.

A standalone diagnostic reproduced these differences using only upstream NS:
plain 2D input versus a leading batch axis, and then batch axis plus zero rows.
Disabling reduced-precision BF16 reductions did not remove the batch-axis error.
The mismatch was therefore observable without FSDP gradient reduction, LR scaling
or checkpoint code. Logs preserve the original failures; no GPU tolerance was widened.

`verl/utils/fsdp2_muon_optimizer.py` now subclasses Dion Muon and replaces only
its matrix communication path. It reuses upstream local momentum and post-update
kernels, gathers each BF16 NS input through the actual parameter DTensor mesh,
and uses Dion's unsharded shape-batching path. Every rank computes the complete
un-padded matrix batch and takes its original local rows for the update. AdamW,
optimizer counters, scheduler integration and serialized state remain inherited.
Checkpoint metadata includes `matrix_evaluation=unsharded_shape_batch_v1` and
rejects checkpoints from the earlier evaluation path.

This is a correctness-first tradeoff: NS compute is replicated, and temporary
memory includes a complete batch of all active matrices with the same shape plus
NS workspaces. It can consume substantial memory for repeated large projections.
It does not retain full parameter/momentum copies or introduce hidden optimizer state.

## Final executed tests

Commands run in `/root/verl-muon-20261109` with `.venv/bin` on `PATH`:

```bash
bash tests/special_distributed/run_muon_fsdp2_acceptance.sh \
  --case all --precision both --dynamo-recompile-limit 64 \
  --output /root/verl-muon-results/gather-all-01

python -m pytest tests/special_distributed/test_muon_fsdp2_acceptance.py -q -ra
```

| Final check | Exit | Result |
| --- | ---: | --- |
| Standalone complete GPU suite | 0 | **10 passed worker phases, 0 failed, 0 skipped**; final success marker and `result.json` present. |
| GPU pytest entry point | 0 | **3 passed, 0 failed, 0 skipped** (smoke/alignment/checkpoint, each covers both precisions). |
| Full local CPU regression below | 0 | **70 passed, 0 failed, 5 skipped**. |
| Ruff check / format check, nine added or modified Python files | 0 / 0 | All passed. |
| Python compileall, Bash syntax checks, `git diff --check` | 0 | No syntax or whitespace errors. |

The 10 standalone phases are smoke, alignment, continuous10, save5, resume5,
each in FP32 and BF16. The pytest run repeats those checks; it is not an extra
13 independent behaviors. It explicitly gives paired reference tests a Dynamo
cache budget of 64 and runs checkpoint with PyTorch's default budget of 8.
CUDA compilation remained enabled. No model/data download was needed.

Maximum observed errors over all five double-GPU alignment steps:

| Comparison | Weight max absolute | State max absolute | Update max relative L2 |
| --- | ---: | ---: | ---: |
| Identical synthetic gradients, FP32 forward setting | 0 | 0 | 0 |
| Identical synthetic gradients, BF16 forward setting | 0 | 0 | 0 |
| Actual FSDP2 FP32 forward/backward/clipping | 0 | 0 | 0 |
| Actual FSDP2 BF16 forward/backward/clipping | `1.6573817e-5` | `1.8626451e-9` | `0.009568746` |

Both single-rank smoke comparisons had zero error, nonzero updates in both groups,
and finite weights/state. Group audit: 8 Muon tensors / 8448 elements; 15 AdamW
tensors / 4548 elements. The fixture includes repeated rectangular projections,
uneven shards and an empty local shard. Full weights and every momentum/AdamW
state are compared after each update, including steps 1 and 5.

Checkpoint tests use real `FSDPCheckpointManager` model/optimizer/extra-state files.
Save and resume run in separate processes. In both precisions, step-5 restoration
and continuous10 versus save5/exit/rebuild/load/another5 match **exactly**, including
optimizer groups, scheduler and stochastic training trajectory.

Fresh CPU command, run locally with loopback communication permitted:

```bash
TORCH_COMPILE_DISABLE=1 .venv/bin/python -m pytest \
  tests/workers/config/test_fsdp2_muon_on_cpu.py \
  tests/utils/test_fsdp2_muon_layout_on_cpu.py \
  tests/workers/config/test_optim_config_on_cpu.py \
  tests/utils/megatron/test_muon_optim_wiring_on_cpu.py \
  tests/utils/megatron/test_muon_optim_realcfg_on_cpu.py \
  tests/workers/test_fsdp_gradient_accumulation_sync_on_cpu.py \
  tests/special_distributed/test_muon_fsdp2_acceptance.py -q -ra
```

CPU skips: 2 missing Megatron cases and 3 CUDA wrappers on the local Mac. They
are not counted as passes. The new actual two-process CPU/Gloo update test covers
five updates, repeated shapes, uneven/empty shards and uniform missing gradients.
Muon weights/momentum match exactly; CPU AdamW slice kernels use the existing
FP32 optimizer-state budget (`atol=2e-7`, `rtol=2e-6`) for vector rounding.
The original test failed before the adapter because Gloo does not support all-to-all.

## Reproduce on a fresh Ubuntu 24.04 GPU instance

Install uv first if absent. The actual test used system Python 3.12 and therefore
needed its development headers for Triton. From the source checkout:

```bash
apt-get update
apt-get install -y git build-essential python3.12-dev
uv venv --python /usr/bin/python3 .venv
UV_HTTP_TIMEOUT=600 UV_CONCURRENT_DOWNLOADS=8 uv --no-config pip install \
  --python .venv/bin/python --index-url https://pypi.org/simple \
  torch==2.13.0 transformers==5.16.1 tensordict==0.10.0 \
  hydra-core==1.3.8 pytest==9.2.0 ray==2.60.0 accelerate==1.15.0 \
  peft==0.21.2 pandas==3.1.1 pillow==13.0.0 codetiming==1.4.0 \
  torchdata==0.11.0 einops==0.8.2 datasets==5.1.0
uv --no-config pip install --python .venv/bin/python --no-deps \
  -r requirements-fsdp2-muon.txt
source .venv/bin/activate
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.device_count())'
```

Then run the two final GPU commands above, choosing a new empty output directory.
The PyPI torch wheel resolved to `2.13.0+cu130`; the driver was left unchanged.
Full transitive versions are archived in `remote_packages.txt` below. This minimal
environment suffices for synthetic acceptance only; GRPO requires the repository's
full rollout/training environment and explicit model/data paths.

## Earlier failures retained separately

- Initial local-directory Dion install generated an untracked `build/` directory:
  provenance preflight exited 1, **GPU computation NOT RUN**. Reinstalled from the
  exact VCS requirement; no provenance check was bypassed.
- Missing `Python.h`: first CUDA compile run exited 1. Installed matching
  `python3.12-dev`, preserving compilation.
- Default Dynamo limit 8: the initial paired sharded/plain-reference test exited 1
  with `FailOnRecompileLimitHit`. An explicit harness option records a budget of 64;
  omitting it preserves defaults, and no production global setting is modified.
- Initial native all-to-all numerical failures: `alignment-01`,
  `optimizer-{fp32,bf16}-01`, `training-{fp32,bf16}-01`, exit 1 each. These are not passes.
- A sandboxed local CPU rerun produced 67 passed / 2 failed / 5 skipped because
  Gloo could not bind/resolve loopback. The final authorized loopback run above
  completed with 70 passed / 5 skipped after adding the new test.

## Evidence and limits

Local primary artifacts: `logs/fsdp2_muon_gpu_20261109/` (ignored by Git), including
source SHA256 manifests, `remote_packages.txt`, `cpu_regression_final.log`, original
failure logs, NS diagnostic source/output, and `verl-muon-results/gather-all-01/`.
That directory contains `result.json`, per-phase logs, full snapshots, actual
checkpoint files and `metrics_summary.json`. No SSH credentials are stored there.

**NOT RUN:** real GRPO training, Qwen3.5-4B, rollout weight synchronization under
load, throughput/peak-memory/reward benchmarks, other GPU architectures/world sizes,
nondefault NS/learning-rate options on CUDA, and full-model parameter classification.
The unchanged default AdamW path is covered by CPU regression, not an end-to-end
GPU AdamW training run. Unsupported FSDP1/HSDP/TP/MoE/LoRA/NPU/offload combinations
remain rejected. The gather adapter depends on internal APIs of the exact pinned
Dion commit. GPU tests validate the synthetic fixture and documented scope, not
universal production readiness.

Independent read-only code review of the gather adapter, builder and acceptance
tests reported no actionable findings; the reviewer did not rerun GPU tests.
Larger architectures with many matrix shapes may also need an explicitly chosen
Dynamo cache budget; no production global compiler setting is changed here.
Final working tree: four tracked files modified and sixteen untracked additions,
all unstaged. The source archive and final SHA256 manifest retain the exact code.
