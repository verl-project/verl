# Validation record — 2026-11-06

Last updated: 11/09/2026

Historical CPU-only record. See [2026-11-09 GPU validation](fsdp2_muon_gpu_validation.md)
for later hardware execution, the full-matrix communication correction, and fresh CPU results.

Status: implementation and CPU eager validation complete; GPU acceptance scripts
prepared. **Muon + FSDP2 has NOT passed complete GPU validation.**

## Checkout/environment

The provided `/Users/zengyue/Desktop/muon` directory was empty, not a Git checkout.
Cloned the official repository into that directory. Branch `main`, base/HEAD
`689fa1a2a1a35fbfee6cdd8cabdec0c15688ccaf`, initially clean. No original changes
were overwritten, and no commits, pushes, PRs or community messages were made.

macOS ARM64, isolated `.venv` managed with `uv`, Python 3.12.12, torch 2.9.1,
transformers 5.16.1, hydra-core 1.3.8, tensordict 0.10.0, pytest 9.2.0.
CUDA unavailable, device count 0. This is a focused local CPU environment, not
this checkout's complete Linux/CUDA lockfile environment.

Dion 0.1.0 was installed using the optional VCS requirement. Its installed
`direct_url.json` confirms commit `7692479288f928a3d61d1cc58137b4dbe4f71000`.
No Megatron dependency was installed for the FSDP2 implementation.

## Final executable suite

Run from the repository root:

```bash
TORCH_COMPILE_DISABLE=1 .venv/bin/python -m pytest \
  tests/workers/config/test_fsdp2_muon_on_cpu.py \
  tests/utils/test_fsdp2_muon_layout_on_cpu.py \
  tests/workers/config/test_optim_config_on_cpu.py \
  tests/utils/megatron/test_muon_optim_wiring_on_cpu.py \
  tests/utils/megatron/test_muon_optim_realcfg_on_cpu.py \
  tests/workers/test_fsdp_gradient_accumulation_sync_on_cpu.py \
  tests/special_distributed/test_muon_fsdp2_acceptance.py \
  -q -ra
```

**Exit 0: 69 passed, 0 failed, 5 skipped** (7 warnings: absent optional engines and
existing warmup-style deprecation). Of the passed cases, 42 are in the two new CPU
test files. The skipped cases are separate from passed:

- 2 existing Megatron real-config cases: no `megatron.core.optimizer`.
- 3 GPU wrapper cases: respectively 1, 2, 2 CUDA devices required.
- **0 new CPU tests skipped after installing real Dion.**

`test_dependency_is_optional` simulates a missing import; `test_stub_constructor_receives_options`
uses a stub optimizer to inspect construction. These are wiring tests, not numeric
validation. Separate real-Dion tests construct and step the actual class, compare
five-step unsharded reference updates for both Nesterov settings and all three LR
scaling choices, exercise zero/None gradients and scheduler/state restore.
The ordinary AdamW engine path is compared exactly against `torch.optim.AdamW`.

True two-process CPU/Gloo DTensor tests cover invalid layouts, actual mesh use,
collective rejection of inconsistent gradient presence/layout, and sharded
optimizer state serialization/restoration including AdamW step counter 257.
They are **not** CUDA FSDP2 training or distributed NS-update tests.

The CPU command explicitly disables upstream `torch.compile` to execute its real
PyTorch eager math on this machine. This test-only environment setting does not
modify production settings or replace the optimizer algorithm.

## Other executed checks and blockers

| Command/check | Exit | Result |
| --- | ---: | --- |
| `uv pip install --python .venv/bin/python --no-deps -r requirements-fsdp2-muon.txt` (after torch/numpy setup) | 0 | Installed actual pinned Dion; VCS provenance verified. |
| `.venv/bin/python -m pytest tests/workers/config/test_fsdp2_muon_on_cpu.py::test_real_zero_and_missing_gradients -q -ra` (compile enabled) | 1 | **0 passed, 1 failed, 0 skipped**: Inductor invokes local `clang++`, which cannot find C++ standard header `<algorithm>`. Compiled CPU execution remains blocked; not passed. |
| `TORCH_COMPILE_DISABLE=1 .venv/bin/python -m pytest tests/utils/megatron/test_muon_layerwise_bridge_ddp_on_cpu.py -q -ra` | 2 | **0 passed, 0 failed, 0 skipped, 1 collection error**: module imports unavailable `megatron.core` before tests run. No bridge test result claimed. |
| `.venv/bin/python tests/special_distributed/run_muon_fsdp2_acceptance.py --case alignment` | 1 | Device preflight rejects 0 GPUs; **GPU tests NOT RUN**, no success marker. |
| `PATH="$PWD/.venv/bin:$PATH" bash scripts/generate_trainer_config.sh` | 1 | All four files generated; script's final `git diff --exit-code` intentionally fails because generated changes are uncommitted. |
| Separate regeneration/comparison of all four files using `scripts/print_cfg.py --cfg job` with each backend override | 0 | All generated contents equal current Hydra composition; only FSDP reference YAML differs from HEAD. |
| Ruff check + format check on all eight added/modified Python files | 0 each | All checks passed; eight files formatted. |
| `python -m compileall -q` on those files | 0 | Python syntax valid; this is not an execution test. |
| `bash -n tests/special_distributed/run_muon_fsdp2_acceptance.sh examples/grpo_trainer/run_dense_muon_fsdp2.sh` | 0 | Both shell scripts parse. |
| `git diff --check` | 0 | No whitespace errors. |
| Capture GRPO launcher arguments using a temporary fake python3, then compose the actual Hydra config | 0 | 55 overrides accepted, including a model path with spaces; FSDP2/Muon/FP32/offloads/GRPO verified. No trainer executed. |

The first test-first run, before implementation and Dion installation, produced
16 failures and 2 dependency skips. Subsequent focused tests exposed input-embedding
identity and attention-output-name classification issues before the fixes.
Independent review also caught a negative test using two different model objects;
it now uses one model and verifies the specific rejection message.

Exact final Ruff commands:

```bash
.venv/bin/ruff check \
  verl/utils/fsdp2_muon.py verl/workers/config/optimizer.py \
  verl/workers/engine/fsdp/transformer_impl.py \
  tests/workers/config/test_fsdp2_muon_on_cpu.py \
  tests/utils/test_fsdp2_muon_layout_on_cpu.py \
  tests/special_distributed/*muon*.py
.venv/bin/ruff format --check \
  verl/utils/fsdp2_muon.py verl/workers/config/optimizer.py \
  verl/workers/engine/fsdp/transformer_impl.py \
  tests/workers/config/test_fsdp2_muon_on_cpu.py \
  tests/utils/test_fsdp2_muon_layout_on_cpu.py \
  tests/special_distributed/*muon*.py
```

Local raw logs and environment provenance are retained in `logs/fsdp2_muon/`
(ignored by Git). Main suite: `pytest.log`; blocked compiled CPU run:
`compiled_cpu_failure.log`; bridge collection: `megatron_bridge_collection.log`;
CUDA preflight: `gpu_preflight.log`; packages and VCS provenance: `environment.json`.

## NOT RUN

- CUDA single-rank actual FSDP2 forward/backward/update (FP32/BF16).
- Two-GPU optimizer and training reference alignment, including uneven/empty shards.
- Real CUDA checkpoint process-exit/resume through FSDPCheckpointManager.
- Any GRPO training, Qwen3.5-4B run, memory/speed/reward benchmark.

Reason: no CUDA GPU here; the user explicitly excluded GPU rental and training.
Commands and installation instructions are in [the implementation guide](fsdp2_muon.md).
Numerical tolerances are preregistered heuristics, not GPU-validated results.

## Remaining risks and working tree

GPU collectives, compiled CUDA kernels, mixed-precision multi-step tolerance and
process-restart checkpoint behavior require execution on the target A100 machine.
Classification is conservative and requires auditing custom model module roles.
Hybrid Qwen3.5 blocks are routed to AdamW, but its full GRPO pipeline is unverified.
No throughput or memory advantage is established; full-matrix communication/workspaces
and two guard reductions add cost.

HEAD remains the base commit. Four tracked files modified: optimizer config,
FSDP engine, source Hydra YAML, generated reference YAML. New helper, dependency
pin, two CPU test files, four GPU test/harness files, GRPO launcher and documentation
are untracked reviewable additions. Nothing staged or committed.
