# Native FSDP2 Muon (initial implementation)

Last updated: 11/09/2026

Implementation and CPU validation are available. The initial 2026-11-06 CPU-only
record is retained in [validation history](fsdp2_muon_validation.md); subsequent
GPU execution is recorded separately in [GPU validation](fsdp2_muon_gpu_validation.md)
and the [real GRPO/submission checks](fsdp2_muon_grpo_validation.md).
No large model or dataset download, GRPO run, or performance claim is implied.

## Configuration and implementation

```bash
actor_rollout_ref.actor.strategy=fsdp2 \
actor_rollout_ref.actor.optim.optimizer=Muon
```

`AdamW` remains the default, including its original dynamic import, constructor
arguments and update behavior. Selecting `Muon` (case insensitive) lazily imports
Microsoft **Dion's Muon class**, not the Dion/Dion2 algorithm and not Megatron.
DTensor parameters use a small subclass that replaces matrix communication and
reuses the pinned momentum, NS, LR scaling, decay, AdamW and serialization code.
Install [requirements-fsdp2-muon.txt](../../requirements-fsdp2-muon.txt) only when
needed. It pins commit `7692479288f928a3d61d1cc58137b4dbe4f71000` (package version
`0.1.0`, MIT license, Python >=3.9 and torch >=2.7.1 + numpy). Package version alone
is insufficient: upstream retains `0.1.0` across commits. Use the exact VCS pin;
the GPU driver checks installed commit provenance before launching.

[Muon source](https://github.com/microsoft/dion/blob/7692479288f928a3d61d1cc58137b4dbe4f71000/dion/muon.py),
[distributed implementation](https://github.com/microsoft/dion/blob/7692479288f928a3d61d1cc58137b4dbe4f71000/dion/megabatch_base.py),
[license](https://github.com/microsoft/dion/blob/7692479288f928a3d61d1cc58137b4dbe4f71000/LICENSE).
No optional Triton kernel or Gram-NS package is selected. Upstream's PyTorch
functions use `torch.compile`; CUDA PyTorch must have a working compiler backend.

The selected update definition is:

- Stored parameter, gradient, Muon momentum and AdamW moments: FP32. Forward
  parameters may be FP32 or BF16 under the existing FSDP2 mixed-precision policy.
- Momentum `M <- mu*M + G`; default `mu=.95`. Nesterov is false by default; if
  enabled, NS receives `G + mu*M` rather than `M`.
- Upstream converts the NS input to **BF16 before communication**. Thus the FP32
  baseline means FP32 stored weights/state and forward computation, **not an
  all-FP32 NS implementation**. Both distributed and unsharded references use
  exactly this definition. An all-FP32 orthogonalizer is outside this version.
- Default `muon_use_polar_express=false` selects five NS iterations with coefficient
  triples `(4.0848,-6.8946,2.9270)`, `(3.9505,-6.3029,2.6377)`,
  `(3.7418,-5.5913,2.3037)`, `(2.8769,-3.1427,1.2046)`,
  `(2.8366,-3.0525,1.2012)`. Input is transposed when tall and normalized by
  Frobenius norm + epsilon; `X <- aX + (b XX^T + c (XX^T)^2)X`.
  `true` selects upstream Polar Express; it is recorded in checkpoint metadata,
  but GPU acceptance currently targets the default five-step path.
- `muon_adjust_lr=spectral_norm` scales the update by `sqrt(rows/cols)` using
  full matrix dimensions. `rms_norm` uses `0.2*sqrt(max(rows,cols))`; `null` uses
  the base LR. These are upstream's definitions, not Megatron's modes.
- Decoupled decay is `P *= (1 - base_lr*weight_decay)` before the update. Cautious
  decay is disabled. No QKV/head splitting or tensor flattening is requested.
- AdamW groups use the same base LR, weight decay, config `betas` and
  `muon_epsilon` (default `1e-8`). AdamW moments and per-parameter bias-correction
  counter are stored in standard `optimizer.state`. Biases/norms retain the
  existing all-parameter weight-decay convention; no new no-decay policy is added.

The initial direct Dion all-to-all path failed the fixed GPU numerical budgets:
its batch-axis/zero-padding changes produced different BF16 NS rounding for narrow
matrices. The integration therefore uses a correctness-first **all-gather** path.
After Dion updates local FP32 momentum and forms the BF16 NS input, the adapter
reconstructs each input through its real `DTensor.full_tensor()` layout. Every rank
runs Dion's unsharded same-shape batching path, with the reference's exact matrix
dimensions and batch count, then extracts its original `Shard(0)` rows for Dion's
local decay/update kernel. No padding rows are orthogonalized, and no local shard
is independently orthogonalized. Mesh and local rank come from each parameter;
no default global process group is substituted.

This replicates NS computation on every rank and adds a BF16 all-gather per active
Muon matrix. Peak temporary storage includes the entire same-shape matrix batch
and its NS workspaces; it can be substantial for repeated large projections.
Momentum stays FP32 and sharded in standard `optimizer.state`; full parameter or
momentum copies are not retained. AdamW retains Dion's native local shard updates.
No measured speed or memory advantage is claimed. Two small all-reduces per step
check gradient-presence/layout consistency before matrix communication.

## Parameter classification and boundaries

Only trainable ordinary `nn.Linear.weight` matrices use Muon. HF input/output
embedding accessor identities, `nn.Embedding`, output heads (including TRL
`v_head.summary`), norms, biases,
unknown tensors (even 2D), and hybrid `linear_attn`/GatedDeltaNet blocks use AdamW.
Special ownership wins when a parameter is shared with a projection. Frozen
parameters are omitted. Every remaining parameter belongs to exactly one group,
sorted lexicographically by its first alias. Rank zero logs tensor/global-element
counts; `optimizer.param_groups[*]['param_names']` records the exact audit list.

Native FSDP2 with FP32 stored parameters and one 1D `Shard(0)` mesh is the scope.
HSDP/2D mesh, TP placements, mixed meshes/plain-and-sharded parameters, FSDP1,
fsdp_turbo, LoRA/PEFT, detected MoE, NPU/MPS and CPU offload are rejected.
Unknown optimizer overrides are rejected; use the named `muon_*` fields.
CPU ordinary tensors remain usable for reference/unit testing. Custom architectures
with nonstandard module names/metadata still require human review of the group list;
this is not universal automatic recognition of every model's special weights.

Uniform `grad=None` skips that parameter, decay and moments. A zero gradient
continues historical momentum and decay; an initially zero momentum stays finite.
Rank-asymmetric missing gradients or inconsistent gradient layouts are rejected
collectively before updating. Sparse gradients and optimizer closures are rejected.
Clipping remains exclusively in the existing engine; it is not repeated here.

Checkpoint compatibility is same world size, group membership/order, shapes,
sharded layout, matrix evaluation path and NS definition. Metadata stays in standard `param_groups`, with
pre-load validation. Dion restores its device LR cache from serialized group LRs;
there is no second optimizer or hidden momentum store. Elastic/world-size-changing
resume and CPU offload are not supported. Existing engine checkpoint/save/load,
GRPO loss/backward, scheduler and rollout weight-sync implementations are reused.

## Files and actual repository paths

- `verl/workers/config/optimizer.py`: fields and optional builder branch.
- `verl/trainer/config/optim/fsdp.yaml`: Hydra fields; generated reference YAML is
  refreshed by `scripts/generate_trainer_config.sh` (actor and critic sections).
- `verl/workers/engine/fsdp/transformer_impl.py`: early scope validation and module
  metadata passed through `_build_optimizer`; `_build_lr_scheduler` and
  `optimizer_step` retain existing scheduling/clipping/update behavior.
- `verl/utils/fsdp2_muon.py`: classification, validation, construction and standard optimizer hooks.
- `verl/utils/fsdp2_muon_optimizer.py`: optional Dion subclass; full-matrix input
  gathering and local update extraction, reusing the upstream numeric kernels.
- Existing `verl/utils/fsdp_utils.py`: FSDP2 clipping, model/optimizer load/offload.
- Existing `verl/utils/checkpoint/fsdp_checkpoint_manager.py`: per-rank model,
  optimizer and extra/scheduler/RNG files. No change required.
- Tests under `tests/workers/config`, `tests/utils`, `tests/special_distributed`.

[PR #7120 evidence and differences](fsdp2_muon_reference_tests.md) separates source
assertions from author-reported experiments. This implementation adds semantic
partition checks, optional-dependency isolation, real CPU updates/state restore,
DTensor collective guards, and prepared real FSDP2 numerical/restart acceptance.

## Linux / 2 x A100 setup and acceptance

From the repository root, use its native FSDP environment. The repository's current
full runtime setup is `uv sync --extra fsdp --extra vllm` (CUDA/platform requirements
are those of this checkout); vLLM is only needed for subsequent GRPO. Then:

```bash
source .venv/bin/activate
uv pip install -r requirements-fsdp2-muon.txt
python -c 'import torch; from dion import Muon; print(torch.__version__, torch.cuda.device_count())'

# Stage 1: one CUDA rank, real builder, forward/backward/clip/update, both precisions.
bash tests/special_distributed/run_muon_fsdp2_acceptance.sh --case smoke --precision both --dynamo-recompile-limit 64

# Stage 2: two ranks, actual fully_shard/DTensor, same full gradients and global batch.
bash tests/special_distributed/run_muon_fsdp2_acceptance.sh --case alignment --precision both --dynamo-recompile-limit 64

# Stage 3: continuous10 vs save5/process-exit/rebuild/load5 via FSDPCheckpointManager.
bash tests/special_distributed/run_muon_fsdp2_acceptance.sh --case checkpoint --precision both
```

Activate the environment and invoke Python directly: an automatic `uv sync` after
installing the separate optional requirements may remove Dion as an unlocked package.
Do not set `TORCH_COMPILE_DISABLE=1` for the standard GPU acceptance. Each process
has a 900-second timeout by default; `--timeout` overrides it. `--output` must be a
new/empty directory. Logs and an all-passed `result.json` are kept there. The driver
checks exit codes and worker success markers, kills timed-out process groups and
returns failure for insufficient devices. Pytest wrappers instead display SKIPPED.

The paired plain/sharded reference processes can exceed Dynamo's default eight
compiled variants. `--dynamo-recompile-limit 64` is an explicit test-only cache
budget, recorded in logs/result metadata; omitting it preserves PyTorch settings.
It does not disable compilation or change numeric tolerances. The checkpoint
command above intentionally keeps the default. `--case optimizer` and `--case
training` run the two alignment layers independently when diagnosing a failure.
With a system Python interpreter, install its matching development headers
(e.g. `python3.12-dev`) and a C/C++ compiler for Triton. See the GPU record for the
exact minimal environment used; vLLM and Megatron are unnecessary for these tests.

The GPU fixture uses synthetic tokens and a small model with embedding, norm, head,
several projection matrices, uneven shards and an empty local shard. It downloads
nothing. Both optimizer and training alignment compare full weights and every
Muon/AdamW state after each of five updates, with update-relative-L2 guards so
unchanged large weights cannot conceal incorrect updates. Training uses matched
microbatch shapes and the same global batch, FP32 reduction and actual verl clipping.
Checkpoint uses three separate torchrun processes, stochastic data/dropout and
exact full weight/state/group/scheduler equality; files are written by the actual
verl manager, not a substitute serialization-only test.

Tolerances are declared before GPU execution in `muon_fsdp2_acceptance_worker.py`.
They account for BF16 NS (unit roundoff 2^-8), five NS rounds and multi-step feedback;
they are documented heuristic acceptance budgets, not proven error bounds. Logs
report maximum absolute weight/state errors and relative update errors. They have
not been widened after GPU execution; observed errors are in the GPU record.
A failure needs diagnosis of math, batching, precision and reduction order first.

[GRPO launcher and Qwen3.5-4B notes](fsdp2_muon_grpo.md) require explicit local model
and data paths. Qwen3.5 hybrid attention/kernel/rollout compatibility remains
unverified. The short GSM8K observations do not establish memory, speed, reward
or convergence advantages over AdamW.

## Local evidence

See the [initial CPU record](fsdp2_muon_validation.md),
[synthetic GPU record](fsdp2_muon_gpu_validation.md), and
[GRPO/submission record](fsdp2_muon_grpo_validation.md) for exact executed commands,
versions, exit codes, passed/failed/skipped counts and environment blockers.
