# Explicit runtime backports

Apply only in a disposable build environment with official TE 2.18.0 release
packages, vLLM 0.27.1, Core 0.19.0, and the candidate FlashInfer 0.6.18
dependencies installed (see the [precision contract](../README.md)). The
scripts check dependency source hashes and intentionally reject other inputs.
Do not patch running jobs or replace a shared Python environment. An exact
known patched file is a no-op; unknown or partially patched bytes are rejected.
The installed-runtime verifier still runs when all patch steps are no-ops.

```bash
PYTHON_BIN=/path/to/runtime/python bash examples/real_nvfp4/runtime_backports/apply_backports.sh
/path/to/runtime/python examples/real_nvfp4/runtime_backports/verify_runtime.py
```

The verifier checks the actual installed bytes and the native refit lifecycle,
not merely package version strings. It does not install dependencies or prove
that every platform/backend works. Configure PYTHONPATH to the frozen checkout.

Payloads:

- Core 0.19.0 already includes the stateless grouped extra-state correction from
  Megatron-LM [#5997](https://github.com/NVIDIA/Megatron-LM/pull/5997).
  `apply_backports.sh` no longer invokes `patch_megatron_checkpoint.py`;
  `verify_runtime.py` checks the unchanged release implementation instead.
  The historical patch script is not an installation step for Core 0.19.0.
- vLLM 0.27.1 already includes the CuMem-aware memory-profiling correction from
  [#49208](https://github.com/vllm-project/vllm/pull/49208), first released in
  0.27.0. No local memory-profiling backport is applied.

- `patch_megatron_fa4.py`: backport Megatron-LM
  [#6964](https://github.com/NVIDIA/Megatron-LM/pull/6964) to the pinned Core 0.19.0
  source. Check FA4 distribution metadata before importing its optional module;
  FA2's bundled `flash_attn.cute` namespace is not evidence that FA4 is installed.
  Apply this before the first Megatron/Bridge import in a freshly synced runtime.
  Source hashes reject unrelated changes; repeated application is a checked no-op.
- `apply_vllm_online_nvfp4_50029_50074.py`: native online packing and kernel reuse
  from [#50029](https://github.com/vllm-project/vllm/pull/50029) and
  [#50074](https://github.com/vllm-project/vllm/pull/50074), plus the local
  derived-scale and level-2 sleep/refit storage-lifetime corrections. These are
  still required with the audited 0.27.1 source; upgrading alone does not replace
  them.
- `candidate.py` / `patch_te.py`: TE row-scale grouped-GEMM epilogue batching;
  unsupported layouts retain the original function. The candidate's historical
  diagnostic header is preserved for hash identity; `patch_te.py` installs the
  already tested function, not its diagnostic context manager.
- `patch_runtime.py`: atomic port allocation for a single-rank vLLM executor;
  multi-rank/elastic/agent-store configurations keep the upstream selector.

Run all validation in a scheduled Blackwell allocation, with one visible GPU
and bounded CPU/thread counts. From the checkout root, using the patched runtime:

```bash
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# These limits bound incidental compilation during this test allocation.
# Dedicated native-wheel builds may use larger values within their allocation.
export MAX_JOBS=1 CMAKE_BUILD_PARALLEL_LEVEL=1
export FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH=1 TRTLLM_DISABLE_FP4_QUANT_FAST_MATH=1
export PYTHONPATH="$PWD:$PWD/examples/real_nvfp4/runtime_backports"
export EFFICIENCY_OUTPUT=/shared/diagnostics/new-unique-rowscale-output
python examples/real_nvfp4/runtime_backports/test_exported.py
python -m pytest -q examples/real_nvfp4/runtime_backports/test_atomic.py \
  examples/real_nvfp4/runtime_backports/test_native_packing.py
python -m pytest -q tests/utils/real_nvfp4 tests/utils/test_bucketed_weight_transfer.py
python examples/real_nvfp4/runtime_backports/verify_runtime.py
```

`EFFICIENCY_OUTPUT` must not exist beforehand. The differential probe tests
forward/input-gradient/weight-gradient equality across two weight versions,
uniform and ragged expert splits, and fallback/zero/small/large inputs. These
operator timings are not full-training benchmarks. The communication tests
exercise real stores/process groups and the packing tests use the actual GPU
quantizer. Full multi-node training, resume, quality and performance still need
separate matched BF16/W4A4 validation after dependency or source changes.

The runtime-delivery bundle has separate component regressions, including
real CUDA IPC and native packing checks. Installation through the repository's
updated official dependency entry points, matched six-step BF16/W4A4 execution,
repeated refit/sleep/wake, and full-Adam recovery remain separate acceptance
gates. Component checks or a successful first refit do not prove that the entire
upgrade resolves OOM or preserves training quality.

### Build with the project lock

The official uv Dockerfile can materialize the vLLM/Megatron environment and run
this existing build-time entry before publishing the image:

```bash
docker build -f docker/Dockerfile.uv.cu130 \
  --build-arg NVFP4_RUNTIME=1 --build-arg MAX_JOBS=16 -t verl:nvfp4 .
```

For a disposable build environment outside Docker, use the same two steps:

```bash
uv sync --frozen --extra megatron --extra vllm
PYTHON_BIN="$PWD/.venv/bin/python" bash examples/real_nvfp4/runtime_backports/apply_backports.sh
```

The patch entry derives the NCCL library directory from the selected Python.
For training outside the Docker image, preserve that loader preference in the
launch environment:

```bash
export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/nvidia/nccl/lib:${LD_LIBRARY_PATH:-}"
```

Apex and FlashAttention fall back to their official sources when matching native
wheels are unavailable. uv keys those builds by the selected runtime Torch; keep
`UV_CACHE_DIR` in a persistent image directory so a later build can reuse them.
The build allocation controls `MAX_JOBS`; avoid concurrent native package builds
on the same allocation (`UV_CONCURRENT_BUILDS=1`). Rebuild and reverify the NVFP4
environment when changing backend extras or dependency versions.

The Dockerfile still prefetches all supported backend combinations. Until
Torch-qualified wheelhouse sources are available, this includes one native
Apex/FlashAttention build per selected Torch ABI, including the SGLang/Torch
2.11 branch. The standard vLLM/Megatron uv installation is validated separately
from a full multi-backend Docker build; the opt-in Docker wiring alone does not
establish validation of every prefetched backend.

The universal lock keeps SGLang's Torch 2.11 and CPU Core 0.18 selections. GPU
trainers share Core 0.19, Bridge 0.6.1, TE 2.18 and Energon 7.4.1; selecting
SGLang with Megatron therefore also selects those shared trainer upgrades.
Apex/FlashAttention source builds are keyed to the runtime Torch ABI, so the
first SGLang build also needs its own native cache. SGLang training and the
complete Docker prefetch matrix have not been validated by the vLLM runtime
checks.

The version-scoped Torch metadata in `pyproject.toml` preserves the complete
published dependency lists and extras. Torch 2.11 retains the existing NCCL
2.30.7 override; both Torch versions retain the ARM cuSPARSELt 0.9.1 wheel-tag
workaround while x86 keeps each release's original pin. CUDA toolkit 13.0.2
retains the existing cublas 13.5.1.27 override. These overrides are scoped to
package versions because a global override with only an ARM marker would drop
the x86 requirement in uv. Sources audited for this metadata:

- [Torch 2.11 cu130 metadata](https://download.pytorch.org/whl/cu130/torch-2.11.0%2Bcu130-cp312-cp312-manylinux_2_28_aarch64.whl.metadata),
  SHA256 `3f58d2a0bbf02637643e6bb9b9d5926a0e577b38b37bf82ca930145529e3aafe`.
- [Torch 2.13 cu130 metadata](https://download.pytorch.org/whl/cu130/torch-2.13.0%2Bcu130-cp312-cp312-manylinux_2_28_aarch64.whl.metadata),
  SHA256 `7477a1e61a4ecc055e52577f79be3dcc6d4d2ef6d423089469c9366cf4191a80`.
- [CUDA toolkit 13.0.2 metadata](https://pypi.org/pypi/cuda-toolkit/13.0.2/json).

The Torch aarch64 and x86 wheel metadata have identical dependency lists and
extras. Re-audit the full published metadata when changing these versions.
