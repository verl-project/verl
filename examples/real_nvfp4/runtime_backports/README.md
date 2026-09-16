# Explicit runtime backports

Apply only in a disposable build environment with official TE 2.18.0 release
packages, vLLM 0.26.0 and the pinned FlashInfer dependencies installed. The
scripts check dependency source hashes and intentionally reject other inputs.
Do not patch running jobs or replace a shared Python environment. The TE and
TCPStore patch steps are not idempotent: for an already patched image, verify
instead of applying again.

```bash
PYTHON_BIN=/path/to/runtime/python bash examples/real_nvfp4/runtime_backports/apply_backports.sh
/path/to/runtime/python examples/real_nvfp4/runtime_backports/verify_runtime.py
```

The verifier checks the actual installed bytes and the native refit lifecycle,
not merely package version strings. It does not install dependencies or prove
that every platform/backend works. Configure PYTHONPATH to the frozen checkout.

Payloads:

- `patch_megatron_checkpoint.py`: backport the stateless grouped extra-state
  handling from Megatron-LM [#5997](https://github.com/NVIDIA/Megatron-LM/pull/5997).
  TE 2.18 returns an empty byte tensor for stateless recipes; this is valid and
  must not be decoded and indexed as a nonempty FP8 metadata dictionary.
  The guard checks the installed source, not just its version string.

- `patch_megatron_fa4.py`: backport Megatron-LM
  [#6964](https://github.com/NVIDIA/Megatron-LM/pull/6964) to the pinned Core 0.18
  source. Check FA4 distribution metadata before importing its optional module;
  FA2's bundled `flash_attn.cute` namespace is not evidence that FA4 is installed.
  Apply this before the first Megatron/Bridge import in a freshly synced runtime.
  Source hashes reject unrelated changes; repeated application is a checked no-op.
- `apply_vllm_online_nvfp4_50029_50074.py`: native online packing, kernel reuse,
  derived-scale and level-2 sleep/refit storage-lifetime corrections.
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

These payloads were copied from the separately tested runtime-delivery bundle.
Merged-source regression passed in the existing patched runtime, including
real CUDA IPC and native packing checks. Installation from the newly resolved
lock and subsequent full-model validation remain separate acceptance gates.
