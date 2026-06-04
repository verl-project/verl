# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Runtime patches for colocated FSDP + vLLM on Intel GPU.

Root cause: Intel GPU has no CuMemAllocator equivalent.  CUDA uses a shared memory
pool so that FSDP and vLLM see a unified allocation picture; on Intel GPU they
each see raw Level-Zero driver free-memory.

  Upstream fix in progress:
    https://github.com/vllm-project/vllm/pull/37149   (XpuMemAllocator /
    transparent sleep mode — draft, requires torch-xpu 2.11)

Patches in apply() will be removed once PR #37149 merges and torch-xpu ≥ 2.11 is
the minimum supported version.

Usage:
    python3 -c "from verl.utils.vllm import intel_gpu_patches; intel_gpu_patches.apply()"
"""

import logging

logger = logging.getLogger(__name__)

# If this is set, patches have already been applied in this process.
_APPLIED = False


def _patch_request_memory() -> bool:
    """Wrap vllm.v1.worker.utils.request_memory to skip the OOM check on Intel GPU.

    vLLM checks ``free_memory >= requested_memory`` at startup.  When FSDP is
    already resident on the GPU this raises ValueError even though FSDP will
    CPU-offload before inference begins.

    The check is skipped entirely on Intel GPU because Level-Zero context overhead
    from Ray pre-started workers inflates ``used`` memory by ~20 GB, making
    the free-memory figure unreliable.

    Upstream reference: https://github.com/vllm-project/vllm/pull/37149
    """
    try:
        import torch
        import vllm.v1.worker.utils as _utils

        if not hasattr(_utils, "request_memory"):
            logger.warning(
                "[XPU patch] vllm.v1.worker.utils.request_memory not found — skipping Patch 1 (API may have changed)"
            )
            return False

        if getattr(_utils.request_memory, "_xpu_patched", False):
            return True  # already applied

        _orig = _utils.request_memory

        def _patched_request_memory(init_snapshot, requested_memory):
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                # Intel GPU: Level-Zero context overhead inflates "used" memory.
                # FSDP will CPU-offload before inference so the ValueError is
                # spurious.  See verl/utils/vllm/intel_gpu_patches.py for details.
                return
            return _orig(init_snapshot, requested_memory)

        _patched_request_memory._xpu_patched = True
        _utils.request_memory = _patched_request_memory
        logger.info("[XPU patch] Applied Patch 1: request_memory OOM check skipped on XPU")
        return True

    except ImportError as e:
        logger.warning(f"[XPU patch] Could not import vllm.v1.worker.utils: {e} — skipping Patch 1")
        return False


def _patch_profiling_assert() -> bool:
    """Wrap the vLLM GPUWorker profiling method to tolerate Intel GPU assert.

    vLLM asserts that free GPU memory did not grow during the profiling run.
    When FSDP offloads parameters to CPU during vLLM init, free memory
    *increases*, tripping the assert even though this is expected behaviour.

    Instead of string-patching the source, we wrap the method and convert the
    AssertionError to a warning on Intel GPU only.

    Affected code (vllm/v1/worker/gpu_worker.py):
        assert self.init_snapshot.free_memory >= free_gpu_memory, (...)

    Upstream reference: https://github.com/vllm-project/vllm/pull/36720
    """
    try:
        import torch
        from vllm.v1.worker.gpu_worker import GPUWorker

        # The assert lives in the method that calls profile_run and then
        # compares snapshots — typically `determine_num_available_blocks`.
        # Try both known method names for robustness across vLLM versions.
        _TARGET_METHODS = ("determine_num_available_blocks", "profile_run")

        patched_any = False
        for method_name in _TARGET_METHODS:
            orig = getattr(GPUWorker, method_name, None)
            if orig is None:
                continue
            if getattr(orig, "_xpu_patched", False):
                patched_any = True
                continue

            def _make_patched(original, mname):
                def _patched(self, *args, **kwargs):
                    try:
                        return original(self, *args, **kwargs)
                    except AssertionError as exc:
                        if hasattr(torch, "xpu") and torch.xpu.is_available():
                            logger.warning(
                                f"[XPU patch] Suppressed profiling AssertionError in "
                                f"GPUWorker.{mname} (FSDP offloaded params, free "
                                f"memory increased — expected on XPU). "
                                f"Original: {exc}"
                            )
                            # Return expected type to avoid TypeError in caller
                            return (0, 0) if mname == "determine_num_available_blocks" else None
                        raise

                _patched._xpu_patched = True
                _patched.__name__ = original.__name__
                _patched.__qualname__ = original.__qualname__
                return _patched

            setattr(GPUWorker, method_name, _make_patched(orig, method_name))
            logger.info(f"[XPU patch] Applied Patch 2: GPUWorker.{method_name} profiling assert wrapped")
            patched_any = True

        if not patched_any:
            logger.warning(
                f"[XPU patch] None of {_TARGET_METHODS} found on GPUWorker — "
                "skipping Patch 2 (vLLM API may have changed)"
            )
        return patched_any

    except ImportError as e:
        logger.warning(f"[XPU patch] Could not import vllm.v1.worker.gpu_worker: {e} — skipping Patch 2")
        return False


def apply() -> None:
    """Apply all XPU-specific vLLM patches.  Safe to call multiple times.

    Returns quietly if vLLM is not installed or patches are already applied.
    """
    global _APPLIED
    if _APPLIED:
        return

    try:
        import torch

        if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
            return  # Only apply patches on XPU hosts
    except ImportError:
        logger.warning("[XPU patch] torch not found — skipping XPU vLLM patches")
        return
    except Exception:
        return

    p1 = _patch_request_memory()
    p2 = _patch_profiling_assert()

    if p1 and p2:
        logger.info("[XPU patch] All vLLM XPU patches applied successfully")
    elif not p1 and not p2:
        logger.warning("[XPU patch] No patches applied — check vLLM installation")

    _APPLIED = True
