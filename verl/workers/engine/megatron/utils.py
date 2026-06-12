# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

import logging
import os

import torch

from verl.utils.device import get_torch_device
from verl.utils.nccl_suspend import (
    ResumeResult,
    SuspendResult,
    is_supported,
    resume_batch,
    suspend_batch,
)

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def set_random_seed(seed):
    import random

    import numpy as np
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if get_torch_device().device_count() > 0:
        from megatron.core import tensor_parallel

        tensor_parallel.model_parallel_cuda_manual_seed(seed)
    # FIXME: torch cumsum not support deterministic (used in vllm sampler),
    # https://github.com/pytorch/pytorch/issues/89492
    # torch.use_deterministic_algorithms(True, warn_only=True)
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


# Megatron NCCL communicator suspend/resume. Reflective walk of
# ``megatron.core.parallel_state`` named group globals; generic primitives in
# ``verl/utils/nccl_suspend.py``. State is module-level (process-scoped) since
# ``parallel_state`` itself is a process-global module.

_megatron_suspended: bool = False
_megatron_suspended_handles: list[tuple[str, int]] = []


def _collect_megatron_comms() -> list[tuple[str, int]]:
    """Collect warm NCCL handles from ``megatron.core.parallel_state`` globals.

    Walks ``_*GROUP*`` attrs (skipping ``_*GLOO*``), supports singleton / list /
    dict container shapes, and dedups by ``ncclComm_t`` handle since PyTorch
    may share one communicator across multiple ``ProcessGroup`` objects.

    Raises ``RuntimeError`` if Megatron is not importable. Returns ``[]``
    (with ERROR log) if model parallel is not yet initialized.
    """
    try:
        from megatron.core import parallel_state as ps
    except ImportError as e:
        raise RuntimeError(
            "suspend_nccl_comms=True requires Megatron, but megatron.core is not "
            "importable. Install Megatron or set suspend_nccl_comms=False."
        ) from e

    try:
        if not ps.model_parallel_is_initialized():
            logger.error("Megatron model parallel not initialized; skipping comm collection.")
            return []
    except Exception as e:
        logger.error("Megatron model_parallel_is_initialized check failed: %s", e)
        return []

    handles: list[tuple[str, int]] = []
    seen_ptrs: set[int] = set()

    for attr_name in sorted(dir(ps)):
        if not attr_name.startswith("_") or "GROUP" not in attr_name or "GLOO" in attr_name:
            continue
        attr = getattr(ps, attr_name, None)
        if attr is None:
            continue

        if isinstance(attr, dict):
            items = [(f"{attr_name.lstrip('_')}[{k}]", v) for k, v in attr.items()]
        elif isinstance(attr, list | tuple):
            items = [(f"{attr_name.lstrip('_')}[{i}]", v) for i, v in enumerate(attr)]
        else:
            items = [(attr_name.lstrip("_"), attr)]

        for label, pg in items:
            if pg is None:
                continue
            try:
                backend = pg._get_backend(torch.device("cuda"))
            except Exception:
                continue
            if not hasattr(backend, "_comm_ptr"):
                continue
            try:
                ptr = backend._comm_ptr()
            except Exception:
                continue
            if ptr == 0 or ptr in seen_ptrs:
                continue
            seen_ptrs.add(ptr)
            handles.append((label, int(ptr)))

    logger.info(
        "Discovered %d warm Megatron NCCL comm(s): %s",
        len(handles),
        [name for name, _ in handles],
    )
    return handles


def suspend_via_parallel_state(*, measure_per_comm: bool = False) -> SuspendResult:
    """Suspend all warm NCCL comms reachable via ``megatron.core.parallel_state``.

    Idempotent: returns a no-op result if already suspended.
    """
    global _megatron_suspended, _megatron_suspended_handles

    if not is_supported():
        return SuspendResult(success=False, skipped_reason="nccl_too_old")

    if _megatron_suspended:
        logger.debug("Megatron NCCL comms already suspended; no-op.")
        return SuspendResult(success=False, skipped_reason="already_suspended")

    handles = _collect_megatron_comms()
    if not handles:
        return SuspendResult(success=False, skipped_reason="no_warm_comms")

    result = suspend_batch(handles, measure_per_comm=measure_per_comm)
    if result.success:
        _megatron_suspended = True
        _megatron_suspended_handles = handles
    return result


def resume_via_parallel_state(*, measure_per_comm: bool = False) -> ResumeResult:
    """Resume Megatron NCCL comms suspended by ``suspend_via_parallel_state``.

    Idempotent: returns a no-op result if not suspended.
    """
    global _megatron_suspended

    if not is_supported():
        return ResumeResult(success=False, skipped_reason="nccl_too_old")

    if not _megatron_suspended:
        logger.debug("Megatron NCCL comms not suspended; no-op.")
        return ResumeResult(success=False, skipped_reason="not_suspended")

    handles = _megatron_suspended_handles
    if not handles:
        return ResumeResult(success=False, skipped_reason="no_warm_comms")

    result = resume_batch(handles, measure_per_comm=measure_per_comm)
    if result.success:
        _megatron_suspended = False
    return result
