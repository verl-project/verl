"""Opt-in CUDA memory accounting for the DSV41 Megatron engine.

The helpers in this module deliberately inspect allocator counters and existing
objects only.  They do not retain tensor references or perform CUDA
synchronization when the profile is disabled (the default).
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Iterable

import torch

_GB = float(1 << 30)

# One million allocator events covers a full diagnostic step on the DSV41
# training path while keeping the history bounded.  The history stores a stack
# trace for each event, so this is intentionally opt-in and should be used for
# a diagnostic run only.
_MEMORY_HISTORY_MAX_ENTRIES = 1_000_000
_memory_history_active = False


def enabled() -> bool:
    """Return whether the DSV41 memory profile is enabled."""
    return os.environ.get("DSV41_MEM_PROFILE") == "1"


def parts_enabled() -> bool:
    """Allow the potentially expensive optimizer-state walk to be disabled."""
    return os.environ.get("DSV41_MEM_PARTS", "1") == "1"


def _rank_zero() -> bool:
    return not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0


def _history_rank_zero() -> bool:
    """Select the single process that owns a memory-history output file.

    Global rank zero is also CP rank zero in the Megatron launch topology.  The
    explicit CP check prevents a future rank mapping from recording history on
    a nonzero context-parallel rank while retaining one writer per path.
    """
    if not _rank_zero():
        return False
    try:
        from megatron.core import parallel_state as mpu

        return mpu.get_context_parallel_rank() == 0
    except (AssertionError, ImportError, RuntimeError):
        # Unit tests and non-Megatron callers have no parallel-state object.
        return True


def log_phase(phase: str) -> None:
    """Print allocator counters for one named phase on rank zero."""
    if not enabled() or not _rank_zero():
        return
    if not torch.cuda.is_available():
        print(
            f"[mem] phase={phase} alloc=n/a(cuda unavailable) peak=n/a(cuda unavailable) "
            "reserved=n/a(cuda unavailable)",
            flush=True,
        )
        return
    print(
        f"[mem] phase={phase} alloc={torch.cuda.memory_allocated() / _GB:.2f}GB "
        f"peak={torch.cuda.max_memory_allocated() / _GB:.2f}GB "
        f"reserved={torch.cuda.memory_reserved() / _GB:.2f}GB",
        flush=True,
    )


def _iter_models(model: Any) -> Iterable[Any]:
    if model is None:
        return
    if isinstance(model, list | tuple):
        for child in model:
            yield from _iter_models(child)
    else:
        yield model


def _iter_parameters(model: Any) -> Iterable[Any]:
    for child in _iter_models(model):
        parameters = getattr(child, "parameters", None)
        if parameters is not None:
            yield from parameters()


def _tensor_bytes(tensor: Any) -> int:
    """Count one CUDA tensor without retaining it or touching CPU storage."""
    if not isinstance(tensor, torch.Tensor) or not tensor.is_cuda:
        return 0
    return int(tensor.numel()) * int(tensor.element_size())


def _sum_attribute(
    parameters: Iterable[Any],
    name: str,
    *,
    required: bool = False,
) -> tuple[int, str | None]:
    total = 0
    missing = False
    sentinel = object()
    for parameter in parameters:
        try:
            value = getattr(parameter, name, sentinel)
            if value is sentinel:
                missing = True
            else:
                total += _tensor_bytes(value)
                del value
        finally:
            del parameter
    reason = f"parameter attribute {name!r} is unavailable" if required and missing else None
    return total, reason


def _iter_optimizers(optimizer: Any) -> Iterable[Any]:
    """Expand chains before reading properties forwarded to the inner optimizer."""
    pending = [optimizer]
    visited: set[int] = set()
    while pending:
        current = pending.pop()
        if current is None or id(current) in visited:
            continue
        visited.add(id(current))
        children = getattr(current, "chained_optimizers", None)
        if isinstance(children, list | tuple):
            pending.extend(children)
        else:
            yield current


def _nested_tensor_bytes(value: Any, seen: set[tuple[Any, ...]]) -> int:
    if isinstance(value, torch.Tensor):
        if not value.is_cuda:
            return 0
        key = (value.device, value.data_ptr(), value.numel(), value.element_size())
        if key in seen:
            return 0
        seen.add(key)
        return _tensor_bytes(value)
    if isinstance(value, Mapping):
        return sum(_nested_tensor_bytes(item, seen) for item in value.values())
    if isinstance(value, list | tuple):
        return sum(_nested_tensor_bytes(item, seen) for item in value)
    return 0


def _sum_main_grads(model: Any, optimizer: Any) -> tuple[int | None, str | None]:
    """Use DDP gradients, falling back to optimizer-owned main/shard gradients."""
    total = 0
    seen: set[tuple[Any, ...]] = set()
    missing = False
    try:
        for parameter in _iter_parameters(model):
            if not parameter.requires_grad:
                continue
            if not hasattr(parameter, "main_grad"):
                missing = True
                continue
            total += _nested_tensor_bytes(parameter.main_grad, seen)
        if not missing:
            return total, None

        # DistOpt installs .grad (or precision-aware .decoupled_grad) on the
        # inner optimizer's main parameters; frozen model weights are absent.
        total = 0
        seen.clear()
        found = False
        for current in _iter_optimizers(optimizer):
            if getattr(current, "is_stub_optimizer", False):
                continue
            groups = getattr(current, "param_groups", None)
            if groups is None:
                return None, "main gradients unavailable: optimizer param_groups unavailable"
            for group in groups:
                for parameter in group["params"]:
                    value = getattr(parameter, "decoupled_grad", None)
                    if value is None:
                        value = getattr(parameter, "grad", None)
                    found = found or value is not None
                    total += _nested_tensor_bytes(value, seen)
                    del value
        if found:
            return total, None
    except Exception as exc:  # Diagnostics must not change training behavior.
        return None, f"main gradients unavailable: {type(exc).__name__}: {exc}"
    return None, "main gradients unavailable: no main_grad or optimizer gradient tensors"


def _sum_optimizer_states(optimizer: Any) -> tuple[dict[str, int | str] | None, str | None]:
    totals: dict[str, int | str] = {
        "optim_exp_avg": 0,
        "optim_exp_avg_sq": 0,
        "optim_master": 0,
        "optim_other": 0,
    }
    observed = {name: False for name in totals}
    seen: set[tuple[Any, ...]] = set()
    found = False
    try:
        for current in _iter_optimizers(optimizer):
            if getattr(current, "is_stub_optimizer", False):
                continue
            state = getattr(current, "state", None)
            if state is None:
                inner = getattr(current, "optimizer", None)
                state = getattr(inner, "state", None)
            if not isinstance(state, Mapping):
                return None, "optimizer state unavailable: optimizer exposes no state mapping"
            found = True
            for values in state.values():
                if not isinstance(values, Mapping):
                    totals["optim_other"] += _nested_tensor_bytes(values, seen)
                    continue
                for key, value in values.items():
                    if key == "exp_avg":
                        bucket = "optim_exp_avg"
                    elif key == "exp_avg_sq":
                        bucket = "optim_exp_avg_sq"
                    elif key in ("master_param", "fp32_param", "main_param"):
                        bucket = "optim_master"
                    else:
                        bucket = "optim_other"
                    totals[bucket] += _nested_tensor_bytes(value, seen)
                    observed[bucket] = True
                    del value
            # Ordinary mixed precision stores masters outside state; precision-
            # aware FusedAdam keeps them under the master_param state key.
            for name in ("shard_fp32_from_float16_groups", "fp32_from_float16_groups"):
                master = getattr(current, name, None)
                if master is not None:
                    totals["optim_master"] += _nested_tensor_bytes(master, seen)
                    observed["optim_master"] = True
                    del master
    except Exception as exc:  # Diagnostics must not change training behavior.
        return None, f"optimizer state unavailable: {type(exc).__name__}: {exc}"
    if not found:
        return None, "optimizer exposes no state mapping"
    for bucket in ("optim_exp_avg", "optim_exp_avg_sq", "optim_master"):
        if not observed[bucket]:
            totals[bucket] = "n/a"
    return totals, None


def log_parts(model: Any, optimizer: Any) -> None:
    """Print a best-effort GPU component ledger at the end of an optimizer step."""
    if not enabled() or not _rank_zero():
        return
    if not parts_enabled():
        print("[mem-parts] skipped=DSV41_MEM_PARTS=0", flush=True)
        return
    if not torch.cuda.is_available():
        print(
            "[mem-parts] params=n/a(cuda unavailable) grads=n/a(cuda unavailable) "
            "main_grads=n/a(cuda unavailable) optim_exp_avg=n/a(cuda unavailable) "
            "optim_exp_avg_sq=n/a(cuda unavailable) optim_master=n/a(cuda unavailable) "
            "optim_other=n/a(cuda unavailable) other=n/a(cuda unavailable)",
            flush=True,
        )
        return
    try:
        params, params_reason = _sum_attribute(_iter_parameters(model), "data", required=True)
        grads, grads_reason = _sum_attribute(_iter_parameters(model), "grad")
        main_grads, main_grads_reason = _sum_main_grads(model, optimizer)
    except Exception as exc:  # pragma: no cover - depends on wrapper properties
        params = grads = main_grads = None
        params_reason = grads_reason = main_grads_reason = (
            f"parameter traversal unavailable: {type(exc).__name__}: {exc}"
        )
    else:
        params_reason = params_reason or None
        grads_reason = grads_reason or None
        main_grads_reason = main_grads_reason or None
    state_totals, state_reason = _sum_optimizer_states(optimizer)
    allocated = int(torch.cuda.memory_allocated())
    reasons = (params_reason, grads_reason, main_grads_reason, state_reason)
    known_values = (params, grads, main_grads, *(state_totals or {}).values())
    known = sum(value for value in known_values if isinstance(value, int))
    other = (
        allocated - known
        if all(reason is None for reason in reasons) and all(isinstance(value, int) for value in known_values)
        else None
    )

    def field(value: Any, reason: str | None = None) -> str:
        if isinstance(value, int) and reason is None:
            return f"{value / _GB:.2f}GB"
        return f"n/a({reason or 'unavailable'})"

    print(
        "[mem-parts] "
        f"params={field(params, params_reason)} "
        f"grads={field(grads, grads_reason)} "
        f"main_grads={field(main_grads, main_grads_reason)} "
        f"optim_exp_avg={field(None if state_totals is None else state_totals['optim_exp_avg'], state_reason)} "
        f"optim_exp_avg_sq={field(None if state_totals is None else state_totals['optim_exp_avg_sq'], state_reason)} "
        f"optim_master={field(None if state_totals is None else state_totals['optim_master'], state_reason)} "
        f"optim_other={field(None if state_totals is None else state_totals['optim_other'], state_reason)} "
        f"other={field(other, 'component totals unavailable' if other is None else None)}",
        flush=True,
    )


def write_snapshot(path: str | None) -> None:
    """Write one allocator snapshot, if requested, after the step completes."""
    if not enabled() or not path or not _rank_zero():
        return
    if not torch.cuda.is_available():
        print(f"[mem] snapshot={path} error=cuda unavailable", flush=True)
        return
    try:
        snapshot = torch.cuda.memory_snapshot()
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            json.dump(snapshot, handle)
        del snapshot
        print(f"[mem] snapshot={output}", flush=True)
    except Exception as exc:  # pragma: no cover - allocator/version dependent
        print(f"[mem] snapshot={path} error={type(exc).__name__}: {exc}", flush=True)


def start_memory_history(path: str | None) -> None:
    """Start allocator history recording for one training step, if requested.

    ``path`` is only used as an enable switch here; the snapshot is written by
    :func:`finish_memory_history` after the optimizer step.  Recording starts
    before forward/backward so a peak allocation that is freed before the step
    ends is still present in the history.
    """
    global _memory_history_active
    if not path or not _history_rank_zero() or not torch.cuda.is_available():
        return
    try:
        torch.cuda.memory._record_memory_history(max_entries=_MEMORY_HISTORY_MAX_ENTRIES)
        _memory_history_active = True
        print(
            f"[mem] history_recording=enabled max_entries={_MEMORY_HISTORY_MAX_ENTRIES}",
            flush=True,
        )
    except Exception as exc:  # pragma: no cover - allocator/version dependent
        print(f"[mem] history_recording={path} error={type(exc).__name__}: {exc}", flush=True)


def finish_memory_history(path: str | None) -> None:
    """Dump and disable the allocator history for the current training step."""
    global _memory_history_active
    if not path or not _memory_history_active:
        return
    _memory_history_active = False
    if not _history_rank_zero() or not torch.cuda.is_available():
        return
    try:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.cuda.memory._dump_snapshot(str(output))
        print(f"[mem] history={output}", flush=True)
    except Exception as exc:  # pragma: no cover - allocator/version dependent
        print(f"[mem] history={path} error={type(exc).__name__}: {exc}", flush=True)
    finally:
        try:
            torch.cuda.memory._record_memory_history(enabled=None)
        except Exception as exc:  # pragma: no cover - allocator/version dependent
            print(
                f"[mem] history_recording=disable error={type(exc).__name__}: {exc}",
                flush=True,
            )
