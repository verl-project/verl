# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""Helpers for resident multi-LoRA adapters on a shared base model.

Provides adapter resolution, parameter/submodule selection, per-adapter optimizer
rebind and LR scheduling, batch splitting, and cross-rank execution ordering.
"""

from __future__ import annotations

import json
import os
from collections import OrderedDict
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from contextlib import contextmanager
from typing import Any, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

from verl.utils.py_functional import convert_to_regular_types

DEFAULT_ADAPTER_NAME = "default"

_UNADAPTED_LEAF_NAMES = frozenset({"weight", "bias"})


def validate_adapter_name(adapter_name: str) -> str:
    """Validate that ``adapter_name`` is a non-empty, PEFT-safe, filesystem-safe identifier."""
    if not isinstance(adapter_name, str) or not adapter_name:
        raise ValueError(f"Adapter name must be a non-empty string, got {adapter_name!r}")
    if (
        adapter_name in {".", ".."}
        or "." in adapter_name
        or "/" in adapter_name
        or "\\" in adapter_name
        or "\x00" in adapter_name
    ):
        raise ValueError(
            f"Invalid adapter name {adapter_name!r}: must not contain '.', '/', '\\\\', or NUL characters."
        )
    return adapter_name


def _is_peft_container_segment(segment: str) -> bool:
    """Return whether a dotted parameter-key segment is a PEFT adapter container."""
    return segment == "modules_to_save" or segment.startswith("lora_") or segment.startswith("adapter_")


def unwrap_peft_model(module: Any) -> Any:
    """Unwrap FSDP wrapper layers to reach the underlying PeftModel or nn.Module."""
    cur = module
    seen: set[int] = set()
    while hasattr(cur, "_fsdp_wrapped_module") and id(cur) not in seen:
        seen.add(id(cur))
        cur = cur._fsdp_wrapped_module
    return cur


def is_peft_model(module: Any) -> bool:
    """Return True if ``module`` (or its FSDP-unwrapped inner model) has PEFT adapters."""
    peft_config = getattr(unwrap_peft_model(module), "peft_config", None)
    return isinstance(peft_config, dict) and len(peft_config) > 0


def list_peft_adapters(module: Any) -> list[str]:
    """Return registered PEFT adapter names in registration order."""
    peft_config = getattr(unwrap_peft_model(module), "peft_config", None)
    if not isinstance(peft_config, dict):
        return []
    return list(peft_config.keys())


def resolve_adapter_name(module: Any, adapter_name: Optional[str] = None) -> str:
    """Resolve an explicit or active PEFT adapter name without hardcoding ``"default"``.

    Resolution order when ``adapter_name`` is ``None``:
    1. ``peft_model.active_adapter`` (if a single registered adapter name)
    2. ``peft_model.active_adapters`` (if a single-element sequence of a registered adapter)
    3. ``"default"`` if registered
    4. First registered adapter in ``peft_model.peft_config``
    """
    peft_model = unwrap_peft_model(module)
    peft_config = getattr(peft_model, "peft_config", None)
    if not isinstance(peft_config, dict) or not peft_config:
        raise ValueError("Module does not have any PEFT adapters configured.")

    if adapter_name is not None:
        if adapter_name not in peft_config:
            raise KeyError(
                f"Adapter {adapter_name!r} not found in peft_config; available adapters: {list(peft_config.keys())}"
            )
        return adapter_name

    active = getattr(peft_model, "active_adapter", None)
    if isinstance(active, str) and active in peft_config:
        return active
    if isinstance(active, (list, tuple)) and len(active) == 1 and active[0] in peft_config:
        return active[0]

    active_adapters = getattr(peft_model, "active_adapters", None)
    if isinstance(active_adapters, (list, tuple)) and len(active_adapters) == 1 and active_adapters[0] in peft_config:
        return active_adapters[0]

    if DEFAULT_ADAPTER_NAME in peft_config:
        return DEFAULT_ADAPTER_NAME
    return next(iter(peft_config))


def get_peft_config(module: Any, adapter_name: Optional[str] = None) -> Any:
    """Return the PEFT config object for ``adapter_name`` (or the resolved active adapter)."""
    peft_model = unwrap_peft_model(module)
    resolved = resolve_adapter_name(peft_model, adapter_name)
    return peft_model.peft_config[resolved]


def extract_adapter_name_from_param_key(
    param_name: str,
    known_adapters: Optional[Collection[str]] = None,
) -> Optional[str]:
    """Extract the PEFT adapter name from a parameter or state-dict key.

    Inspects the segment immediately following a PEFT container segment
    (``lora_*``, ``modules_to_save``, ``adapter_*``) so adapter names that happen
    to collide with base module names (such as ``model`` or ``layers``) are never
    misidentified.

    Examples:
        - ``...q_proj.lora_A.tenant_a.weight`` -> ``"tenant_a"``
        - ``...experts.lora_A.tenant_a`` -> ``"tenant_a"``
        - ``...embed_tokens.modules_to_save.tenant_a.weight`` -> ``"tenant_a"``
        - ``...q_proj.lora_A.weight`` (already stripped) -> ``None``
    """
    clean_name = param_name.replace("_fsdp_wrapped_module.", "")
    parts = clean_name.split(".")
    for idx, part in enumerate(parts):
        if _is_peft_container_segment(part) and idx + 1 < len(parts):
            candidate = parts[idx + 1]
            if known_adapters is not None:
                if candidate in known_adapters:
                    return candidate
            elif candidate not in _UNADAPTED_LEAF_NAMES or idx + 2 < len(parts):
                return candidate
    return None


def param_belongs_to_adapter(
    param_name: str,
    adapter_name: str,
    all_adapter_names: Optional[Collection[str]] = None,
) -> bool:
    """Return True if ``param_name`` belongs to ``adapter_name``."""
    extracted = extract_adapter_name_from_param_key(param_name, known_adapters=all_adapter_names)
    return extracted == adapter_name


def strip_adapter_name_from_param_key(param_name: str, adapter_name: str) -> str:
    """Strip the PEFT adapter name segment from a parameter key.

    Matches PEFT's ``get_peft_model_state_dict`` key conventions:
    - ``...q_proj.lora_A.<adapter>.weight`` -> ``...q_proj.lora_A.weight``
    - ``...experts.lora_A.<adapter>`` -> ``...experts.lora_A``
    - ``...head.modules_to_save.<adapter>.weight`` -> ``...head.weight``
    """
    parts = param_name.split(".")
    for idx, part in enumerate(parts):
        if _is_peft_container_segment(part) and idx + 1 < len(parts) and parts[idx + 1] == adapter_name:
            if part == "modules_to_save":
                return ".".join(parts[:idx] + parts[idx + 2 :])
            return ".".join(parts[: idx + 1] + parts[idx + 2 :])
    return param_name


@contextmanager
def adapter_context(module: Any, adapter_name: str):
    """Temporarily activate ``adapter_name`` on a PeftModel and restore the previous adapter on exit."""
    peft_model = unwrap_peft_model(module)
    resolved = resolve_adapter_name(peft_model, adapter_name)

    prev_active_adapters = getattr(peft_model, "active_adapters", None)
    if isinstance(prev_active_adapters, (list, tuple)) and len(prev_active_adapters) > 0:
        prev_to_restore: Any = list(prev_active_adapters) if len(prev_active_adapters) > 1 else prev_active_adapters[0]
    else:
        prev_to_restore = getattr(peft_model, "active_adapter", None)

    peft_model.set_adapter(resolved)
    try:
        yield resolved
    finally:
        if prev_to_restore is not None:
            peft_model.set_adapter(prev_to_restore)


def select_adapter_parameters(module: nn.Module, adapter_name: str) -> list[nn.Parameter]:
    """Return all parameters belonging to ``adapter_name`` regardless of current ``requires_grad`` state."""
    all_adapters: Optional[list[str]] = None
    if is_peft_model(module):
        adapter_name = resolve_adapter_name(module, adapter_name)
        all_adapters = list_peft_adapters(module)

    params: list[nn.Parameter] = []
    seen: set[int] = set()
    for name, param in module.named_parameters():
        if id(param) in seen:
            continue
        if param_belongs_to_adapter(name, adapter_name, all_adapter_names=all_adapters):
            seen.add(id(param))
            params.append(param)
    return params


def collect_adapter_submodules(root_module: nn.Module, adapter_name: str) -> list[nn.Module]:
    """Collect leaf ``nn.Module`` instances owned by ``adapter_name`` under ``root_module``.

    Intended for per-adapter FSDP2 unit wrapping (one FSDP2 unit per adapter per
    decoder layer) so inactive adapters are never all-gathered.
    """
    from peft.tuners.lora import LoraLayer

    submodules: list[nn.Module] = []
    seen: set[int] = set()

    def _add(sub: Any) -> None:
        if isinstance(sub, nn.Module) and id(sub) not in seen and any(True for _ in sub.parameters()):
            seen.add(id(sub))
            submodules.append(sub)

    for mod in root_module.modules():
        if isinstance(mod, LoraLayer):
            for attr in ("lora_A", "lora_B", "lora_magnitude_vector"):
                container = getattr(mod, attr, None)
                if container is not None and adapter_name in container:
                    _add(container[adapter_name])
        modules_to_save = getattr(mod, "modules_to_save", None)
        if isinstance(modules_to_save, nn.ModuleDict) and adapter_name in modules_to_save:
            _add(modules_to_save[adapter_name])

    return submodules


def collect_all_adapter_leaf_module_ids(root_module: nn.Module) -> set[int]:
    """Return ``id(sub)`` for every resident adapter leaf ``nn.Module`` under ``root_module``."""
    adapters = list_peft_adapters(root_module)
    if not adapters:
        return set()
    leaf_ids: set[int] = set()
    for adapter_name in adapters:
        for sub in collect_adapter_submodules(root_module, adapter_name):
            leaf_ids.add(id(sub))
    return leaf_ids


def build_lora_config_from_spec(
    adapter_spec: Mapping[str, Any],
    model_config: Any,
    adapter_name: Optional[str] = None,
):
    """Build a ``peft.LoraConfig`` from a per-adapter spec dict with ``model_config`` fallbacks."""
    from peft import LoraConfig, TaskType

    spec = convert_to_regular_types(adapter_spec) if adapter_spec else {}
    raw_rank = spec.get("r", spec.get("rank", getattr(model_config, "lora_rank", 0)))
    rank = int(raw_rank or 0)
    if rank <= 0:
        label = f" for adapter {adapter_name!r}" if adapter_name is not None else ""
        raise ValueError(f"LoRA rank must be positive{label}, got {raw_rank!r}")

    raw_alpha = spec.get("lora_alpha", spec.get("alpha", getattr(model_config, "lora_alpha", 16)))
    lora_alpha = int(raw_alpha)

    target_modules = spec.get("target_modules", getattr(model_config, "target_modules", "all-linear"))
    target_parameters = spec.get("target_parameters", getattr(model_config, "target_parameters", None))
    exclude_modules = spec.get("exclude_modules", getattr(model_config, "exclude_modules", None))
    lora_dropout = float(spec.get("lora_dropout", 0.0))
    bias = spec.get("bias", "none")
    modules_to_save = spec.get("modules_to_save", None)

    raw_task_type = spec.get("task_type", TaskType.CAUSAL_LM)
    if isinstance(raw_task_type, str):
        task_type = TaskType[raw_task_type] if raw_task_type in TaskType.__members__ else TaskType(raw_task_type)
    else:
        task_type = raw_task_type

    kwargs: dict[str, Any] = {
        "task_type": task_type,
        "r": rank,
        "lora_alpha": lora_alpha,
        "target_modules": convert_to_regular_types(target_modules),
        "target_parameters": convert_to_regular_types(target_parameters),
        "exclude_modules": convert_to_regular_types(exclude_modules),
        "lora_dropout": lora_dropout,
        "bias": bias,
    }
    if modules_to_save is not None:
        kwargs["modules_to_save"] = convert_to_regular_types(modules_to_save)

    return LoraConfig(**kwargs)


def _read_adapter_rank_from_path(adapter_path: str) -> Optional[int]:
    """Read LoRA rank ``r`` from a local PEFT ``adapter_config.json`` if accessible."""
    config_path = os.path.join(os.path.expanduser(str(adapter_path)), "adapter_config.json")
    if not os.path.isfile(config_path):
        return None
    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)
    raw_r = cfg.get("r")
    return int(raw_r) if raw_r is not None and int(raw_r) > 0 else None


def get_max_lora_rank(model_config: Any) -> int:
    """Return the maximum configured LoRA rank across single-adapter and multi-adapter config fields.

    When an adapter specifies ``adapter_path`` (or ``lora_adapter_path``), inspects the
    local ``adapter_config.json`` so path-only adapter configs do not resolve to rank 0
    or underestimate a loaded adapter's rank.
    """
    lora_dict = getattr(model_config, "lora", None) or {}
    ranks = [
        int(lora_dict.get("rank", 0) or 0),
        int(getattr(model_config, "lora_rank", 0) or 0),
    ]
    top_adapter_path = getattr(model_config, "lora_adapter_path", None)
    if top_adapter_path is not None:
        loaded_r = _read_adapter_rank_from_path(top_adapter_path)
        if loaded_r is not None:
            ranks.append(loaded_r)

    lora_adapters = getattr(model_config, "lora_adapters", None) or {}
    for adapter_name, spec in lora_adapters.items():
        if not isinstance(spec, Mapping):
            continue
        raw_r = spec.get("r", spec.get("rank", 0))
        explicit_r = int(raw_r or 0)
        adapter_path = spec.get("adapter_path", spec.get("lora_adapter_path", None))
        loaded_r = _read_adapter_rank_from_path(adapter_path) if adapter_path is not None else None
        if adapter_path is not None and loaded_r is None and explicit_r <= 0:
            raise ValueError(
                f"Cannot determine LoRA rank for adapter {adapter_name!r} from adapter_path={adapter_path!r}: "
                f"no readable local adapter_config.json and no explicit 'r'/'rank' in lora_adapters[{adapter_name!r}]."
            )
        if loaded_r is not None:
            ranks.append(loaded_r)
        if explicit_r > 0:
            ranks.append(explicit_r)
    return max(ranks)


def get_max_loras(model_config: Any) -> int:
    """Return the number of concurrent resident LoRA slots to allocate on rollout servers."""
    lora_dict = getattr(model_config, "lora", None) or {}
    lora_adapters = getattr(model_config, "lora_adapters", None) or {}
    inferred = max(1, len(lora_adapters))
    configured = lora_dict.get("max_loras", None)
    if configured is not None:
        configured_int = int(configured)
        if configured_int < 1:
            raise ValueError(f"model.lora.max_loras must be >= 1, got {configured!r}")
        return max(configured_int, inferred)
    return inferred


def build_adapter_param_groups(
    module: nn.Module,
    adapter_names: Optional[Sequence[str]] = None,
    adapter_specs: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> list[dict[str, Any]]:
    """Build per-adapter optimizer parameter groups tagged with ``"adapter_name"``.

    Each resident adapter gets its own parameter group so ``bind_optimizer_to_adapter``
    can zero, clip, and step a single adapter without disturbing any other adapter's
    accumulated gradients or optimizer state.
    """
    names = list(adapter_names) if adapter_names is not None else list_peft_adapters(module)
    if not names:
        raise ValueError("Cannot build adapter param groups: no PEFT adapters found on module.")

    groups: list[dict[str, Any]] = []
    for name in names:
        params = select_adapter_parameters(module, name)
        if not params:
            raise ValueError(f"Adapter {name!r} has no parameters on module.")
        group: dict[str, Any] = {"params": params, "adapter_name": name}
        if adapter_specs and name in adapter_specs:
            spec = convert_to_regular_types(adapter_specs[name])
            for opt_key in ("lr", "weight_decay", "betas"):
                if opt_key in spec and spec[opt_key] is not None:
                    val = spec[opt_key]
                    if opt_key == "betas" and isinstance(val, list):
                        val = tuple(val)
                    group[opt_key] = val
        groups.append(group)
    return groups


def has_adapter_param_groups(optimizer: Optional[torch.optim.Optimizer]) -> bool:
    """Return True if ``optimizer`` uses per-adapter tagged ``param_groups``."""
    if optimizer is None:
        return False
    return any("adapter_name" in group for group in optimizer.param_groups)


@contextmanager
def bind_optimizer_to_adapter(optimizer: torch.optim.Optimizer, adapter_name: str):
    """Temporarily restrict ``optimizer.param_groups`` to ``adapter_name`` only.

    Keeping one shared optimizer with per-adapter groups preserves checkpoint
    compatibility while ensuring ``zero_grad()``, ``step()``, and grad scaling
    only touch ``adapter_name``'s parameters, gradients, and optimizer moments.
    """
    all_groups = optimizer.param_groups
    matching = [group for group in all_groups if group.get("adapter_name") == adapter_name]
    if not matching:
        available = [group.get("adapter_name") for group in all_groups]
        raise KeyError(f"No optimizer param_groups found for adapter {adapter_name!r}; available: {available}")

    optimizer.param_groups = matching
    try:
        yield optimizer
    finally:
        optimizer.param_groups = all_groups


class MultiAdapterLRScheduler:
    """Independent per-adapter LR schedulers backed by a single multi-adapter optimizer.

    Each adapter maintains its own step count and schedule so stepping one adapter
    does not advance or decay another adapter's learning rate.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler_factory: Callable[[torch.optim.Optimizer], Any],
        adapter_names: Optional[Sequence[str]] = None,
    ):
        self.optimizer = optimizer
        if adapter_names is None:
            seen_names: list[str] = []
            for group in optimizer.param_groups:
                name = group.get("adapter_name")
                if name is not None and name not in seen_names:
                    seen_names.append(name)
            adapter_names = seen_names
        if not adapter_names:
            raise ValueError("MultiAdapterLRScheduler requires at least one adapter.")

        self._schedulers: OrderedDict[str, Any] = OrderedDict()
        for name in adapter_names:
            with bind_optimizer_to_adapter(self.optimizer, name):
                self._schedulers[name] = scheduler_factory(self.optimizer)

    @property
    def adapter_names(self) -> list[str]:
        return list(self._schedulers.keys())

    def step(self, adapter_name: Optional[str] = None) -> None:
        """Advance the LR scheduler for ``adapter_name`` (or all adapters if ``None``)."""
        targets = [adapter_name] if adapter_name is not None else list(self._schedulers.keys())
        for name in targets:
            if name not in self._schedulers:
                raise KeyError(f"Unknown adapter {name!r} for MultiAdapterLRScheduler; available: {self.adapter_names}")
            with bind_optimizer_to_adapter(self.optimizer, name):
                self._schedulers[name].step()

    def get_last_lr(self, adapter_name: Optional[str] = None) -> list[float]:
        """Return the last computed learning rate(s) for ``adapter_name`` (or the first adapter)."""
        target = adapter_name if adapter_name is not None else next(iter(self._schedulers))
        if target not in self._schedulers:
            raise KeyError(f"Unknown adapter {target!r} for MultiAdapterLRScheduler; available: {self.adapter_names}")
        return list(self._schedulers[target].get_last_lr())

    def state_dict(self) -> dict[str, Any]:
        return {
            "multi_adapter": True,
            "schedulers": {name: sched.state_dict() for name, sched in self._schedulers.items()},
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        sched_states = state_dict["schedulers"] if state_dict.get("multi_adapter") else state_dict
        for name, sched in self._schedulers.items():
            if name not in sched_states:
                raise KeyError(f"Missing LR scheduler state for adapter {name!r} in checkpoint.")
            with bind_optimizer_to_adapter(self.optimizer, name):
                sched.load_state_dict(sched_states[name])


def split_indices_by_adapter(
    adapter_names: Sequence[str],
    allowed_adapters: Optional[Collection[str]] = None,
) -> OrderedDict[str, list[int]]:
    """Group sample indices by adapter name in deterministic first-seen order."""
    grouped: OrderedDict[str, list[int]] = OrderedDict()
    for idx, name in enumerate(adapter_names):
        if not isinstance(name, str) or not name:
            raise ValueError(f"Expected non-empty string adapter name at index {idx}, got {name!r}")
        if allowed_adapters is not None and name not in allowed_adapters:
            raise KeyError(f"Adapter {name!r} at index {idx} is not in allowed adapters: {sorted(allowed_adapters)}")
        grouped.setdefault(name, []).append(idx)
    return grouped


def _extract_batch_adapter_names(
    batch: Any,
    batch_size: int,
    adapter_key: str,
    default_adapter: Optional[str],
) -> list[str]:
    """Normalize per-sample adapter names from a TensorDict or dict batch."""
    if adapter_key not in batch:
        if default_adapter is None:
            raise KeyError(f"Batch does not contain {adapter_key!r} and no default_adapter was provided.")
        return [default_adapter] * batch_size

    raw = batch[adapter_key]
    if hasattr(raw, "tolist") and not isinstance(raw, (str, bytes)):
        raw = raw.tolist()
    if hasattr(raw, "data") and isinstance(raw.data, (str, list, tuple, np.ndarray)):
        # NonTensorData / NonTensorStack fallback
        raw = raw.data
        if isinstance(raw, np.ndarray):
            raw = raw.tolist()

    if isinstance(raw, str):
        return [raw] * batch_size
    if isinstance(raw, (list, tuple)):
        if len(raw) != batch_size:
            raise ValueError(f"Length of {adapter_key!r} ({len(raw)}) does not match batch size ({batch_size}).")
        return [str(x) for x in raw]
    raise TypeError(f"Unsupported type for batch[{adapter_key!r}]: {type(raw).__name__}")


def _infer_batch_size(batch: Any) -> int:
    if hasattr(batch, "batch_size") and len(batch.batch_size) > 0:
        return int(batch.batch_size[0])
    if isinstance(batch, Mapping):
        for val in batch.values():
            if isinstance(val, (torch.Tensor, np.ndarray)):
                if val.ndim > 0:
                    return int(val.shape[0])
            elif isinstance(val, (list, tuple)):
                return len(val)
    raise ValueError("Unable to infer batch size from batch.")


def _slice_batch(batch: Any, indices: list[int], batch_size: int) -> Any:
    if hasattr(batch, "batch_size") and len(batch.batch_size) > 0:
        return batch[indices]
    if isinstance(batch, Mapping):
        sliced: dict[str, Any] = {}
        for key, val in batch.items():
            if isinstance(val, torch.Tensor) and val.ndim > 0 and val.shape[0] == batch_size:
                sliced[key] = val[indices]
            elif isinstance(val, np.ndarray) and val.ndim > 0 and val.shape[0] == batch_size:
                sliced[key] = val[indices]
            elif isinstance(val, list) and len(val) == batch_size:
                sliced[key] = [val[i] for i in indices]
            elif isinstance(val, tuple) and len(val) == batch_size:
                sliced[key] = tuple(val[i] for i in indices)
            else:
                sliced[key] = val
        return sliced
    raise TypeError(f"Unsupported batch type for adapter splitting: {type(batch).__name__}")


def split_batch_by_adapter(
    batch: Any,
    adapter_key: str = "adapter_name",
    default_adapter: Optional[str] = None,
    allowed_adapters: Optional[Collection[str]] = None,
) -> tuple[OrderedDict[str, Any], OrderedDict[str, list[int]]]:
    """Split a mixed-adapter batch into per-adapter sub-batches and sample index maps.

    Supports both ``TensorDict`` batches and plain Python ``dict`` batches.
    Returns ``(sub_batches, indices_by_adapter)`` keyed by adapter name.
    """
    batch_size = _infer_batch_size(batch)
    if batch_size == 0:
        return OrderedDict(), OrderedDict()

    adapter_names = _extract_batch_adapter_names(batch, batch_size, adapter_key, default_adapter)
    indices_by_adapter = split_indices_by_adapter(adapter_names, allowed_adapters=allowed_adapters)

    sub_batches: OrderedDict[str, Any] = OrderedDict()
    for name, indices in indices_by_adapter.items():
        sub_batches[name] = _slice_batch(batch, indices, batch_size)
    return sub_batches, indices_by_adapter


def resolve_global_adapter_schedule(
    local_adapters: Iterable[str],
    resident_adapters: Optional[Sequence[str]] = None,
    group: Optional[dist.ProcessGroup] = None,
) -> list[str]:
    """Compute a deterministic, cross-rank-identical execution order of active adapters.

    Every rank in ``group`` gathers its local active adapter names so all ranks
    iterate the exact same adapter sequence in lockstep, including ranks that hold
    zero local samples for one of the scheduled adapters.
    """
    local_set = sorted(set(local_adapters))
    resident_list = list(resident_adapters) if resident_adapters is not None else None

    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size(group=group)
        if world_size > 1:
            gathered: list[tuple[list[str], Optional[list[str]]]] = [([], None) for _ in range(world_size)]
            dist.all_gather_object(gathered, (local_set, resident_list), group=group)
            global_set: set[str] = set()
            ref_resident = gathered[0][1]
            for rank_idx, (rank_adapters, rank_resident) in enumerate(gathered):
                global_set.update(rank_adapters)
                if rank_resident != ref_resident:
                    raise ValueError(
                        f"Resident adapter registration order differs across ranks: "
                        f"rank 0 has {ref_resident}, rank {rank_idx} has {rank_resident}."
                    )
        else:
            global_set = set(local_set)
    else:
        global_set = set(local_set)

    if resident_list is not None:
        resident_set = set(resident_list)
        unknown = sorted(global_set - resident_set)
        if unknown:
            raise KeyError(f"Active adapter(s) {unknown} are not registered in resident adapters {resident_list}.")
        return [name for name in resident_list if name in global_set]

    return sorted(global_set)
