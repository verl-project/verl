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

import importlib
import logging
from collections.abc import Mapping, Sequence
from typing import Any

from verl.workers.rollout.extra_prefix_cache import maybe_advance_extra_prefix_cache_epoch

logger = logging.getLogger(__name__)


def _select(config: Any, key: str, default: Any = None) -> Any:
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(config):
            value = OmegaConf.select(config, key)
            return default if value is None else value
    except Exception:
        pass

    value = config
    for part in key.split("."):
        if value is None:
            return default
        if isinstance(value, Mapping):
            value = value.get(part, default)
        else:
            value = getattr(value, part, default)
    return value


def _to_container(value: Any) -> Any:
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(value):
            return OmegaConf.to_container(value, resolve=True)
    except Exception:
        pass
    return value


def _as_hook_specs(value: Any) -> list[Any]:
    value = _to_container(value)
    if value is None:
        return []
    if isinstance(value, str) or isinstance(value, Mapping):
        return [value]
    if isinstance(value, Sequence):
        return list(value)
    return [value]


def _load_callable(path: str):
    if ":" in path:
        module_name, attr_name = path.split(":", 1)
    else:
        module_name, _, attr_name = path.rpartition(".")
    if not module_name or not attr_name:
        raise ValueError(f"Hook path must be 'module.attr' or 'module:attr', got {path!r}")
    module = importlib.import_module(module_name)
    hook = getattr(module, attr_name)
    if not callable(hook):
        raise TypeError(f"Weight update hook {path!r} is not callable")
    return hook


def _parse_hook_spec(spec: Any) -> tuple[str | None, dict[str, Any], bool]:
    spec = _to_container(spec)
    if isinstance(spec, str):
        return spec, {}, False
    if not isinstance(spec, Mapping):
        return None, {}, False
    required = bool(spec.get("required", False))
    enabled = bool(spec.get("enable", spec.get("enabled", True)))
    if not enabled:
        return None, {}, required
    path = spec.get("path") or spec.get("callable") or spec.get("target") or spec.get("_target_")
    kwargs = spec.get("kwargs") or {}
    if not isinstance(kwargs, Mapping):
        raise TypeError(f"Weight update hook kwargs must be a mapping, got {type(kwargs).__name__}")
    return path, dict(kwargs), required


def maybe_run_weight_update_hooks(
    config: Any,
    global_step: int | str | None,
    *,
    log: logging.Logger | None = None,
) -> list[Any]:
    """Run optional trainer weight-update hooks.

    This is intentionally backend-agnostic: verl never imports downstream
    projects directly. Integrations can opt in by setting
    ``trainer.weight_update_hooks`` to a callable path or a list of hook specs.
    """

    run_logger = log or logger
    results: list[Any] = []
    epc_epoch = maybe_advance_extra_prefix_cache_epoch(config, global_step, log=run_logger)
    if epc_epoch is not None:
        results.append(epc_epoch)
    hooks = _as_hook_specs(_select(config, "trainer.weight_update_hooks"))
    for spec in hooks:
        try:
            path, kwargs, required = _parse_hook_spec(spec)
            if not path:
                continue
            hook = _load_callable(str(path))
            results.append(hook(config, global_step, log=run_logger, **kwargs))
        except Exception as exc:
            message = f"Weight update hook failed spec={spec} error={exc}"
            if isinstance(spec, Mapping) and bool(spec.get("required", False)):
                raise RuntimeError(message) from exc
            run_logger.warning(message)
    return results
