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
from collections.abc import Mapping
from typing import Any

from .config import ExtraPrefixCacheConfig, enabled, normalize_config

logger = logging.getLogger(__name__)

_ACTOR_NAME = "verl_extra_prefix_cache_epoch_registry"
_INTERNAL_KV_NAMESPACE = "verl_extra_prefix_cache"
_INTERNAL_KV_PREFIX = "epoch:"


class _ExtraPrefixCacheEpochRegistry:
    def __init__(self) -> None:
        self._epochs: dict[str, str] = {}

    def get(self, namespace: str, default: str) -> str:
        return self._epochs.get(namespace, default)

    def set(self, namespace: str, epoch: str) -> str:
        self._epochs[namespace] = epoch
        return epoch


def build_epoch(global_step: int | str | None) -> str:
    try:
        step = int(global_step or 0)
    except (TypeError, ValueError):
        step = 0
    return f"step-{max(step, 0)}"


def _try_get_ray():
    try:
        import ray
    except Exception:
        return None
    try:
        if not ray.is_initialized():
            return None
    except Exception:
        return None
    return ray


def _internal_kv_key(namespace: str) -> str:
    return f"{_INTERNAL_KV_PREFIX}{namespace}"


def _internal_kv_available() -> bool:
    if _try_get_ray() is None:
        return False
    try:
        from ray.experimental.internal_kv import _internal_kv_initialized

        return bool(_internal_kv_initialized())
    except Exception:
        return False


def _internal_kv_get_epoch(namespace: str) -> str | None:
    if not _internal_kv_available():
        return None
    try:
        from ray.experimental.internal_kv import _internal_kv_get

        value = _internal_kv_get(_internal_kv_key(namespace), namespace=_INTERNAL_KV_NAMESPACE)
    except Exception:
        return None
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _internal_kv_set_epoch(namespace: str, epoch: str) -> bool:
    if not _internal_kv_available():
        return False
    try:
        from ray.experimental.internal_kv import _internal_kv_put

        _internal_kv_put(
            _internal_kv_key(namespace),
            str(epoch),
            overwrite=True,
            namespace=_INTERNAL_KV_NAMESPACE,
        )
        return True
    except Exception:
        return False


def _get_epoch_registry(*, create: bool):
    ray = _try_get_ray()
    if ray is None:
        return None

    try:
        return ray.get_actor(_ACTOR_NAME)
    except Exception:
        if not create:
            return None

    actor_cls = ray.remote(num_cpus=0)(_ExtraPrefixCacheEpochRegistry)
    try:
        return actor_cls.options(name=_ACTOR_NAME, get_if_exists=True).remote()
    except TypeError:
        try:
            return actor_cls.options(name=_ACTOR_NAME).remote()
        except Exception:
            try:
                return ray.get_actor(_ACTOR_NAME)
            except Exception:
                return None
    except Exception:
        try:
            return ray.get_actor(_ACTOR_NAME)
        except Exception:
            return None


def extract_extra_prefix_cache_config(config: Any) -> dict[str, Any]:
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(config):
            extra_cfg = OmegaConf.select(config, "actor_rollout_ref.rollout.extra_prefix_cache")
            if extra_cfg is not None:
                return normalize_config(extra_cfg)
    except Exception:
        pass

    rollout_cfg = None
    try:
        rollout_cfg = config.actor_rollout_ref.rollout
    except Exception:
        pass
    if rollout_cfg is None and isinstance(config, Mapping):
        rollout_cfg = (config.get("actor_rollout_ref") or {}).get("rollout")
    if rollout_cfg is None:
        return normalize_config(config)
    try:
        return normalize_config(rollout_cfg.get("extra_prefix_cache", None))
    except Exception:
        return normalize_config(getattr(rollout_cfg, "extra_prefix_cache", None))


def maybe_advance_extra_prefix_cache_epoch(
    config: Any,
    global_step: int | str | None,
    *,
    log: logging.Logger | None = None,
) -> str | None:
    run_logger = log or logger
    cfg = ExtraPrefixCacheConfig.from_any(extract_extra_prefix_cache_config(config))
    if not enabled(cfg.__dict__):
        return None
    if not cfg.advance_epoch_on_weight_update:
        run_logger.info("ExtraPrefixCache epoch advance skipped reason=policy_off namespace=%s", cfg.namespace)
        return None

    epoch = build_epoch(global_step)
    if _internal_kv_set_epoch(cfg.namespace, epoch):
        run_logger.info(
            "ExtraPrefixCache epoch advanced namespace=%s epoch=%s global_step=%s backend=ray_internal_kv",
            cfg.namespace,
            epoch,
            global_step,
        )
        return epoch

    actor = _get_epoch_registry(create=True)
    ray = _try_get_ray()
    if actor is None or ray is None:
        run_logger.warning(
            "ExtraPrefixCache epoch advance skipped namespace=%s epoch=%s reason=registry_unavailable",
            cfg.namespace,
            epoch,
        )
        return None
    try:
        epoch = ray.get(actor.set.remote(cfg.namespace, epoch))
    except Exception as exc:
        run_logger.warning(
            "ExtraPrefixCache epoch advance failed namespace=%s global_step=%s error=%s",
            cfg.namespace,
            global_step,
            exc,
        )
        return None
    run_logger.info(
        "ExtraPrefixCache epoch advanced namespace=%s epoch=%s global_step=%s backend=ray_actor",
        cfg.namespace,
        epoch,
        global_step,
    )
    return str(epoch)


async def resolve_runtime_model_cache_epoch(config: Any, *, log: logging.Logger | None = None) -> str:
    cfg = ExtraPrefixCacheConfig.from_any(config)
    if not cfg.enable or not cfg.advance_epoch_on_weight_update:
        return cfg.epoch

    internal_epoch = _internal_kv_get_epoch(cfg.namespace)
    if internal_epoch:
        return internal_epoch

    actor = _get_epoch_registry(create=False)
    if actor is None:
        return cfg.epoch
    try:
        return str(await actor.get.remote(cfg.namespace, cfg.epoch))
    except Exception as exc:
        run_logger = log or logger
        run_logger.warning(
            "ExtraPrefixCache epoch resolve failed namespace=%s default_epoch=%s error=%s",
            cfg.namespace,
            cfg.epoch,
            exc,
        )
        return cfg.epoch
