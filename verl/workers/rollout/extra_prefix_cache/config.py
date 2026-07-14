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

import re
from dataclasses import dataclass
from typing import Any

DEFAULT_NAMESPACE = "verl-extra-prefix"
DEFAULT_MODEL_CACHE_EPOCH = "epoch0"
_SAFE_SALT_RE = re.compile(r"[^A-Za-z0-9_.:-]+")


def normalize_config(config: Any) -> dict[str, Any]:
    if config is None:
        return {}
    if isinstance(config, dict):
        return dict(config)
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(config):
            return OmegaConf.to_container(config, resolve=True) or {}
    except Exception:
        pass
    if hasattr(config, "items"):
        return dict(config.items())
    return {}


def enabled(config: Any) -> bool:
    cfg = normalize_config(config)
    return bool(cfg.get("enable", cfg.get("enabled", False)))


def sanitize_salt(value: str) -> str:
    value = value.replace("@", ":").replace("/", ":").replace("\\", ":").replace("\x00", "")
    value = _SAFE_SALT_RE.sub("_", value).strip("_.:-")
    if not value:
        value = "default"
    if len(value) <= 192:
        return value

    import hashlib

    return f"{value[:160]}:{hashlib.sha256(value.encode('utf-8')).hexdigest()[:24]}"


@dataclass(frozen=True)
class ExtraPrefixCacheConfig:
    enable: bool = False
    backend: str = "lmcache"
    scope: str = "system_prefix"
    prefix_provider: str = "explicit"
    namespace: str = DEFAULT_NAMESPACE
    model_cache_epoch: str = DEFAULT_MODEL_CACHE_EPOCH
    runtime_model_cache_epoch: str | None = None
    advance_epoch_on_weight_update: bool = True
    chunk_size: int = 0
    read_policy: str = "rollout"
    write_policy: str = "warmup_only"
    warmup: bool = True
    validation_mode: str = "off"
    trace_mode: str = "aggregate"
    model_namespace: str | None = None
    tokenizer_fingerprint: str | None = None
    template_fingerprint: str | None = None
    allow_untagged: bool = True
    heuristic_min_prefix_len: int = 0

    @classmethod
    def from_any(cls, config: Any) -> "ExtraPrefixCacheConfig":
        cfg = normalize_config(config)
        return cls(
            enable=enabled(cfg),
            backend=str(cfg.get("backend", "lmcache")),
            scope=str(cfg.get("scope", "system_prefix")),
            prefix_provider=str(cfg.get("prefix_provider", "explicit")),
            namespace=sanitize_salt(str(cfg.get("namespace", DEFAULT_NAMESPACE))),
            model_cache_epoch=str(cfg.get("model_cache_epoch", cfg.get("epoch", DEFAULT_MODEL_CACHE_EPOCH))),
            runtime_model_cache_epoch=(
                str(cfg["runtime_model_cache_epoch"]) if cfg.get("runtime_model_cache_epoch") is not None else None
            ),
            advance_epoch_on_weight_update=bool(cfg.get("advance_epoch_on_weight_update", True)),
            chunk_size=_positive_int(cfg.get("chunk_size", 0)),
            read_policy=str(cfg.get("read_policy", "rollout")),
            write_policy=str(cfg.get("write_policy", "warmup_only")),
            warmup=bool(cfg.get("warmup", True)),
            validation_mode=str(cfg.get("validation_mode", "off")),
            trace_mode=str(cfg.get("trace_mode", "aggregate")),
            model_namespace=str(cfg["model_namespace"]) if cfg.get("model_namespace") is not None else None,
            tokenizer_fingerprint=(
                str(cfg["tokenizer_fingerprint"]) if cfg.get("tokenizer_fingerprint") is not None else None
            ),
            template_fingerprint=(
                str(cfg["template_fingerprint"]) if cfg.get("template_fingerprint") is not None else None
            ),
            allow_untagged=bool(cfg.get("allow_untagged", True)),
            heuristic_min_prefix_len=_positive_int(cfg.get("heuristic_min_prefix_len", 0)),
        )

    @property
    def epoch(self) -> str:
        return self.runtime_model_cache_epoch or self.model_cache_epoch or DEFAULT_MODEL_CACHE_EPOCH


def _positive_int(value: Any, default: int = 0) -> int:
    try:
        parsed = int(value or default)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default
