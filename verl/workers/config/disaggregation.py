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
from dataclasses import dataclass, field
from typing import Any, Optional

from verl.base_config import BaseConfig

__all__ = ["DisaggregationConfig", "RoutingPolicyConfig"]

_ALLOWED_BACKENDS = ("nixl", "mooncake", "ascend", "mori", "fake")
_ALLOWED_MOONCAKE_PROTOCOLS = ("nvlink", "local", "rdma", "tcp")
_ALLOWED_ROUTING_POLICIES = (
    "random",
    "round_robin",
    "cache_aware",
    "power_of_two",
    "consistent_hash",
    "rendezvous_hash",
)


@dataclass
class RoutingPolicyConfig(BaseConfig):
    """Decode routing settings aligned with vLLM Router policy names."""

    type: str = "round_robin"
    load_check_interval_secs: int = 5
    virtual_nodes: int = 160
    cache_threshold: float = 0.3
    balance_abs_threshold: int = 64
    balance_rel_threshold: float = 1.5
    eviction_interval_secs: int = 120
    max_tree_size: int = 2**26

    def __post_init__(self) -> None:
        if self.type not in _ALLOWED_ROUTING_POLICIES:
            raise ValueError(f"routing policy type={self.type!r} not in {_ALLOWED_ROUTING_POLICIES}")
        if self.load_check_interval_secs < 0:
            raise ValueError("load_check_interval_secs must be non-negative")
        if self.virtual_nodes < 1:
            raise ValueError("virtual_nodes must be positive")
        if not 0 <= self.cache_threshold <= 1:
            raise ValueError("cache_threshold must be in [0, 1]")
        if self.balance_abs_threshold < 0:
            raise ValueError("balance_abs_threshold must be non-negative")
        if self.balance_rel_threshold <= 0:
            raise ValueError("balance_rel_threshold must be positive")
        if self.eviction_interval_secs < 0:
            raise ValueError("eviction_interval_secs must be non-negative")
        if self.max_tree_size < 0:
            raise ValueError("max_tree_size must be non-negative")


@dataclass
class DisaggregationConfig(BaseConfig):
    """Prefill-Decode disaggregation knobs."""

    enabled: bool = False
    prefill_replicas: int = 1
    decode_replicas: int = 1
    decode_tensor_model_parallel_size: Optional[int] = None
    prefill_gpu_memory_utilization: Optional[float] = None
    decode_gpu_memory_utilization: Optional[float] = None
    prefill_engine_kwargs: dict[str, Any] = field(default_factory=dict)
    decode_engine_kwargs: dict[str, Any] = field(default_factory=dict)
    transfer_backend: str = "nixl"
    bootstrap_port: Optional[int] = None
    ib_device: Optional[str] = None
    mooncake_protocol: str = "nvlink"
    decode_policy: RoutingPolicyConfig = field(default_factory=RoutingPolicyConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.decode_policy, RoutingPolicyConfig):
            object.__setattr__(self, "decode_policy", RoutingPolicyConfig(**dict(self.decode_policy)))
        if not self.enabled:
            return
        if self.transfer_backend not in _ALLOWED_BACKENDS:
            raise ValueError(f"disaggregation.transfer_backend={self.transfer_backend!r} not in {_ALLOWED_BACKENDS}")
        if self.prefill_replicas < 1 or self.decode_replicas < 1:
            raise ValueError(
                f"disaggregation requires >=1 prefill and >=1 decode replica "
                f"(got prefill_replicas={self.prefill_replicas}, decode_replicas={self.decode_replicas})"
            )
        for role, value in (
            ("prefill", self.prefill_gpu_memory_utilization),
            ("decode", self.decode_gpu_memory_utilization),
        ):
            if value is not None and not 0 < value <= 1:
                raise ValueError(f"{role}_gpu_memory_utilization must be in (0, 1], got {value}")
        for role, engine_kwargs in (
            ("prefill", self.prefill_engine_kwargs),
            ("decode", self.decode_engine_kwargs),
        ):
            if not isinstance(engine_kwargs, dict):
                raise TypeError(f"{role}_engine_kwargs must be a dict, got {type(engine_kwargs).__name__}")
            unsupported_fields = set(engine_kwargs) - {"max_num_batched_tokens", "max_num_seqs"}
            if unsupported_fields:
                raise ValueError(
                    f"unsupported {role}_engine_kwargs fields: {sorted(unsupported_fields)}; "
                    "only max_num_batched_tokens and max_num_seqs are supported"
                )
            for field_name in ("max_num_batched_tokens", "max_num_seqs"):
                value = engine_kwargs.get(field_name)
                if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 1):
                    raise ValueError(f"{role}_engine_kwargs.{field_name} must be a positive integer, got {value!r}")
            max_num_batched_tokens = engine_kwargs.get("max_num_batched_tokens")
            max_num_seqs = engine_kwargs.get("max_num_seqs")
            if (
                max_num_batched_tokens is not None
                and max_num_seqs is not None
                and max_num_batched_tokens < max_num_seqs
            ):
                raise ValueError(
                    f"{role}_engine_kwargs.max_num_batched_tokens must be >= max_num_seqs "
                    f"(got {max_num_batched_tokens} < {max_num_seqs})"
                )
        if self.bootstrap_port is not None and not (0 < self.bootstrap_port < 65536):
            raise ValueError(f"bootstrap_port out of range: {self.bootstrap_port}")
        if self.transfer_backend == "mooncake" and self.mooncake_protocol not in _ALLOWED_MOONCAKE_PROTOCOLS:
            raise ValueError(
                f"disaggregation.mooncake_protocol={self.mooncake_protocol!r} not in {_ALLOWED_MOONCAKE_PROTOCOLS}"
            )

    def effective_decode_tp(self, prefill_tp: int) -> int:
        """Resolve decode TP (defaults to ``prefill_tp``). Test-only helper; runtime paths
        must inline this because OmegaConf/Ray serialization drops dataclass methods."""
        if self.decode_tensor_model_parallel_size is not None:
            return self.decode_tensor_model_parallel_size
        return prefill_tp
