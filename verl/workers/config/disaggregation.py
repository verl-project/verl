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
from dataclasses import dataclass
from typing import Optional

from verl.base_config import BaseConfig

__all__ = ["DisaggregationConfig"]

_ALLOWED_BACKENDS = ("nixl", "mooncake", "ascend", "mori", "fake")
_ALLOWED_MOONCAKE_PROTOCOLS = ("nvlink", "local", "rdma", "tcp")
_ALLOWED_MORI_BACKENDS = ("xgmi", "rdma")
_ALLOWED_MORI_MODES = ("read", "write")


@dataclass
class DisaggregationConfig(BaseConfig):
    """Prefill-Decode disaggregation knobs."""

    enabled: bool = False
    prefill_replicas: int = 1
    decode_replicas: int = 1
    decode_tensor_model_parallel_size: Optional[int] = None
    transfer_backend: str = "nixl"
    bootstrap_port: Optional[int] = None
    ib_device: Optional[str] = None
    mooncake_protocol: str = "nvlink"
    # MoRI-IO transport (vLLM MoRIIOConnector, ROCm): "xgmi" for single-node, "rdma" across nodes.
    mori_backend: str = "xgmi"
    # MoRI-IO transfer mode. "read": prefill finishes first, then the decode engine pulls the KV in one go.
    # "write": both legs start together and the prefill engine pushes KV layer by layer; decode steps are
    # faster, but pushes are serialized per TP rank and requests queue up for KV under load.
    mori_mode: str = "read"
    # Conditional disaggregation (MoRI-IO path): a turn that adds fewer than this many tokens to what its
    # session's decode engine already holds (previous prompt + response) skips the prefill engine and is
    # prefilled by the decode engine itself. Needs a session id per request. 0 = always disaggregate.
    # Multi-turn rollouts add little per turn, and sending every turn through the prefill engine makes it
    # keep (or recompute) every session's whole context.
    # Requires mori_mode="write": in READ mode vLLM's MoRI connector treats every prompt on a decode engine
    # as remotely prefilled, so a request sent without kv_transfer_params is decoded on missing KV (wrong
    # output, no error).
    min_remote_prefill_tokens: int = 0

    def __post_init__(self) -> None:
        if not self.enabled:
            return
        if self.transfer_backend not in _ALLOWED_BACKENDS:
            raise ValueError(f"disaggregation.transfer_backend={self.transfer_backend!r} not in {_ALLOWED_BACKENDS}")
        if self.prefill_replicas < 1 or self.decode_replicas < 1:
            raise ValueError(
                f"disaggregation requires >=1 prefill and >=1 decode replica "
                f"(got prefill_replicas={self.prefill_replicas}, decode_replicas={self.decode_replicas})"
            )
        if self.bootstrap_port is not None and not (0 < self.bootstrap_port < 65536):
            raise ValueError(f"bootstrap_port out of range: {self.bootstrap_port}")
        if self.transfer_backend == "mooncake" and self.mooncake_protocol not in _ALLOWED_MOONCAKE_PROTOCOLS:
            raise ValueError(
                f"disaggregation.mooncake_protocol={self.mooncake_protocol!r} not in {_ALLOWED_MOONCAKE_PROTOCOLS}"
            )
        if self.transfer_backend == "mori" and self.mori_backend not in _ALLOWED_MORI_BACKENDS:
            raise ValueError(f"disaggregation.mori_backend={self.mori_backend!r} not in {_ALLOWED_MORI_BACKENDS}")
        if self.transfer_backend == "mori" and self.mori_mode not in _ALLOWED_MORI_MODES:
            raise ValueError(f"disaggregation.mori_mode={self.mori_mode!r} not in {_ALLOWED_MORI_MODES}")
        if self.min_remote_prefill_tokens < 0:
            raise ValueError(f"min_remote_prefill_tokens must be >= 0 (got {self.min_remote_prefill_tokens})")
        if self.min_remote_prefill_tokens > 0 and (self.transfer_backend != "mori" or self.mori_mode != "write"):
            raise ValueError(
                "min_remote_prefill_tokens > 0 requires transfer_backend=mori and mori_mode=write "
                f"(got transfer_backend={self.transfer_backend!r}, mori_mode={self.mori_mode!r}): READ-mode "
                "decode engines cannot serve requests that skip the prefill engine"
            )

    def effective_decode_tp(self, prefill_tp: int) -> int:
        """Resolve decode TP (defaults to ``prefill_tp``). Test-only helper; runtime paths
        must inline this because OmegaConf/Ray serialization drops dataclass methods."""
        if self.decode_tensor_model_parallel_size is not None:
            return self.decode_tensor_model_parallel_size
        return prefill_tp
