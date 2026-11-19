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

__all__ = ["DisaggregationConfig"]

_ALLOWED_BACKENDS = ("nixl", "mooncake", "ascend", "mori", "fake")
_ALLOWED_MOONCAKE_PROTOCOLS = ("nvlink", "local", "rdma", "tcp")


# 【讲解入口 1：配置含义】建议按下面的顺序阅读这次 PD 改动：
# 本文件（配置）→ llm_server.py（分组）→ vllm_pd_replica.py（创建 P/D）
# → vllm_async_server.py（一次请求）→ vllm_rollout.py / utils.py（权重 IPC）。
# P = Prefill：处理整段 prompt，计算 KV；D = Decode：接手后逐 token 生成答案。
# 一个 PD replica 是一整组 P/D，不是一个 P 或一个 D；一台机器也不天然等于一个 replica。
# 例如 2P2D、各 TP4：一组需要 2×4 + 2×4 = 16 个设备/worker。
# 这里的 prefill_replicas/decode_replicas 是组内引擎数，外层 replica 数由 Manager 另算。
@dataclass
class DisaggregationConfig(BaseConfig):
    """Prefill-Decode disaggregation knobs."""

    enabled: bool = False
    prefill_replicas: int = 1
    decode_replicas: int = 1
    # P 的 TP 沿用 rollout.tensor_model_parallel_size；D 的 TP 可独立设置。
    # None 表示 D 跟 P 一样。允许配置不同 TP，不等于任意组合都被底层 Connector 支持。
    decode_tensor_model_parallel_size: Optional[int] = None
    # 显式角色显存比例优先于公共/角色 engine_kwargs 中的同名参数。
    # None 保留原有继承和 engine_kwargs 覆盖规则；实际优先级在 _spawn_pd_server 中落实。
    # 字段名虽包含 gpu，当前这套配置同样用于 NPU 引擎的设备内存比例。
    prefill_gpu_memory_utilization: Optional[float] = None
    decode_gpu_memory_utilization: Optional[float] = None
    prefill_engine_kwargs: dict[str, Any] = field(default_factory=dict)
    decode_engine_kwargs: dict[str, Any] = field(default_factory=dict)
    # transfer_backend 是配置选项，不是最终 Connector 类名：
    # GPU 的 mooncake → MooncakeConnector；当前 Ascend 路径 → MooncakeConnectorV1。
    # 枚举中保留的 ascend/mori/fake 只表示配置层接受，当前 vLLMPDReplica 仍会拒绝它们。
    transfer_backend: str = "nixl"
    bootstrap_port: Optional[int] = None
    ib_device: Optional[str] = None
    mooncake_protocol: str = "nvlink"

    def __post_init__(self) -> None:
        # 关闭 PD 时不让这组可选配置影响普通 rollout；启用后再做范围和冲突校验。
        # 这些检查只验证配置，不会创建 Ray actor、分配设备或验证真实 KV 传输。
        if not self.enabled:
            return
        if self.transfer_backend not in _ALLOWED_BACKENDS:
            raise ValueError(f"disaggregation.transfer_backend={self.transfer_backend!r} not in {_ALLOWED_BACKENDS}")
        if self.prefill_replicas < 1 or self.decode_replicas < 1:
            raise ValueError(
                f"disaggregation requires >=1 prefill and >=1 decode replica "
                f"(got prefill_replicas={self.prefill_replicas}, decode_replicas={self.decode_replicas})"
            )
        if self.decode_tensor_model_parallel_size is not None and self.decode_tensor_model_parallel_size < 1:
            raise ValueError("disaggregation.decode_tensor_model_parallel_size must be positive")
        for role in ("prefill", "decode"):
            utilization = getattr(self, f"{role}_gpu_memory_utilization")
            if utilization is not None and not 0 < utilization <= 1:
                raise ValueError(f"disaggregation.{role}_gpu_memory_utilization must be in (0, 1]")
            overrides = getattr(self, f"{role}_engine_kwargs")
            # 这些参数决定设备拓扑、worker 扩展和 KV Connector，必须由 PD 启动器统一生成。
            # 例如已按 D TP4 分配四个 worker，再偷偷传 pipeline_parallel_size=2 会让两边布局不一致。
            # max_num_seqs、批 token 预算等调优项可以覆盖，拓扑项不能从角色 kwargs 绕过。
            topology_keys = {
                "tensor_parallel_size",
                "pipeline_parallel_size",
                "data_parallel_size",
                "data_parallel_size_local",
                "data_parallel_start_rank",
                "kv_transfer_config",
                "worker_extension_cls",
            }
            # 先把横杠按下划线理解再检查：pipeline-parallel-size 和 pipeline_parallel_size 都要拦截。
            # 这里只用于比较，不改写调用方字典；值为 None 的覆盖项稍后会被启动器忽略。
            conflicting = sorted(
                key for key, value in overrides.items() if value is not None and key.replace("-", "_") in topology_keys
            )
            if conflicting:
                raise ValueError(f"disaggregation.{role}_engine_kwargs cannot override PD topology: {conflicting}")
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
