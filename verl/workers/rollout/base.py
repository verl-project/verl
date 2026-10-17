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

import importlib
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Generator, Optional

import torch
from torch.distributed.device_mesh import DeviceMesh

from verl import DataProto
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import HFModelConfig, RolloutConfig

__all__ = ["BaseRollout", "RolloutLayout"]


@dataclass
class RolloutLayout:
    """The parameters one inference worker loads, reported by its server adapter so
    ownership-aware checkpoint engines (``sharded_nixl``) send it only those. Plain Python
    fields: it crosses Ray calls as a dict.

    * ``tp_*``, ``pp_*``, ``ep_*``: the worker's place in the inference parallel layout.
    * ``layer_range``: ``(start, end)`` decoder layers of this pipeline stage (``None`` = all).
    * ``local_experts``: ``{layer_idx: [global expert ids]}`` routed experts on this worker.
    * ``wire_formats``: what ``receive_weights`` accepts: ``named_tensors`` (full HF tensors) and
      optionally ``rank_local_named_tensors`` (each tensor arrives as its ``slices`` box).
    * ``slices``: ``{hf_name: {"full_shape": [...], "box": [[start, len], ...]}}``; names not
      listed arrive full.
    * ``slice_digest``: hash of ``slices`` (empty = computed by the engine).
    """

    tp_size: int = 1
    tp_rank: int = 0
    pp_size: int = 1
    pp_rank: int = 0
    ep_size: int = 1
    ep_rank: int = 0
    layer_range: Optional[tuple[int, int]] = None
    local_experts: dict[int, list[int]] = field(default_factory=dict)
    wire_formats: list[str] = field(default_factory=lambda: ["named_tensors"])
    slices: Optional[dict[str, dict]] = None
    slice_digest: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> Optional["RolloutLayout"]:
        if d is None:
            return None
        d = dict(d)
        if d.get("layer_range") is not None:
            d["layer_range"] = tuple(d["layer_range"])
        d["local_experts"] = {int(k): list(v) for k, v in (d.get("local_experts") or {}).items()}
        if d.get("wire_formats") is not None:
            d["wire_formats"] = list(d["wire_formats"])
        else:
            d.pop("wire_formats", None)
        return cls(**d)


class BaseRollout(ABC):
    """Base class for rollout."""

    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        device_mesh: DeviceMesh,
        *args,
        **kwargs,
    ):
        self.config = omega_conf_to_dataclass(config)
        self.model_config: HFModelConfig = omega_conf_to_dataclass(model_config)
        self.device_mesh = device_mesh

    @abstractmethod
    async def resume(self, tags: list[str]):
        """Resume rollout weights or kv cache in GPU memory.

        Args:
            tags: weights or kv_cache.
        """
        pass

    @abstractmethod
    async def update_weights(
        self,
        weights: Generator[tuple[str, torch.Tensor], None, None],
        wire_format: str = "named_tensors",
        **kwargs,
    ):
        """Update the weights of the rollout model.

        Args:
            weights: A generator yielding ``(name, tensor)`` pairs for
                ``named_tensors``, or ``(named_tensors, is_last)`` flushes for
                ``delta_flush``.
            wire_format: ``named_tensors`` for full tensors (the default), or
                ``delta_flush`` for delta-engine flushes. Adapters must branch on
                this value instead of treating a delta flush as named weights.
        """
        pass

    @abstractmethod
    async def release(self):
        """Release weights and kv cache in GPU memory."""
        pass

    async def get_weight_ownership(self) -> Optional[RolloutLayout]:
        """This worker's :class:`RolloutLayout`; ``None`` means it loads every parameter."""
        return None

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Batch generate sequences in sync mode.

        Args:
            prompts: The input prompts.

        Returns:
            The output sequences.
        """
        raise NotImplementedError


_ROLLOUT_REGISTRY = {
    ("vllm", "async"): "verl.workers.rollout.vllm_rollout.ServerAdapter",
    ("sglang", "async"): "verl.workers.rollout.sglang_rollout.sglang_rollout.ServerAdapter",
    ("trtllm", "async"): "verl.workers.rollout.trtllm_rollout.trtllm_rollout.ServerAdapter",
}


def get_rollout_class(rollout_name: str, mode: str = "async") -> type[BaseRollout]:
    """Get the rollout class by name.

    Args:
        rollout_name: The name of the rollout.
        mode: The mode of the rollout, async: server mode.

    Returns:
        The rollout class.
    """
    assert (rollout_name, mode) in _ROLLOUT_REGISTRY, f"Rollout {rollout_name} with mode {mode} not found"
    fqdn = _ROLLOUT_REGISTRY[(rollout_name, mode)]
    module_name, class_name = fqdn.rsplit(".", 1)
    rollout_module = importlib.import_module(module_name)
    return getattr(rollout_module, class_name)
