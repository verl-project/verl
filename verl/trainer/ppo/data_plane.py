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
"""Data-plane strategies used by the V0 PPO trainer."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any

import torch

from verl import DataProto
from verl.utils import tensordict_utils as tu
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding


@dataclass
class PreparedBatch:
    """Payload sent to a worker plus context needed to collect its output."""

    payload: Any
    context: Any = None


class PPODataPlane:
    """Strategy boundary between PPO control flow and a batch representation."""

    name = "classic"
    data_proto_cls: type[DataProto] = DataProto

    def __init__(self, *, strict_mode: bool = False):
        self.strict_mode = strict_mode

    def setup(self) -> None:
        """Initialize process-local data-plane resources."""

    def prepare_inference(self, batch: DataProto, metadata: dict[str, Any]) -> PreparedBatch:
        batch_td = left_right_2_no_padding(batch.to_tensordict())
        tu.assign_non_tensor(batch_td, **metadata)
        return PreparedBatch(payload=batch_td, context=batch_td)

    def collect_inference(
        self,
        output: Any,
        prepared: PreparedBatch,
        key_map: dict[str, str],
        *,
        restore_keys: tuple[str, ...] = (),
        fp32_keys: tuple[str, ...] = (),
    ) -> DataProto:
        result: dict[str, Any] = {}
        output_keys = set(output.keys())
        for source_key, target_key in key_map.items():
            if source_key not in output_keys:
                continue
            value = tu.get(output, source_key)
            if value is None:
                continue
            if source_key in restore_keys and torch.is_tensor(value):
                value = no_padding_2_padding(value, prepared.context)
            if source_key in fp32_keys and torch.is_tensor(value):
                value = value.float()
            result[target_key] = value
        return self.data_proto_cls.from_tensordict(tu.get_tensordict(result))

    def prepare_training(self, batch: DataProto, metadata: dict[str, Any]) -> PreparedBatch:
        return self.prepare_inference(batch, metadata)

    def collect_metrics(self, output: Any) -> dict[str, Any]:
        return tu.get(output, "metrics")

    def prefetch(self, batch: DataProto, keys: list[str]) -> None:
        batch.prefetch(keys)

    def reset_materialize_stats(self) -> None:
        self.data_proto_cls.reset_materialize_stats()

    def pop_materialize_stats(self) -> dict[str, float]:
        return self.data_proto_cls.pop_materialize_stats()


class ClassicPPODataPlane(PPODataPlane):
    """Classic TensorDict-backed DataProto strategy."""


_DATA_PLANE_CLASSES = {
    "classic": "verl.trainer.ppo.data_plane.ClassicPPODataPlane",
    "neoproto": "verl.experimental.neoproto.data_plane.NeoPPODataPlane",
}
_DEFAULT_DATA_PLANE = "classic"


def _load_data_plane_class(name: str) -> type[PPODataPlane]:
    try:
        class_path = _DATA_PLANE_CLASSES[name]
    except KeyError as exc:
        raise ValueError(f"Unknown trainer.data_plane {name!r}; expected one of {sorted(_DATA_PLANE_CLASSES)}") from exc
    module_name, class_name = class_path.rsplit(".", 1)
    return getattr(import_module(module_name), class_name)


def _resolve_data_plane_name(trainer_config) -> str:
    """Resolver for deciding which data plane a trainer should use.
    ==============  ================  ==========================================
    ``data_plane``  ``use_neoproto``  resolved name
    ==============  ================  ==========================================
    unset           unset / false     ``classic``
    unset           true              ``neoproto``  (legacy upgrade)
    ``classic``     unset / false     ``classic``
    ``classic``     true              ``neoproto``  (legacy upgrade)
    ``neoproto``    any               ``neoproto``
    anything else   any               returned as-is, rejected by the registry
    ==============  ================  ==========================================
    """
    name = trainer_config.get("data_plane", None)
    name = _DEFAULT_DATA_PLANE if name is None else str(name)
    if name == _DEFAULT_DATA_PLANE and bool(trainer_config.get("use_neoproto", False)):
        return "neoproto"
    return name


def resolve_data_proto_cls(config) -> type[DataProto]:
    """Resolve the configured batch container without building a data plane.

    Ray actors outside the trainer must produce batches of the same container
    type, but must not run the trainer's data-plane setup.
    """
    return _load_data_plane_class(_resolve_data_plane_name(config.trainer)).data_proto_cls


def build_data_plane(config) -> PPODataPlane:
    """Build and initialize the single data-plane strategy (classic Dataproto or NeoProto) for a trainer."""
    trainer_config = config.trainer
    data_plane_cls = _load_data_plane_class(_resolve_data_plane_name(trainer_config))
    data_plane = data_plane_cls(strict_mode=bool(trainer_config.get("neoproto_strict_mode", False)))
    data_plane.setup()
    return data_plane
