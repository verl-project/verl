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

from numbers import Integral
from typing import Any


def _require_int_at_least(name: str, value: Any, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"Invalid rollout topology: {name} must be an integer >= {minimum}, got {value!r}.")

    value = int(value)
    if value < minimum:
        raise ValueError(f"Invalid rollout topology: {name} must be >= {minimum}, got {value}.")
    return value


def get_rollout_replica_world_size(config: Any) -> int:
    """Return the accelerator footprint of one rollout replica."""
    tensor_parallel_size = _require_int_at_least(
        "rollout.tensor_model_parallel_size", config.tensor_model_parallel_size, 1
    )
    data_parallel_size = _require_int_at_least("rollout.data_parallel_size", config.data_parallel_size, 1)
    pipeline_parallel_size = _require_int_at_least(
        "rollout.pipeline_model_parallel_size", config.pipeline_model_parallel_size, 1
    )

    disaggregation = getattr(config, "disaggregation", None)
    if disaggregation is None or not getattr(disaggregation, "enabled", False):
        return tensor_parallel_size * data_parallel_size * pipeline_parallel_size

    prefill_replicas = _require_int_at_least(
        "rollout.disaggregation.prefill_replicas", disaggregation.prefill_replicas, 1
    )
    decode_replicas = _require_int_at_least("rollout.disaggregation.decode_replicas", disaggregation.decode_replicas, 1)
    decode_parallel_size = getattr(disaggregation, "decode_tensor_model_parallel_size", None)
    if decode_parallel_size is None:
        decode_parallel_size = tensor_parallel_size
    else:
        decode_parallel_size = _require_int_at_least(
            "rollout.disaggregation.decode_tensor_model_parallel_size", decode_parallel_size, 1
        )

    return (
        (tensor_parallel_size * prefill_replicas + decode_parallel_size * decode_replicas)
        * data_parallel_size
        * pipeline_parallel_size
    )


def get_rollout_num_replicas(config: Any, resource_world_size: int, *, allow_empty: bool = False) -> int:
    """Validate a rollout resource pool and return the number of replicas it can host."""
    replica_world_size = get_rollout_replica_world_size(config)
    resource_world_size = _require_int_at_least("resource_world_size", resource_world_size, 0 if allow_empty else 1)

    if resource_world_size == 0:
        return 0
    gpus_per_node = _require_int_at_least("rollout.n_gpus_per_node", config.n_gpus_per_node, 1)
    if resource_world_size < replica_world_size:
        raise ValueError(
            f"Invalid rollout topology: resource_world_size={resource_world_size} is smaller than "
            f"replica_world_size={replica_world_size}. Adjust the resource pool or rollout parallel sizes."
        )

    num_replicas, remainder = divmod(resource_world_size, replica_world_size)
    if remainder:
        raise ValueError(
            f"Invalid rollout topology: resource_world_size={resource_world_size} must be divisible by "
            f"replica_world_size={replica_world_size} (remainder={remainder}). "
            "Adjust the resource pool or rollout parallel sizes."
        )

    if getattr(config, "name", None) in ("vllm", "sglang"):
        smaller, larger = sorted((replica_world_size, gpus_per_node))
        if larger % smaller:
            raise ValueError(
                f"Invalid rollout topology: replica_world_size={replica_world_size} is not node-aligned with "
                f"rollout.n_gpus_per_node={gpus_per_node}; one must divide the other so replicas do not cross "
                "partial-node boundaries."
            )
    return num_replicas
