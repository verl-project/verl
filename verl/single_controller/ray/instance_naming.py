# Copyright 2025 Individual Contributor: Zehan Zhu
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
"""Stable names for resources owned by one VERL runtime instance."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass

from omegaconf import open_dict

_VALID_INSTANCE_ID = re.compile(r"^[A-Za-z0-9_-]+$")
_INSTANCE_SEPARATOR = "__"


@dataclass(frozen=True)
class InstanceNameBuilder:
    """Build names that are isolated by an optional instance identifier.

    An unset identifier deliberately preserves VERL's existing names.  The
    builder is immutable so it can safely be shared by resource managers and
    copied into Ray actor construction arguments.
    """

    instance_id: str | None = None
    max_name_length: int = 255

    def __post_init__(self) -> None:
        if self.max_name_length < 16:
            raise ValueError("max_name_length must be at least 16")
        if self.instance_id is not None:
            if not isinstance(self.instance_id, str) or not self.instance_id.strip():
                raise ValueError("instance_id must be a non-empty string")
            if not _VALID_INSTANCE_ID.fullmatch(self.instance_id):
                raise ValueError("instance_id must contain only ASCII letters, digits, '_' or '-'")

    def actor(self, logical_name: str) -> str:
        """Return the instance-scoped name for an actor resource.

        The result is a complete actor name for resources created with
        ``.options(name=...)``, but only a prefix for worker groups built by
        ``RayWorkerGroup``, which appends the class name and placement index.
        """
        return self._build(logical_name)

    def placement_group(self, logical_name: str) -> str:
        """Return the instance-scoped prefix for a placement group name.

        Callers append their own group index to the result.
        """
        return self._build(logical_name)

    def _digest(self, value: str) -> str:
        """Return the stable suffix used when a generated name is too long."""
        return hashlib.sha1(value.encode("utf-8")).hexdigest()[:10]

    def _build(self, logical_name: str) -> str:
        if not logical_name:
            raise ValueError("logical_name must be a non-empty string")
        scoped = logical_name if self.instance_id is None else f"{self.instance_id}{_INSTANCE_SEPARATOR}{logical_name}"
        if len(scoped) <= self.max_name_length:
            return scoped
        digest = self._digest(scoped)
        return f"{scoped[: self.max_name_length - len(digest) - 1]}_{digest}"


def get_instance_id(config: object) -> str | None:
    """Read the instance identifier, preferring the top-level keys.

    Both Hydra ``DictConfig`` and VERL's ``BaseConfig`` implement ``Mapping``, so
    one lookup path covers the trainer config and the per-component configs.
    """
    if not isinstance(config, Mapping):
        return None

    instance = config.get("instance")
    if isinstance(instance, Mapping) and instance.get("id") is not None:
        return str(instance["id"])

    if config.get("instance_id") is not None:
        return str(config["instance_id"])

    rollout_root = config.get("actor_rollout_ref")
    rollout = rollout_root.get("rollout") if isinstance(rollout_root, Mapping) else None
    if isinstance(rollout, Mapping) and rollout.get("instance_id") is not None:
        return str(rollout["instance_id"])

    return None


def inject_instance_id(config: object) -> str | None:
    """Write the instance id into the rollout configs downstream components read.

    The trainer sees the top-level config while rollout components only receive
    their own ``RolloutConfig``, so the value has to be copied across before they
    are constructed. Writing ``None`` when no identifier is configured is
    deliberate: it clears any value left over from a previous composition.
    """
    instance_id = get_instance_id(config)
    with open_dict(config.actor_rollout_ref.rollout):
        config.actor_rollout_ref.rollout.instance_id = instance_id
    reward_model = config.reward.get("reward_model")
    reward_rollout = reward_model.get("rollout") if reward_model is not None else None
    if reward_rollout is not None:
        with open_dict(reward_rollout):
            reward_rollout.instance_id = instance_id
    return instance_id


def build_vllm_server_prefix(instance_id: str | None, base: str = "vllm_") -> str:
    """Return the server prefix shared by vLLM creators and clients.

    Delegates to the builder so an over-long instance id is shortened here the
    same way it is for every other generated name.
    """
    return InstanceNameBuilder(instance_id).actor(base)


def build_weight_transfer_socket(
    ray_job_id: str,
    instance_id: str | None,
    replica_rank: str | int,
    trainer_rank: str | int,
) -> str:
    """Build the sender/receiver IPC endpoint for weight synchronization."""
    instance_suffix = f"-{instance_id}" if instance_id else ""
    path = f"/tmp/rl-colocate-zmq-{ray_job_id}{instance_suffix}-replica-{replica_rank}-rank-{trainer_rank}.sock"
    # sockaddr_un has a platform-dependent fixed capacity. Keep the path at
    # most 103 bytes (excluding the NUL), also safe on systems with sun_path[104].
    # Hash the entire identity, including ranks, so shortening never drops them.
    if len(path.encode("utf-8")) > 103:
        digest = hashlib.sha256(path.encode("utf-8")).hexdigest()
        path = f"/tmp/rl-colocate-zmq-{digest}.sock"
    return f"ipc://{path}"
