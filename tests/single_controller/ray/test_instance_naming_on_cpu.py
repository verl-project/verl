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
"""Instance naming primitives plus the config plumbing that feeds them."""

import pytest
from omegaconf import OmegaConf

from verl.single_controller.ray.base import ResourcePoolManager
from verl.single_controller.ray.instance_naming import (
    InstanceNameBuilder,
    build_vllm_server_prefix,
    build_weight_transfer_socket,
    get_instance_id,
    inject_instance_id,
)


def test_default_names_are_backward_compatible():
    names = InstanceNameBuilder()
    assert names.actor("reward_loop_worker_0") == "reward_loop_worker_0"
    assert names.placement_group("global_poolverl_group_0") == "global_poolverl_group_0"


def test_instance_names_are_isolated_and_stable():
    policy_a = InstanceNameBuilder("policy_1")
    policy_b = InstanceNameBuilder("policy_2")

    assert policy_a.actor("reward_loop_worker_0") != policy_b.actor("reward_loop_worker_0")
    assert policy_a.placement_group("group_0") != policy_b.placement_group("group_0")
    # A builder is deterministic: same logical name, same result.
    assert policy_a.actor("reward_loop_worker_0") == policy_a.actor("reward_loop_worker_0")


def test_invalid_instance_id_is_rejected():
    with pytest.raises(ValueError, match="instance_id"):
        InstanceNameBuilder("bad/id")


def test_long_names_use_stable_hash_suffix():
    names = InstanceNameBuilder("p" * 80, max_name_length=64)

    value = names.actor("actor")

    assert len(value) == 64
    # Deterministic for the same logical name.
    assert value == names.actor("actor")
    # Both names truncate to the same "p"-prefix, so only the digest can tell them apart.
    assert value != names.actor("actor2")


def test_vllm_prefix_applies_the_same_length_guard():
    long_id = "p" * 300

    prefix = build_vllm_server_prefix(long_id)

    assert prefix == InstanceNameBuilder(long_id).actor("vllm_")
    assert len(prefix) <= 255
    assert prefix != build_vllm_server_prefix("p" * 299 + "q")


def test_vllm_prefix_and_weight_socket_share_instance_id():
    assert build_vllm_server_prefix("policy_1") == "policy_1__vllm_"
    assert build_weight_transfer_socket("job", "policy_1", 0, 1) == (
        "ipc:///tmp/rl-colocate-zmq-job-policy_1-replica-0-rank-1.sock"
    )


def test_instance_id_can_be_read_from_nested_or_rollout_config():
    assert get_instance_id({"instance": {"id": "policy_1"}}) == "policy_1"
    assert get_instance_id({"actor_rollout_ref": {"rollout": {"instance_id": "policy_2"}}}) == "policy_2"


def test_instance_id_is_read_from_a_dataclass_config():
    """RolloutReplica reads the id from a dataclass, not a DictConfig."""
    from dataclasses import replace

    from verl.workers.config import RolloutConfig

    base = RolloutConfig(name="vllm")

    assert get_instance_id(base) is None
    assert get_instance_id(replace(base, instance_id="policy_a")) == "policy_a"


def _struct_config(payload: dict):
    """A config that rejects new keys unless the writer uses ``open_dict``."""
    return OmegaConf.create(payload, flags={"struct": True})


def test_instance_id_is_injected_into_both_rollout_configs():
    config = _struct_config(
        {
            "instance": {"id": "policy_a"},
            "actor_rollout_ref": {"rollout": {"name": "vllm"}},
            "reward": {"reward_model": {"rollout": {"name": "vllm"}}},
        }
    )

    assert inject_instance_id(config) == "policy_a"
    assert config.actor_rollout_ref.rollout.instance_id == "policy_a"
    assert config.reward.reward_model.rollout.instance_id == "policy_a"


def test_rollout_level_instance_id_is_honoured_when_top_level_is_unset():
    config = _struct_config(
        {
            "actor_rollout_ref": {"rollout": {"instance_id": "policy_x"}},
            "reward": {"reward_model": {"rollout": {}}},
        }
    )

    assert inject_instance_id(config) == "policy_x"
    assert config.reward.reward_model.rollout.instance_id == "policy_x"


def test_injection_tolerates_absent_reward_model():
    config = _struct_config({"actor_rollout_ref": {"rollout": {"name": "vllm"}}, "reward": {}})

    assert inject_instance_id(config) is None
    assert config.actor_rollout_ref.rollout.instance_id is None


def test_resource_pool_manager_propagates_instance_builder():
    manager = ResourcePoolManager(
        resource_pool_spec={}, mapping={}, instance_name_builder=InstanceNameBuilder("policy_1")
    )
    assert manager.instance_name_builder.actor("global_pool") == "policy_1__global_pool"
