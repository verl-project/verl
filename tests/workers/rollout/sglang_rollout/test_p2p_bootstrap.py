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

from verl.workers.config import CheckpointEngineConfig, RolloutConfig
from verl.workers.rollout.sglang_rollout.p2p_bootstrap import (
    sanitize_sglang_engine_kwargs_for_p2p,
    should_enable_p2p_weight_transfer_bootstrap,
)


def test_should_enable_p2p_bootstrap_from_checkpoint_backend():
    config = RolloutConfig(
        name="sglang",
        checkpoint_engine=CheckpointEngineConfig(backend="p2p"),
    )
    assert should_enable_p2p_weight_transfer_bootstrap(config) is True


def test_should_disable_p2p_bootstrap_by_default():
    config = RolloutConfig(name="sglang")
    assert should_enable_p2p_weight_transfer_bootstrap(config) is False


def test_should_disable_p2p_bootstrap_when_engine_kwargs_set_seed_flag():
    config = RolloutConfig(
        name="sglang",
        engine_kwargs={"sglang": {"remote_instance_weight_loader_start_seed_via_transfer_engine": True}},
    )
    assert should_enable_p2p_weight_transfer_bootstrap(config) is False


def test_sanitize_strips_p2p_bootstrap_flags_from_engine_kwargs():
    config = RolloutConfig(
        name="sglang",
        engine_kwargs={
            "sglang": {
                "engine_info_bootstrap_port": 19001,
                "remote_instance_weight_loader_start_seed_via_transfer_engine": True,
            }
        },
    )
    sanitized = sanitize_sglang_engine_kwargs_for_p2p(config)
    assert "engine_info_bootstrap_port" not in sanitized
    assert "remote_instance_weight_loader_start_seed_via_transfer_engine" not in sanitized
