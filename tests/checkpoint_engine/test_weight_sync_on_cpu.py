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

from types import SimpleNamespace

import pytest
import torch

from verl.checkpoint_engine.base import CheckpointEngineManager
from verl.checkpoint_engine.weight_sync import assert_vllm_weight_sync_supported_parallelism, assert_weight_sync_equal
from verl.workers.config import CheckpointEngineConfig


def test_checkpoint_engine_manager_checks_initial_sync_only():
    manager = object.__new__(CheckpointEngineManager)
    manager.config = CheckpointEngineConfig(check_weight_sync=True)

    assert manager._should_check_loaded_weights_equal(None)
    assert manager._should_check_loaded_weights_equal(0)
    assert not manager._should_check_loaded_weights_equal(1)


def test_vllm_weight_sync_parallelism_check_accepts_tp1_dp1():
    parallel_config = SimpleNamespace(tensor_parallel_size=1, data_parallel_size=1, data_parallel_size_local=1)

    assert_vllm_weight_sync_supported_parallelism(parallel_config)


@pytest.mark.parametrize(
    "parallel_config",
    [
        SimpleNamespace(tensor_parallel_size=2, data_parallel_size=1, data_parallel_size_local=1),
        SimpleNamespace(tensor_parallel_size=1, data_parallel_size=2, data_parallel_size_local=1),
        SimpleNamespace(tensor_parallel_size=1, data_parallel_size=2, data_parallel_size_local=2),
        SimpleNamespace(tensor_parallel_size=1, data_parallel_size=1, data_parallel_size_local=2),
    ],
)
def test_vllm_weight_sync_parallelism_check_rejects_uncovered_parallelism(parallel_config):
    with pytest.raises(NotImplementedError, match="tensor_parallel_size=1 and data_parallel_size=1"):
        assert_vllm_weight_sync_supported_parallelism(parallel_config)


def test_assert_weight_sync_equal_accepts_direct_state_dict():
    expected_state_dict = {"model.embed_tokens.weight": torch.arange(6, dtype=torch.float32).view(2, 3)}
    actual_state_dict = {"model.embed_tokens.weight": expected_state_dict["model.embed_tokens.weight"].clone()}

    result = assert_weight_sync_equal(
        expected_state_dict=expected_state_dict,
        actual_state_dict=actual_state_dict,
        num_hidden_layers=0,
    )

    assert result == {"checked": 1, "missing": 0, "unexpected": 0, "mismatched": 0}


def test_assert_weight_sync_equal_rejects_any_value_mismatch():
    expected_state_dict = {"model.embed_tokens.weight": torch.ones(2, 3)}
    actual_state_dict = {"model.embed_tokens.weight": torch.ones(2, 3)}
    actual_state_dict["model.embed_tokens.weight"][0, 0] += 1e-6

    with pytest.raises(AssertionError, match="mismatched"):
        assert_weight_sync_equal(
            expected_state_dict=expected_state_dict,
            actual_state_dict=actual_state_dict,
            num_hidden_layers=0,
        )


def test_assert_weight_sync_equal_accepts_vllm_fused_state_dict():
    prefix = "model.layers.0"
    q_proj = torch.arange(8, dtype=torch.float32).view(2, 4)
    k_proj = q_proj + 10
    v_proj = q_proj + 20
    gate_proj = torch.arange(12, dtype=torch.float32).view(3, 4)
    up_proj = gate_proj + 30
    q_bias = torch.arange(2, dtype=torch.float32)
    k_bias = q_bias + 10
    v_bias = q_bias + 20
    embed_tokens = torch.arange(20, dtype=torch.float32).view(5, 4)

    expected_state_dict = {
        f"{prefix}.self_attn.q_proj.weight": q_proj,
        f"{prefix}.self_attn.k_proj.weight": k_proj,
        f"{prefix}.self_attn.v_proj.weight": v_proj,
        f"{prefix}.self_attn.q_proj.bias": q_bias,
        f"{prefix}.self_attn.k_proj.bias": k_bias,
        f"{prefix}.self_attn.v_proj.bias": v_bias,
        f"{prefix}.mlp.gate_proj.weight": gate_proj,
        f"{prefix}.mlp.up_proj.weight": up_proj,
        "model.embed_tokens.weight": embed_tokens,
        "lm_head.weight": embed_tokens,
    }
    actual_state_dict = {
        f"{prefix}.self_attn.qkv_proj.weight": torch.cat([q_proj, k_proj, v_proj], dim=0),
        f"{prefix}.self_attn.qkv_proj.bias": torch.cat([q_bias, k_bias, v_bias], dim=0),
        f"{prefix}.mlp.gate_up_proj.weight": torch.cat([gate_proj, up_proj], dim=0),
        "model.embed_tokens.weight": embed_tokens.clone(),
    }

    result = assert_weight_sync_equal(
        expected_state_dict=expected_state_dict,
        actual_state_dict=actual_state_dict,
        num_hidden_layers=1,
        tie_word_embeddings=True,
    )

    assert result == {"checked": 10, "missing": 0, "unexpected": 0, "mismatched": 0}


def test_assert_weight_sync_equal_reports_fused_value_mismatch():
    prefix = "model.layers.0"
    q_proj = torch.ones(2, 4)
    k_proj = torch.ones(2, 4) * 2
    v_proj = torch.ones(2, 4) * 3
    actual_qkv = torch.cat([q_proj, k_proj, v_proj], dim=0)
    actual_qkv[0, 0] += 1

    expected_state_dict = {
        f"{prefix}.self_attn.q_proj.weight": q_proj,
        f"{prefix}.self_attn.k_proj.weight": k_proj,
        f"{prefix}.self_attn.v_proj.weight": v_proj,
    }
    actual_state_dict = {f"{prefix}.self_attn.qkv_proj.weight": actual_qkv}

    with pytest.raises(AssertionError, match="mismatched"):
        assert_weight_sync_equal(
            expected_state_dict=expected_state_dict,
            actual_state_dict=actual_state_dict,
            num_hidden_layers=1,
        )


def test_assert_weight_sync_equal_reports_missing_and_unexpected_keys():
    with pytest.raises(AssertionError, match="missing=.*unexpected="):
        assert_weight_sync_equal(
            expected_state_dict={"expected.weight": torch.ones(1)},
            actual_state_dict={"actual.weight": torch.ones(1)},
            num_hidden_layers=0,
        )
