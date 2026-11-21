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

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch


@pytest.fixture
def callback():
    """Exercise the real callback without importing optional Megatron GPU dependencies."""
    source = Path(__file__).resolve().parents[2] / "verl/workers/engine/megatron/transformer_impl.py"
    tree = ast.parse(source.read_text())
    engine_class = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MegatronEngineWithLMHead"
    )
    method = next(
        node
        for node in engine_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "postprocess_micro_batch_func"
    )
    mpu = SimpleNamespace(get_context_parallel_rank=lambda: 0)
    tu = SimpleNamespace(get_non_tensor_data=lambda data, key, default=None: data.get(key, default))
    namespace = {
        "TensorDict": object,
        "torch": torch,
        "mpu": mpu,
        "tu": tu,
        "detach_tree": lambda value: value,
        "DCP_LOCAL_NUM_TOKENS": "dcp_local_num_tokens",
        "DCP_SAMPLE_IDS": "dcp_sample_ids",
        "DCP_GROUP_LEADER": "dcp_group_leader",
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    return SimpleNamespace(run=namespace["postprocess_micro_batch_func"], mpu=mpu, tu=tu)


@pytest.mark.parametrize(
    ("cp_size", "token_count"),
    [(1, 0), (1, 7), (2, 1), (2, 77), (2, 78), (3, 1), (3, 7), (4, 2), (4, 132)],
)
def test_static_cp_token_counts_sum_to_real_tokens(callback, cp_size, token_count):
    """Megatron sums rank counts before normalizing the shared loss gradient."""
    engine = SimpleNamespace(
        tf_config=SimpleNamespace(calculate_per_token_loss=True),
        engine_config=SimpleNamespace(
            dynamic_context_parallel=False, use_remove_padding=True, context_parallel_size=cp_size
        ),
        prepare_model_outputs=lambda output, data: {},
        get_data_parallel_group=lambda: None,
        _routed_num_tokens=lambda data: data["attention_mask"].sum(),
    )
    data = {
        "input_ids": torch.ones((1, token_count), dtype=torch.long),
        "attention_mask": torch.ones((1, token_count), dtype=torch.long),
        "num_micro_batch": 1,
        "routed_num_tokens": token_count,
        "dp_size": 1,
    }
    loss = torch.tensor(2.0)

    def loss_function(**kwargs):
        return loss, {}

    counts = []
    for rank in range(cp_size):
        with (
            patch.object(callback.mpu, "get_context_parallel_rank", return_value=rank),
            patch.object(
                callback.tu,
                "get_non_tensor_data",
                side_effect=lambda data, key, default=None: data.get(key, default),
            ),
        ):
            loss_sum, count, _ = callback.run(
                engine, output=None, data=data, forward_only=False, loss_function=loss_function
            )
        assert loss_sum.item() == 2.0 * token_count
        counts.append(count.item())

    assert sum(counts) == token_count
    assert max(counts) - min(counts) <= 1


def test_dynamic_cp_preserves_scheduler_token_count(callback):
    engine = SimpleNamespace(
        tf_config=SimpleNamespace(calculate_per_token_loss=True),
        engine_config=SimpleNamespace(dynamic_context_parallel=True, use_remove_padding=True),
        prepare_model_outputs=lambda output, data: {},
        get_data_parallel_group=lambda: None,
        _routed_num_tokens=lambda data: data["attention_mask"].sum(),
    )
    data = {
        "input_ids": torch.ones((1, 7), dtype=torch.long),
        "attention_mask": torch.ones((1, 7), dtype=torch.long),
        "num_micro_batch": 1,
        "routed_num_tokens": 7,
        "dp_size": 1,
        "dcp_local_num_tokens": 3,
    }

    def loss_function(**kwargs):
        return torch.tensor(2.0), {}

    with (
        patch.object(callback.mpu, "get_context_parallel_rank", side_effect=AssertionError("unexpected rank")),
        patch.object(
            callback.tu,
            "get_non_tensor_data",
            side_effect=lambda data, key, default=None: data.get(key, default),
        ),
    ):
        loss_sum, count, _ = callback.run(
            engine, output=None, data=data, forward_only=False, loss_function=loss_function
        )
    assert loss_sum.item() == 14.0
    assert count.item() == 3
