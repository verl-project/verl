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
"""CPU regression tests for outer-chunk loss-mask ownership and lifetime."""

import sys
import types
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from verl.workers.engine.megatron import outer_chunk
from verl.workers.engine.megatron.chunk_loss_mask import chunk_loss_data, reset_chunk_loss_mask_state
from verl.workers.engine.megatron.transformer_impl import MegatronEngineWithLMHead


class _Batch(dict):
    def copy(self):
        return _Batch(self)


def _nested_mask(values):
    return torch.nested.nested_tensor([torch.tensor(values, dtype=torch.float32)], layout=torch.jagged)


def _pass_data(engine, batch, index, count):
    engine._dsv41_chunk_ctx = (index, count, object())
    return chunk_loss_data(engine, {"log_probs": torch.zeros(4)}, batch)


def test_two_passes_read_the_original_mask_and_get_their_own_window():
    engine = SimpleNamespace(_dsv41_mask_full=None, _dsv41_n0=None)
    batch = _Batch(loss_mask=_nested_mask([1, 0, 1, 0, 0, 1, 0, 1]))
    original = batch["loss_mask"].values().clone()

    first = _pass_data(engine, batch, index=0, count=2)
    second = _pass_data(engine, batch, index=1, count=2)

    assert torch.equal(batch["loss_mask"].values(), original)
    assert torch.equal(engine._dsv41_mask_full, original)
    assert torch.equal(first["loss_mask"].values(), original[:4])
    assert torch.equal(second["loss_mask"].values(), original[4:])
    assert first is not batch
    assert second is not batch


def test_postprocess_passes_isolated_masks_to_loss_without_mutating_batch():
    engine = object.__new__(MegatronEngineWithLMHead)
    engine._dsv41_mask_full = None
    engine._dsv41_n0 = None
    engine.prepare_model_outputs = lambda output, data: output
    engine.get_data_parallel_group = lambda: None
    engine.engine_config = SimpleNamespace(dynamic_context_parallel=False)
    engine.tf_config = None
    batch = TensorDict(
        {
            "input_ids": torch.zeros((1, 8), dtype=torch.long),
            "loss_mask": _nested_mask([1, 0, 1, 0, 0, 1, 0, 1]),
            "num_micro_batch": torch.ones(1, dtype=torch.long),
        },
        batch_size=[1],
    )
    original = batch["loss_mask"].values().clone()
    seen = []

    def loss_function(model_output, data, dp_group):
        seen.append(data)
        return torch.ones(()), {}

    for index in (0, 1):
        engine._dsv41_chunk_ctx = (index, 2, object())
        engine.postprocess_micro_batch_func(
            {"log_probs": torch.zeros(4)}, batch, forward_only=False, loss_function=loss_function
        )

    assert torch.equal(batch["loss_mask"].values(), original)
    assert torch.equal(seen[0]["loss_mask"].values(), original[:4])
    assert torch.equal(seen[1]["loss_mask"].values(), original[4:])
    assert seen[0] is not batch
    assert seen[1] is not batch
    assert seen[0]["input_ids"] is batch["input_ids"]
    assert seen[1]["input_ids"] is batch["input_ids"]


def test_each_logical_step_starts_without_previous_mask_state():
    engine = SimpleNamespace(_dsv41_mask_full=None, _dsv41_n0=None)
    first_batch = _Batch(loss_mask=_nested_mask([1, 1, 1, 1, 0, 0, 0, 0]))
    _pass_data(engine, first_batch, index=0, count=2)
    assert engine._dsv41_n0 == 4
    assert engine._dsv41_mask_full is not None

    reset_chunk_loss_mask_state(engine)
    assert engine._dsv41_mask_full is None
    assert engine._dsv41_n0 is None

    second_values = torch.tensor([0, 0, 1, 0, 1, 0, 1, 0], dtype=torch.float32)
    second_batch = _Batch(loss_mask=_nested_mask(second_values.tolist()))
    second = _pass_data(engine, second_batch, index=0, count=2)

    assert torch.equal(engine._dsv41_mask_full, second_values)
    assert torch.equal(second["loss_mask"].values(), second_values[:4])


def test_chunk_driver_resets_mask_state_at_each_logical_step(monkeypatch):
    class _Tape:
        def __init__(self, count):
            self.count = count

        def begin(self, index, mode):
            pass

    chunk_tape_module = types.ModuleType("megatron.core.pipeline_parallel.chunk_tape")
    chunk_tape_module.ChunkTape = _Tape
    monkeypatch.setitem(sys.modules, "megatron.core.pipeline_parallel.chunk_tape", chunk_tape_module)
    monkeypatch.setattr("verl.workers.engine.megatron.transformer_impl.unwrap_model", lambda model: model)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 1)

    engine = object.__new__(MegatronEngineWithLMHead)
    engine._dsv41_mask_full = torch.ones(8)
    engine._dsv41_n0 = 4
    engine.get_context_parallel_group = lambda: None
    model = torch.nn.Linear(1, 1)
    from megatron.core.transformer.transformer_config import TransformerConfig

    model.config = TransformerConfig(num_layers=1, hidden_size=16, num_attention_heads=2)
    batch = _Batch()
    observed = []
    pass_contexts = []

    def forward_step(*args, **kwargs):
        def loss_callback(output):
            pass_contexts.append(engine._dsv41_chunk_ctx)
            return output["log_probs"].sum() * 0.0, {"loss": 0.0, "metrics": {}}

        return {"log_probs": torch.ones(4)}, loss_callback

    engine.forward_step = forward_step

    def run_one_step(**kwargs):
        observed.append((engine._dsv41_mask_full, engine._dsv41_n0))
        return kwargs["run_pass"](1).detach()

    monkeypatch.setattr(outer_chunk, "run_chunked_step", run_one_step)
    engine._dsv41_chunked_forward_step(batch, model, None, None, 2)
    engine._dsv41_mask_full = torch.zeros(8)
    engine._dsv41_n0 = 99
    engine._dsv41_chunked_forward_step(batch, model, None, None, 2)

    assert observed == [(None, None), (None, None)]
    assert pass_contexts == [None, None]


def test_mask_shorter_than_log_probs_raises():
    engine = object.__new__(MegatronEngineWithLMHead)
    engine._dsv41_mask_full = None
    engine._dsv41_n0 = None
    engine._dsv41_chunk_ctx = (0, 2, object())
    engine.prepare_model_outputs = lambda output, data: output
    engine.get_data_parallel_group = lambda: None
    engine.engine_config = SimpleNamespace(dynamic_context_parallel=False)
    engine.tf_config = None
    batch = TensorDict(
        {
            "input_ids": torch.zeros((1, 3), dtype=torch.long),
            "loss_mask": _nested_mask([1, 0, 1]),
            "num_micro_batch": torch.ones(1, dtype=torch.long),
        },
        batch_size=[1],
    )
    loss_function = lambda model_output, data, dp_group: (torch.ones(()), {})

    with pytest.raises(ValueError, match="shorter than log_probs"):
        engine.postprocess_micro_batch_func(
            {"log_probs": torch.zeros(4)}, batch, forward_only=False, loss_function=loss_function
        )


def test_mask_window_overflow_raises_instead_of_truncating():
    engine = SimpleNamespace(_dsv41_mask_full=None, _dsv41_n0=None)
    batch = _Batch(loss_mask=_nested_mask([1, 0, 1, 0, 1, 0]))
    _pass_data(engine, batch, index=0, count=2)

    engine._dsv41_chunk_ctx = (1, 2, object())
    with pytest.raises(ValueError, match="window exceeds"):
        chunk_loss_data(engine, {"log_probs": torch.zeros(4)}, batch)


def test_k_one_is_identity_for_the_ordinary_full_length_mask():
    engine = object.__new__(MegatronEngineWithLMHead)
    engine._dsv41_mask_full = None
    engine._dsv41_n0 = None
    engine._dsv41_chunk_ctx = (0, 1, object())
    engine.prepare_model_outputs = lambda output, data: output
    engine.get_data_parallel_group = lambda: None
    engine.engine_config = SimpleNamespace(dynamic_context_parallel=False)
    engine.tf_config = None
    batch = TensorDict(
        {
            "input_ids": torch.zeros((1, 4), dtype=torch.long),
            "loss_mask": _nested_mask([1, 0, 1, 0]),
            "num_micro_batch": torch.ones(1, dtype=torch.long),
        },
        batch_size=[1],
    )
    seen = []

    def loss_function(model_output, data, dp_group):
        seen.append(data)
        return torch.ones(()), {}

    engine.postprocess_micro_batch_func(
        {"log_probs": torch.zeros(4)}, batch, forward_only=False, loss_function=loss_function
    )

    assert seen == [batch]
    assert engine._dsv41_mask_full is None
    assert engine._dsv41_n0 is None
