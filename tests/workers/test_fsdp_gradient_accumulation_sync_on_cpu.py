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

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from verl.workers.engine.fsdp import transformer_impl
from verl.workers.engine.fsdp.transformer_impl import FSDPEngine


class _FSDP1Module:
    def __init__(self):
        self.events = []

    @contextmanager
    def no_sync(self):
        self.events.append("enter")
        try:
            yield
        finally:
            self.events.append("exit")


class _FSDP2Module:
    def __init__(self):
        self.events = []

    def set_requires_gradient_sync(self, enabled):
        self.events.append(enabled)


def _make_engine(module, *, enabled=True):
    engine = object.__new__(FSDPEngine)
    engine.module = module
    engine.engine_config = SimpleNamespace(use_no_sync_for_gradient_accumulation=enabled)
    return engine


@pytest.mark.parametrize("version,module_cls", [(1, _FSDP1Module), (2, _FSDP2Module)])
def test_gradient_sync_context_skips_non_final_micro_batch(monkeypatch, version, module_cls):
    module = module_cls()
    engine = _make_engine(module)
    monkeypatch.setattr(transformer_impl, "fsdp_version", lambda _: version)

    with engine._gradient_sync_context(is_last_micro_batch=False):
        module.events.append("backward")

    expected = ["enter", "backward", "exit"] if version == 1 else [False, "backward", True]
    assert module.events == expected


def test_gradient_sync_context_restores_fsdp2_after_error(monkeypatch):
    module = _FSDP2Module()
    engine = _make_engine(module)
    monkeypatch.setattr(transformer_impl, "fsdp_version", lambda _: 2)

    with pytest.raises(RuntimeError, match="backward failed"):
        with engine._gradient_sync_context(is_last_micro_batch=False):
            raise RuntimeError("backward failed")

    assert module.events == [False, True]


@pytest.mark.parametrize("enabled,is_last", [(False, False), (True, True)])
def test_gradient_sync_context_keeps_sync_for_default_or_final_micro_batch(monkeypatch, enabled, is_last):
    module = _FSDP2Module()
    engine = _make_engine(module, enabled=enabled)
    monkeypatch.setattr(transformer_impl, "fsdp_version", lambda _: 2)

    with engine._gradient_sync_context(is_last_micro_batch=is_last):
        module.events.append("backward")

    assert module.events == ["backward"]


def test_gradient_sync_context_defaults_to_sync_for_other_engine_configs(monkeypatch):
    module = _FSDP2Module()
    engine = _make_engine(module)
    engine.engine_config = SimpleNamespace()
    monkeypatch.setattr(transformer_impl, "fsdp_version", lambda _: 2)

    with engine._gradient_sync_context(is_last_micro_batch=False):
        module.events.append("backward")

    assert module.events == ["backward"]


def test_forward_backward_batch_syncs_only_final_micro_batch(monkeypatch):
    engine = _make_engine(_FSDP2Module())
    engine.ulysses_sequence_parallel_size = 1
    engine.scaler = None
    engine.get_data_parallel_group = lambda: None
    engine.get_data_parallel_size = lambda: 1
    sync_states = []

    @contextmanager
    def record_sync(*, is_last_micro_batch):
        sync_states.append(is_last_micro_batch)
        yield

    engine._gradient_sync_context = record_sync
    engine.forward_step = lambda micro_batch, loss_function, forward_only: (
        micro_batch["loss"].sum(),
        {"metrics": {}},
    )

    micro_batches = [
        TensorDict({"loss": torch.tensor([float(i)], requires_grad=True)}, batch_size=[1]) for i in range(3)
    ]
    monkeypatch.setattr(
        transformer_impl,
        "prepare_micro_batches",
        lambda **_: (micro_batches, None),
    )
    monkeypatch.setattr(
        transformer_impl,
        "postprocess_batch_func",
        lambda output_lst, indices, data: output_lst,
    )
    monkeypatch.setattr(transformer_impl, "get_device_id", lambda: "cpu")
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *args, **kwargs: None)

    data = TensorDict({"loss_mask": torch.ones(3)}, batch_size=[3])
    output = engine.forward_backward_batch(data, loss_function=lambda: None)

    assert sync_states == [False, False, True]
    assert len(output) == 3
    assert all(micro_batch["loss"].grad is not None for micro_batch in micro_batches)
