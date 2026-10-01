# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""CPU tests for Megatron training hook registration."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from verl.utils import megatron_utils


class _FakeDDP:
    def __init__(self, config, *, overlap_grad_reduce=True, align_param_gather=True):
        self.config = config
        self.ddp_config = SimpleNamespace(
            overlap_grad_reduce=overlap_grad_reduce,
            align_param_gather=align_param_gather,
        )

    def no_sync(self):
        pass

    def start_grad_sync(self):
        pass

    def start_param_sync(self):
        pass


@pytest.fixture
def patch_megatron(monkeypatch):
    finalize_model_grads = Mock(name="finalize_model_grads")
    get_model_config = Mock(side_effect=lambda chunk: chunk.config)
    monkeypatch.setattr(megatron_utils, "DDP", _FakeDDP)
    monkeypatch.setattr("megatron.core.distributed.finalize_model_grads", finalize_model_grads)
    monkeypatch.setattr("megatron.core.utils.get_model_config", get_model_config)
    return SimpleNamespace(
        finalize_model_grads=finalize_model_grads,
        get_model_config=get_model_config,
    )


def _new_config():
    return SimpleNamespace(
        grad_scale_func=None,
        finalize_model_grads_func=None,
        no_sync_func=None,
        grad_sync_func=None,
        param_sync_func=None,
    )


def _assert_hooks_match_chunks(hooks, chunks):
    assert len(hooks) == len(chunks)
    assert all(hook.__self__ is chunk for hook, chunk in zip(hooks, chunks, strict=True))


def test_register_training_hooks_once_for_shared_vpp_config(patch_megatron):
    config = _new_config()
    chunks = [_FakeDDP(config) for _ in range(3)]
    scale_loss = object()
    optimizer = SimpleNamespace(
        scale_loss=scale_loss,
        config=SimpleNamespace(overlap_param_gather=True),
    )

    megatron_utils.register_megatron_training_hooks(chunks, optimizer)

    assert config.grad_scale_func is scale_loss
    assert config.finalize_model_grads_func is patch_megatron.finalize_model_grads
    patch_megatron.get_model_config.assert_called_once_with(chunks[0])
    _assert_hooks_match_chunks(config.no_sync_func, chunks)
    _assert_hooks_match_chunks(config.grad_sync_func, chunks)
    _assert_hooks_match_chunks(config.param_sync_func, chunks)


def test_register_training_hooks_uses_callables_without_vpp(patch_megatron):
    config = _new_config()
    chunk = _FakeDDP(config)
    optimizer = SimpleNamespace(
        scale_loss=object(),
        config=SimpleNamespace(overlap_param_gather=True),
    )

    megatron_utils.register_megatron_training_hooks([chunk], optimizer)

    assert callable(config.no_sync_func)
    assert callable(config.grad_sync_func)
    assert callable(config.param_sync_func)
    assert config.no_sync_func.__self__ is chunk
    assert config.grad_sync_func.__self__ is chunk
    assert config.param_sync_func.__self__ is chunk
