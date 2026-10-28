# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

"""CPU-policy ownership belongs to engine contexts, including failures."""

import math
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from verl.workers.engine.fsdp.fsdp_turbo_impl import FSDPTurboEngineWithLMHead
from verl.workers.engine.fsdp.transformer_impl import EngineEvalModeCtx, EngineTrainModeCtx, FSDPEngine


@pytest.mark.parametrize("context_class", [EngineEvalModeCtx, EngineTrainModeCtx])
@pytest.mark.parametrize("failed", [False, True])
@pytest.mark.parametrize("disabled", [False, True])
def test_policy_shards_park_on_context_exit(context_class, failed, disabled):
    engine = object.__new__(FSDPEngine)
    engine.module = torch.nn.Linear(2, 2)
    engine._uses_fsdp2_cpu_offload_policy = True
    engine._is_offload_param = engine._is_offload_optimizer = False
    engine.ulysses_parallel_group = None
    engine.to = Mock()
    engine.optimizer_zero_grad = Mock()
    context = context_class(engine, disable_auto_offload=disabled)
    with patch("verl.workers.engine.fsdp.transformer_impl.fsdp_version", return_value=2):
        context.__enter__()
        context.__exit__(RuntimeError if failed else None, None, None)
    parked = [call for call in engine.to.call_args_list if call.kwargs.get("model")]
    assert len(parked) == (0 if disabled else 1)
    if parked:
        assert parked[0].kwargs["device"] == "cpu"
        assert parked[0].kwargs["optimizer"] == (context_class is EngineTrainModeCtx)
    assert engine.mode is None


def test_policy_to_cpu_keeps_buffers_ready_without_loading_parameters():
    engine = object.__new__(FSDPEngine)
    engine._uses_fsdp2_cpu_offload_policy = True
    engine.module = torch.nn.Linear(2, 2)
    engine.module.register_buffer("rotary", torch.ones(3))
    engine.optimizer = None
    with (
        patch("verl.workers.engine.fsdp.transformer_impl.offload_fsdp_model_to_cpu") as offload,
        patch("verl.workers.engine.fsdp.transformer_impl.load_fsdp_model_to_gpu") as load,
        patch("verl.workers.engine.fsdp.transformer_impl.get_device_id", return_value="cpu"),
    ):
        engine.to("cpu", model=True, optimizer=False, grad=False)
        engine.to("cuda", model=True, optimizer=False, grad=False)
    offload.assert_called_once_with(engine.module)
    load.assert_not_called()
    torch.testing.assert_close(engine.module.rotary, torch.ones(3))


@pytest.mark.parametrize("tp_rank", [0, 1])
@pytest.mark.parametrize("cp_rank", [None, 0, 1])
def test_turbo_output_collector_requires_both_tp_and_cp_source(tp_rank, cp_rank):
    engine = object.__new__(FSDPTurboEngineWithLMHead)
    engine._parallel_state = SimpleNamespace(
        get_tp_rank=lambda: tp_rank,
        is_group_enable=lambda name: cp_rank is not None,
        get_ulysses_rank=lambda: cp_rank,
    )
    assert engine.is_mp_src_rank_with_outputs() is (tp_rank == 0 and cp_rank in (None, 0))


@pytest.mark.parametrize("is_last", [False, True])
def test_turbo_gradient_sync_context_leaves_ownership_to_native_wrapper(is_last):
    engine = object.__new__(FSDPTurboEngineWithLMHead)
    engine.module = object()  # No torch FSDP sync methods: Turbo owns its collectives.
    with engine._gradient_sync_context(is_last_micro_batch=is_last):
        pass


@pytest.mark.parametrize("grad_norm", [1.25, float("inf"), float("nan")])
@pytest.mark.parametrize("scaled", [False, True])
def test_turbo_float_clip_result_and_nonfinite_guard_without_legacy_qat(monkeypatch, grad_norm, scaled):
    engine = object.__new__(FSDPTurboEngineWithLMHead)
    engine.optimizer_config = SimpleNamespace(clip_grad=1.0)
    engine.module = torch.nn.Linear(2, 2)
    engine.optimizer = Mock()
    engine.scaler = Mock() if scaled else None
    monkeypatch.setitem(
        sys.modules,
        "fsdp_turbo.training.clip_grads",
        SimpleNamespace(clip_grad_norm=lambda **kwargs: grad_norm),
    )
    if not math.isfinite(grad_norm):
        with pytest.raises(FloatingPointError, match="NONFINITE GRADIENT"):
            engine.optimizer_step()
        engine.optimizer.zero_grad.assert_called_once()
        engine.optimizer.step.assert_not_called()
        if scaled:
            engine.scaler.step.assert_not_called()
    else:
        assert engine.optimizer_step() == grad_norm
        if scaled:
            engine.scaler.unscale_.assert_called_once_with(engine.optimizer)
            engine.scaler.step.assert_called_once_with(engine.optimizer)
            engine.scaler.update.assert_called_once()
        else:
            engine.optimizer.step.assert_called_once()
