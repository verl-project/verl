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

"""The split training API must retain successful gradients and release failures."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from tensordict import NonTensorData, TensorDict

from verl.utils.device import get_device_name
from verl.workers.engine_workers import TrainingWorker


def _worker(failure=None, *, native=False):
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    events = []
    engine = SimpleNamespace(optimizer=optimizer, mode=None)
    if native:
        pytest.importorskip("megatron.core")
        from verl.workers.engine.megatron.transformer_impl import EngineTrainModeCtx, MegatronEngineWithLMHead

        engine = object.__new__(MegatronEngineWithLMHead)
        engine.optimizer = optimizer
        engine.module = []
        engine.mode = None
        engine._is_offload_param = True
        engine._is_offload_optimizer = True
        engine.moves = []
        # Exercise the real context while recording device-transfer requests.
        engine.to = lambda **kwargs: engine.moves.append(kwargs["device"])

    @contextmanager
    def train_mode(**kwargs):
        if native:
            with EngineTrainModeCtx(engine, **kwargs):
                events.append("enter")
                try:
                    yield
                finally:
                    events.append("exit")
            return
        engine.mode = "train"
        events.append("enter")
        try:
            yield
        finally:
            # EngineTrainModeCtx clears gradients and offloads on exit.
            optimizer.zero_grad()
            engine.mode = None
            events.append("exit")

    zero_grad_failed = False

    def zero_grad():
        nonlocal zero_grad_failed
        if failure == "zero_grad" and not zero_grad_failed:
            zero_grad_failed = True
            raise RuntimeError("injected zero_grad failure")
        optimizer.zero_grad()

    def forward_backward(data, loss_function):
        (parameter * 2).backward()
        if failure == "forward_backward":
            raise RuntimeError("injected forward_backward failure")
        return {"model_output": {}, "loss": [2.0], "metrics": {}}

    def step():
        if failure == "optimizer":
            raise RuntimeError("injected optimizer failure")
        optimizer.step()
        return None

    def schedule():
        if failure == "scheduler":
            raise RuntimeError("injected scheduler failure")
        return 0.1

    def postprocess(output, **kwargs):
        if failure == "postprocess":
            raise RuntimeError("injected postprocess failure")
        return TensorDict({}, batch_size=[])

    engine.train_mode = train_mode
    engine.optimizer_zero_grad = zero_grad
    engine.forward_backward_batch = forward_backward
    engine.optimizer_step = step
    engine.lr_scheduler_step = schedule
    engine.is_mp_src_rank_with_outputs = lambda: True
    worker = SimpleNamespace(
        engine=engine,
        loss_fn=lambda **kwargs: None,
        engine_config=SimpleNamespace(
            forward_only=False,
            use_dynamic_bsz=False,
            max_token_len_per_gpu=8,
            micro_batch_size_per_gpu=1,
            use_fused_kernels=False,
        ),
        model_config={"use_remove_padding": True},
        optimizer_config=SimpleNamespace(lr=0.1),
        _client_lr=None,
        _postprocess_output=postprocess,
    )
    data = TensorDict({"global_token_num": NonTensorData([1])}, batch_size=[1])
    return worker, data, parameter, events


def test_successful_split_step_retains_gradients_until_optimizer():
    worker, data, parameter, events = _worker()
    TrainingWorker.forward_backward_only(worker, data)
    assert events == ["enter"]
    assert worker.engine.mode == "train"
    assert parameter.grad.item() == 2.0
    TrainingWorker.optim_step_only(worker)
    assert parameter.item() == pytest.approx(0.8)
    assert parameter.grad is None
    assert worker.engine.mode is None
    assert events == ["enter", "exit"]
    assert worker._train_mode_ctx is None


@pytest.mark.parametrize("failure", ["zero_grad", "forward_backward", "postprocess"])
def test_failed_forward_backward_releases_context_and_partial_gradients(failure):
    worker, data, parameter, events = _worker(failure)
    with pytest.raises(RuntimeError, match=f"injected {failure} failure"):
        TrainingWorker.forward_backward_only(worker, data)
    assert events == ["enter", "exit"]
    assert worker.engine.mode is None
    assert parameter.grad is None
    assert getattr(worker, "_train_mode_ctx", None) is None


@pytest.mark.parametrize("failure", ["optimizer", "scheduler"])
def test_failed_optimizer_releases_retained_context(failure):
    worker, data, parameter, events = _worker(failure)
    TrainingWorker.forward_backward_only(worker, data)
    with pytest.raises(RuntimeError, match=f"injected {failure} failure"):
        TrainingWorker.optim_step_only(worker)
    assert events == ["enter", "exit"]
    assert worker.engine.mode is None
    assert parameter.grad is None
    assert worker._train_mode_ctx is None


def test_second_forward_backward_cannot_discard_pending_gradients():
    worker, data, parameter, events = _worker()
    TrainingWorker.forward_backward_only(worker, data)
    context = worker._train_mode_ctx
    try:
        with pytest.raises(RuntimeError, match="optim_step_only"):
            TrainingWorker.forward_backward_only(worker, data)
        assert worker._train_mode_ctx is context
        assert events == ["enter"]
        assert parameter.grad.item() == 2.0
    finally:
        TrainingWorker.optim_step_only(worker)


@pytest.mark.parametrize("failure", [None, "zero_grad", "forward_backward", "postprocess", "optimizer", "scheduler"])
def test_real_megatron_context_cleanup(failure):
    worker, data, parameter, events = _worker(failure, native=True)

    def train():
        TrainingWorker.forward_backward_only(worker, data)
        assert parameter.grad.item() == 2.0
        TrainingWorker.optim_step_only(worker)

    if failure is None:
        train()
        assert parameter.item() == pytest.approx(0.8)
    else:
        with pytest.raises(RuntimeError, match=f"injected {failure} failure"):
            train()
    assert events == ["enter", "exit"]
    assert worker.engine.mode is None
    assert parameter.grad is None
    assert worker.engine.moves == [get_device_name(), "cpu"]
    assert getattr(worker, "_train_mode_ctx", None) is None
