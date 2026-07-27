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

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from verl.workers.engine.fsdp.transformer_impl import FSDPEngine
from verl.workers.engine_workers_tinker import TinkerTrainingWorker


def test_tinker_forward_backward_requests_model_output():
    captured = {}

    def forward_backward_batch(data, **kwargs):
        captured["return_model_output"] = tu.get_non_tensor_data(data, key="return_model_output", default=None)
        return {}

    engine = SimpleNamespace(
        train_mode=lambda **kwargs: nullcontext(),
        forward_backward_batch=forward_backward_batch,
        is_mp_src_rank_with_outputs=lambda: False,
    )
    worker = SimpleNamespace(
        loss_fn=lambda: None,
        engine=engine,
        engine_config=SimpleNamespace(
            forward_only=False,
            use_dynamic_bsz=False,
            max_token_len_per_gpu=128,
            micro_batch_size_per_gpu=1,
            use_fused_kernels=False,
        ),
        model_config={},
    )

    result = TinkerTrainingWorker.forward_backward(worker, TensorDict({}, batch_size=[]))

    assert result is None
    assert captured["return_model_output"] is True


@pytest.mark.parametrize(("return_model_output", "expected"), [(False, False), (True, True)])
def test_fsdp_forward_backward_honors_return_model_output(monkeypatch, return_model_output, expected):
    data = TensorDict({"loss_mask": torch.ones(1)}, batch_size=[1])
    if return_model_output:
        tu.assign_non_tensor(data, return_model_output=True)

    loss = torch.tensor(1.0, requires_grad=True)
    model_output = {"log_probs": torch.tensor([-1.25])}
    engine = SimpleNamespace(
        ulysses_sequence_parallel_size=1,
        scaler=None,
        get_data_parallel_group=lambda: None,
        get_data_parallel_size=lambda: 1,
        forward_step=lambda micro_batch, loss_function, forward_only: (
            loss,
            {"model_output": model_output.copy(), "loss": 1.0, "metrics": {}},
        ),
    )

    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *args, **kwargs: None)
    monkeypatch.setattr("verl.workers.engine.fsdp.transformer_impl.get_device_id", lambda: "cpu")
    monkeypatch.setattr(
        "verl.workers.engine.fsdp.transformer_impl.prepare_micro_batches",
        lambda data, **kwargs: ([data], None),
    )
    monkeypatch.setattr(
        "verl.workers.engine.fsdp.transformer_impl.postprocess_batch_func",
        lambda output_lst, **kwargs: output_lst[0],
    )

    result = FSDPEngine.forward_backward_batch(engine, data, loss_function=lambda: None, forward_only=False)

    assert ("model_output" in result) is expected
    if expected:
        assert torch.equal(result["model_output"]["log_probs"], model_output["log_probs"])
