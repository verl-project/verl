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
"""Exercise the real split worker protocol with an engine that finalizes every group."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from verl.workers.engine_workers_group_delivery import GroupDeliveryTrainingWorker
from verl.workers.engine_workers_tinker import TinkerTrainingWorker


class FinalizingEngine:
    """CPU stand-in whose finalization would corrupt gradients carried across calls."""

    def __init__(self):
        self.grad = torch.zeros(3)
        self.module = [SimpleNamespace(buffers=[SimpleNamespace(grad_data=self.grad)])]
        self.optimizer = SimpleNamespace(config=SimpleNamespace(optimizer="adam"))
        self.tf_config = SimpleNamespace()
        self.steps = self.schedules = 0
        self.applied = None
        self.fail = False
        self.retain_output = None

    @contextmanager
    def train_mode(self, zero_grad_on_exit=True, **kwargs):
        """Mirror the native engine's gradient lifetime."""
        try:
            yield
        finally:
            if zero_grad_on_exit:
                self.optimizer_zero_grad()

    def optimizer_zero_grad(self):
        """Reset native schedule buffers."""
        self.grad.zero_()

    def forward_backward_batch(self, data, **kwargs):
        """Finalize each group; previous gradients must already have been removed."""
        assert not self.grad.any()
        self.retain_output = tu.get(data, "return_model_output")
        self.grad.add_(data["gradient"][0]).mul_(2)
        if self.fail:
            raise RuntimeError("engine backward failed")
        return {"model_output": {"log_probs": torch.zeros(1)}, "loss": [1.0], "metrics": {}}

    def optimizer_step(self):
        """Capture the final gradient received by the optimizer."""
        self.steps += 1
        self.applied = self.grad.clone()
        return self.grad.norm().item()

    def lr_scheduler_step(self):
        """Count updates independently from backward calls."""
        self.schedules += 1
        return 0.01

    def is_mp_src_rank_with_outputs(self):
        """Use a non-output rank to avoid GPU-only performance metrics on CPU."""
        return False


def worker():
    value = object.__new__(GroupDeliveryTrainingWorker)
    value.engine = FinalizingEngine()
    value.engine_config = OmegaConf.create(
        dict(
            strategy="megatron",
            use_megatron_fsdp=False,
            dtype="bfloat16",
            dynamic_context_parallel=False,
            virtual_pipeline_model_parallel_size=None,
            param_offload=False,
            optimizer_offload=False,
            forward_only=False,
            use_dynamic_bsz=True,
            max_token_len_per_gpu=128,
            micro_batch_size_per_gpu=1,
            use_fused_kernels=False,
        )
    )
    value.optimizer_config = SimpleNamespace(use_precision_aware_optimizer=False)
    value.model_config = OmegaConf.create({"mtp": {"enable": False}})
    value.loss_fn = lambda *args: None
    return value


def group(uid, weight, gradient):
    data = TensorDict({"gradient": torch.tensor([gradient], dtype=torch.float32)}, batch_size=1)
    tu.assign_non_tensor(data, group_delivery_id=uid, group_delivery_weight=weight)
    return data


def test_real_worker_reuses_split_api_and_commits_once_per_batch():
    value = worker()
    for _ in range(2):
        prior_steps = value.engine.steps
        value.begin_group_batch(["a", "b"])
        value.backward_group(group("b", 3, [1, 2, 3]))
        assert value.engine.steps == prior_steps
        assert value.engine.schedules == prior_steps
        with pytest.raises(RuntimeError, match="1/2"):
            value.finish_group_batch()
        value.backward_group(group("a", 1, [3, 4, 5]))
        assert value.engine.retain_output is False
        value.finish_group_batch()
        torch.testing.assert_close(value.engine.applied, torch.tensor([3.0, 5.0, 7.0]))
        assert value.engine.steps == value.engine.schedules == prior_steps + 1
        assert value.group_accumulator is None
        assert not value.engine.grad.any()


def test_failed_worker_batch_does_not_step_or_keep_partial_gradients():
    value = worker()
    value.begin_group_batch(["a", "b"])
    value.backward_group(group("a", 1, [1, 1, 1]))
    value.engine.fail = True
    with pytest.raises(RuntimeError, match="engine backward failed"):
        value.backward_group(group("b", 1, [2, 2, 2]))
    value.abort_group_batch()
    assert value.engine.steps == value.engine.schedules == 0
    assert value.group_accumulator is None
    assert not value.engine.grad.any()


def test_tinker_preserves_default_model_outputs_and_explicit_opt_out():
    value = worker()
    for retain in (None, False, True):
        data = group("a", 1, [1, 1, 1])
        if retain is not None:
            tu.assign_non_tensor(data, return_model_output=retain)
        value.optimizer_zero_grad()
        TinkerTrainingWorker.forward_backward(value, data)
        assert value.engine.retain_output is (True if retain is None else retain)


@pytest.mark.parametrize(
    ("key", "setting"),
    [
        ("dtype", "float16"),
        ("param_offload", True),
        ("dynamic_context_parallel", True),
        ("virtual_pipeline_model_parallel_size", 2),
        ("use_megatron_fsdp", True),
    ],
)
def test_incompatible_engine_fails_before_rollout(key, setting):
    value = worker()
    value.engine_config[key] = setting
    with pytest.raises(ValueError):
        value.begin_group_batch(["a"])
    assert value.engine.steps == 0


@pytest.mark.parametrize("setting", ["qat", "fp16", "fp8", "moe_aux_loss_coeff", "moe_router_enable_expert_bias"])
def test_actual_transformer_state_is_checked_before_rollout(setting):
    value = worker()
    if setting == "qat":
        value.engine._qat_enabled = True
    else:
        setattr(value.engine.tf_config, setting, True)
    with pytest.raises(ValueError):
        value.begin_group_batch(["a"])
    assert value.engine.steps == 0
