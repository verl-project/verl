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
"""VeOmni engine: optim.override_optimizer_config reaches build_optimizer; fused AdamW under FSDP2 CPU offload.

VeOmni is optional in CPU CI, so its import surfaces are stubbed as in the other VeOmni engine tests.
"""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

for _mod in (
    "veomni",
    "veomni.arguments",
    "veomni.distributed",
    "veomni.distributed.torch_parallelize",
    "veomni.models",
    "veomni.models.auto",
    "veomni.models.checkpoint_tensor_loading",
    "veomni.optim",
    "veomni.utils",
    "veomni.utils.seqlen_pos_transform_utils",
):
    sys.modules.setdefault(_mod, MagicMock())

from verl.workers.config import VeOmniEngineConfig, VeOmniOptimizerConfig  # noqa: E402
from verl.workers.engine.veomni import transformer_impl  # noqa: E402


def _engine(enable_fsdp_offload=False, override_optimizer_config=None):
    # Only the attributes the methods under test read: __init__ needs a distributed mesh.
    engine = object.__new__(transformer_impl.VeOmniEngine)
    engine.engine_config = VeOmniEngineConfig(use_torch_compile=False, enable_fsdp_offload=enable_fsdp_offload)
    engine.optimizer_config = VeOmniOptimizerConfig(lr=1e-6, override_optimizer_config=override_optimizer_config)
    return engine


@pytest.fixture
def build_optimizer_calls(monkeypatch):
    calls = []

    def build_optimizer(module, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(register_step_pre_hook=lambda hook: None)

    monkeypatch.setattr(transformer_impl, "build_optimizer", build_optimizer)
    return calls


@pytest.mark.parametrize(
    ("enable_fsdp_offload", "override", "fused"),
    [
        (False, None, None),
        (False, {"fused": True}, True),
        # FSDP2 CPU offload steps AdamW on host tensors, where foreach allocates a temporary
        # as large as the local parameters; the fused kernel is the default there.
        (True, None, True),
        (True, {"fused": False}, False),
    ],
    ids=["default", "override", "offload-default", "offload-override"],
)
def test_optimizer_overrides_reach_build_optimizer(build_optimizer_calls, enable_fsdp_offload, override, fused):
    _engine(enable_fsdp_offload, override)._build_optimizer(SimpleNamespace())

    (kwargs,) = build_optimizer_calls
    assert kwargs.get("fused") == fused
    assert kwargs["optimizer_type"] == "adamw"


def test_override_keys_other_than_fused_are_forwarded(build_optimizer_calls):
    _engine(True, {"eps": 1e-6})._build_optimizer(SimpleNamespace())

    (kwargs,) = build_optimizer_calls
    assert kwargs["eps"] == 1e-6 and kwargs["fused"] is True


def test_overrides_replace_the_base_optimizer_arguments(build_optimizer_calls, monkeypatch):
    override = {"lr": 2e-6, "betas": (0.9, 0.99), "weight_decay": 0.0}
    engine = _engine(False, override)
    engine._build_optimizer(SimpleNamespace())

    (kwargs,) = build_optimizer_calls
    assert {key: kwargs[key] for key in override} == override

    scheduler_calls = []
    monkeypatch.setattr(transformer_impl, "build_lr_scheduler", lambda optimizer, **kw: scheduler_calls.append(kw))
    engine._build_lr_scheduler(SimpleNamespace())
    (scheduler_kwargs,) = scheduler_calls
    assert scheduler_kwargs["lr"] == 2e-6
