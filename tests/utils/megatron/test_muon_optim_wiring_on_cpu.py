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

"""CPU wiring test for verl -> Megatron Muon optimizer passthrough.

Megatron-Core is not importable in the lightweight CPU CI image, so we stub the two
``megatron.core`` modules that ``verl.utils.megatron.optimizer`` imports. This lets us assert
verl's own logic: that ``init_megatron_optim_config`` forwards the ``muon_*`` hyperparameters onto
the Megatron ``OptimizerConfig`` when ``optimizer=muon``, only forwards fields the installed
``OptimizerConfig`` declares, and fails loudly (rather than silently falling back to Adam) when the
installed Megatron exposes no Muon fields at all.
"""

import dataclasses
import importlib
import sys
import types

import pytest


def _install_stub_megatron(optimizer_config_fields):
    """Register stub ``megatron.core`` optimizer modules exposing ``optimizer_config_fields``.

    Returns the freshly (re)imported ``verl.utils.megatron.optimizer`` module bound to the stubs.
    """
    field_defaults = {
        "optimizer": "adam",
        "lr": 0.0,
        "min_lr": 0.0,
        "clip_grad": 1.0,
        "weight_decay": 0.0,
        "use_distributed_optimizer": True,
        "bf16": False,
        "fp16": False,
        "params_dtype": None,
        "initial_loss_scale": 1,
        "min_loss_scale": 1,
        "use_precision_aware_optimizer": False,
        "store_param_remainders": False,
    }
    # Muon knobs the stub Megatron "supports" for this test run.
    for name in optimizer_config_fields:
        field_defaults.setdefault(name, None)

    OptimizerConfig = dataclasses.make_dataclass(
        "OptimizerConfig",
        [(k, object, v) for k, v in field_defaults.items()],
    )

    opt_mod = types.ModuleType("megatron.core.optimizer")
    opt_mod.OptimizerConfig = OptimizerConfig
    opt_mod.get_megatron_optimizer = lambda config, model_chunks: ("optimizer", config)

    sched_mod = types.ModuleType("megatron.core.optimizer_param_scheduler")
    sched_mod.OptimizerParamScheduler = object

    core_mod = types.ModuleType("megatron.core")
    root_mod = types.ModuleType("megatron")

    sys.modules["megatron"] = root_mod
    sys.modules["megatron.core"] = core_mod
    sys.modules["megatron.core.optimizer"] = opt_mod
    sys.modules["megatron.core.optimizer_param_scheduler"] = sched_mod

    sys.modules.pop("verl.utils.megatron.optimizer", None)
    return importlib.import_module("verl.utils.megatron.optimizer")


_ALL_MUON_FIELDS = (
    "use_layer_wise_distributed_optimizer",
    "muon_momentum",
    "muon_nesterov",
    "muon_split_qkv",
    "muon_scale_mode",
    "muon_coefficient_type",
    "muon_num_ns_steps",
    "muon_tp_mode",
    "muon_fp32_matmul_prec",
    "muon_extra_scale_factor",
    "muon_scalar_optimizer",
)


@pytest.fixture
def muon_config():
    from verl.workers.config.optimizer import McoreOptimizerConfig

    return McoreOptimizerConfig(
        lr=1e-3,
        optimizer="muon",
        muon_momentum=0.9,
        muon_num_ns_steps=6,
        muon_tp_mode="allgather",
        muon_scalar_optimizer="lion",
        use_layer_wise_distributed_optimizer=True,
    )


def test_muon_fields_forwarded(muon_config):
    mod = _install_stub_megatron(_ALL_MUON_FIELDS)
    cfg = mod.init_megatron_optim_config(muon_config, use_distributed_optimizer=True, fp16=False)
    assert cfg.optimizer == "muon"
    assert cfg.muon_momentum == 0.9
    assert cfg.muon_num_ns_steps == 6
    assert cfg.muon_tp_mode == "allgather"
    assert cfg.muon_scalar_optimizer == "lion"
    assert cfg.use_layer_wise_distributed_optimizer is True
    # untouched knobs keep their configured defaults
    assert cfg.muon_split_qkv is True


def test_only_supported_muon_fields_forwarded(muon_config):
    # Stub Megatron only knows a subset of the muon knobs; the rest must be dropped, not crash.
    mod = _install_stub_megatron(("muon_momentum", "muon_num_ns_steps"))
    cfg = mod.init_megatron_optim_config(muon_config, use_distributed_optimizer=True, fp16=False)
    assert cfg.muon_momentum == 0.9
    assert cfg.muon_num_ns_steps == 6
    assert not hasattr(cfg, "muon_tp_mode")


def test_adam_path_does_not_forward_muon(muon_config):
    from verl.workers.config.optimizer import McoreOptimizerConfig

    mod = _install_stub_megatron(_ALL_MUON_FIELDS)
    adam_config = McoreOptimizerConfig(lr=1e-3, optimizer="adam", muon_momentum=0.42)
    cfg = mod.init_megatron_optim_config(adam_config, use_distributed_optimizer=True, fp16=False)
    assert cfg.optimizer == "adam"
    # Muon knobs are left at Megatron's own default (None on the stub), never forwarded.
    assert cfg.muon_momentum is None


def test_muon_without_megatron_support_fails_loud(muon_config):
    # Installed Megatron exposes no muon fields -> must raise instead of silently building Adam.
    mod = _install_stub_megatron(())
    with pytest.raises(ValueError, match="emerging_optimizers"):
        mod.init_megatron_optim_config(muon_config, use_distributed_optimizer=True, fp16=False)
