# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
"""Tests for the precision-aware dispatch in ``init_megatron_optim_config``.

These tests stub out ``megatron.core.optimizer.OptimizerConfig`` so they can
run on CPU without TransformerEngine — the goal is to verify which kwargs
verl assembles for each precision mode, not Megatron's downstream validation.

The precision-aware optimizer is opt-in: the bf16 branch keeps the fp32
optimizer state unless ``use_precision_aware_optimizer`` is set on the config,
at which point the moment / grad dtypes follow the configured fields.
"""

from unittest.mock import MagicMock

import pytest
import torch
from omegaconf import OmegaConf

from verl.utils.megatron import optimizer as opt_mod
from verl.utils.megatron.optimizer import init_megatron_optim_config


def _base_optim_config(**overrides):
    cfg = {
        "optimizer": "adam",
        "lr": 1e-3,
        "min_lr": 0.0,
        "clip_grad": 1.0,
        "weight_decay": 0.01,
    }
    cfg.update(overrides)
    return OmegaConf.create(cfg)


def _precision_aware_optim_config(**overrides):
    """bf16 optimizer state explicitly opted in via the new config flags."""
    fields = {
        "use_precision_aware_optimizer": True,
        "main_grads_dtype": "bf16",
        "exp_avg_dtype": "bf16",
        "exp_avg_sq_dtype": "bf16",
    }
    fields.update(overrides)
    return _base_optim_config(**fields)


@pytest.fixture
def captured_args(monkeypatch):
    """Replace ``OptimizerConfig`` with a recorder so we can inspect kwargs."""
    captured: dict = {}

    def _fake(**kwargs):
        captured.clear()
        captured.update(kwargs)
        return MagicMock(name="OptimizerConfig", **kwargs)

    monkeypatch.setattr(opt_mod, "OptimizerConfig", _fake)
    return captured


def test_bf16_branch_defaults_to_fp32_optimizer_state(captured_args):
    """Opt-out default: bf16 params, but the precision-aware optimizer stays off."""
    init_megatron_optim_config(_base_optim_config(), fp16=False, bf16=True)

    assert captured_args["bf16"] is True
    assert captured_args["params_dtype"] is torch.bfloat16
    # No precision-aware optimizer and no sub-fp32 moment / grad dtypes by default.
    assert "use_precision_aware_optimizer" not in captured_args
    assert "main_grads_dtype" not in captured_args
    assert "exp_avg_dtype" not in captured_args
    assert "exp_avg_sq_dtype" not in captured_args


def test_bf16_branch_opt_in_enables_precision_aware_with_bf16_state(captured_args):
    init_megatron_optim_config(_precision_aware_optim_config(), fp16=False, bf16=True)

    assert captured_args["bf16"] is True
    assert captured_args["params_dtype"] is torch.bfloat16
    assert captured_args["use_precision_aware_optimizer"] is True
    assert captured_args["main_grads_dtype"] is torch.bfloat16
    assert captured_args["exp_avg_dtype"] is torch.bfloat16
    assert captured_args["exp_avg_sq_dtype"] is torch.bfloat16
    # Master params dtype intentionally left at Megatron default (fp32) —
    # TE FusedAdam rejects bf16 master at init.
    assert "main_params_dtype" not in captured_args


def test_bf16_opt_in_respects_per_field_dtypes(captured_args):
    """Per-flag control: opting in but pinning a moment to fp32 is honored."""
    cfg = _precision_aware_optim_config(exp_avg_sq_dtype="fp32")
    init_megatron_optim_config(cfg, fp16=False, bf16=True)

    assert captured_args["use_precision_aware_optimizer"] is True
    assert captured_args["main_grads_dtype"] is torch.bfloat16
    assert captured_args["exp_avg_dtype"] is torch.bfloat16
    assert captured_args["exp_avg_sq_dtype"] is torch.float32


def test_fp16_branch_uses_precision_aware_but_keeps_fp32_optimizer_state(captured_args):
    init_megatron_optim_config(_base_optim_config(), fp16=True, bf16=False)

    assert captured_args["fp16"] is True
    assert captured_args["bf16"] is False
    assert captured_args["params_dtype"] is torch.float16
    assert captured_args["use_precision_aware_optimizer"] is True
    assert captured_args["initial_loss_scale"] == 32768
    assert captured_args["min_loss_scale"] == 1
    assert captured_args["store_param_remainders"] is False
    # Adam moment / grad dtypes left at Megatron's fp32 default in fp16 mode.
    assert "main_grads_dtype" not in captured_args
    assert "exp_avg_dtype" not in captured_args
    assert "exp_avg_sq_dtype" not in captured_args


def test_fp32_branch_disables_precision_aware_optimizer(captured_args):
    init_megatron_optim_config(_base_optim_config(), fp16=False, bf16=False)

    assert captured_args["fp16"] is False
    assert captured_args["bf16"] is False
    assert captured_args["params_dtype"] is torch.float32
    # Precision-aware optimizer must stay off — Megatron asserts the dtype
    # fields equal fp32 when it's disabled.
    assert "use_precision_aware_optimizer" not in captured_args
    assert "main_grads_dtype" not in captured_args
    assert "exp_avg_dtype" not in captured_args
    assert "exp_avg_sq_dtype" not in captured_args


def test_default_kwargs_dispatch_to_bf16_branch(captured_args):
    """Backward compatibility: callers that omit ``bf16`` get the bf16 path (fp32 state)."""
    init_megatron_optim_config(_base_optim_config())

    assert captured_args["bf16"] is True
    assert captured_args["params_dtype"] is torch.bfloat16
    # Opt-in default keeps the precision-aware optimizer off.
    assert "use_precision_aware_optimizer" not in captured_args


def test_fp16_wins_over_bf16_when_both_true(captured_args):
    init_megatron_optim_config(_precision_aware_optim_config(), fp16=True, bf16=True)

    assert captured_args["fp16"] is True
    assert captured_args["params_dtype"] is torch.float16
    # bf16-branch-only fields must not appear when fp16 is selected, even when
    # the precision-aware config flags are present.
    assert "main_grads_dtype" not in captured_args
    assert "exp_avg_dtype" not in captured_args


def test_use_distributed_optimizer_passes_through(captured_args):
    init_megatron_optim_config(_base_optim_config(), use_distributed_optimizer=False)
    assert captured_args["use_distributed_optimizer"] is False

    init_megatron_optim_config(_base_optim_config(), use_distributed_optimizer=True)
    assert captured_args["use_distributed_optimizer"] is True


def test_basic_optim_config_fields_pass_through(captured_args):
    cfg = _base_optim_config(optimizer="sgd", lr=5e-4, min_lr=1e-5, clip_grad=0.5, weight_decay=0.1)
    init_megatron_optim_config(cfg)

    assert captured_args["optimizer"] == "sgd"
    assert captured_args["lr"] == pytest.approx(5e-4)
    assert captured_args["min_lr"] == pytest.approx(1e-5)
    assert captured_args["clip_grad"] == pytest.approx(0.5)
    assert captured_args["weight_decay"] == pytest.approx(0.1)


def test_override_optimizer_config_overrides_branch_defaults(captured_args):
    cfg = _precision_aware_optim_config(
        override_optimizer_config={
            "use_precision_aware_optimizer": False,
            "exp_avg_dtype": "sentinel-override",
        },
    )
    init_megatron_optim_config(cfg, bf16=True)

    # User-supplied overrides win over the opted-in bf16 defaults …
    assert captured_args["use_precision_aware_optimizer"] is False
    assert captured_args["exp_avg_dtype"] == "sentinel-override"
    # … but non-overridden bf16 defaults remain.
    assert captured_args["main_grads_dtype"] is torch.bfloat16
    assert captured_args["exp_avg_sq_dtype"] is torch.bfloat16


def test_missing_override_config_leaves_branch_defaults_intact(captured_args):
    """``optim_config.get('override_optimizer_config', {})`` must not crash when absent."""
    cfg = _precision_aware_optim_config()
    assert "override_optimizer_config" not in cfg

    init_megatron_optim_config(cfg, bf16=True)

    assert captured_args["use_precision_aware_optimizer"] is True
    assert captured_args["exp_avg_dtype"] is torch.bfloat16
