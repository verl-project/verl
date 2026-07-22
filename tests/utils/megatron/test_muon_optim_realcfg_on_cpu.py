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

"""Real-Megatron (non-stub) CPU check for the verl -> Megatron-Core Muon wiring.

Unlike ``test_muon_optim_wiring_on_cpu.py`` (which stubs ``megatron.core`` so it can run in the
lightweight CI image), this test imports the *actual* ``megatron.core.optimizer`` and asserts that
verl's ``init_megatron_optim_config`` produces a genuine Megatron ``OptimizerConfig`` -- proving the
forwarded ``muon_*`` field names/types are accepted by the installed Megatron -- and that Megatron
would route it to the emerging (Muon) builder rather than Adam.

It skips automatically when the installed Megatron has no Muon support (older builds without
``emerging_optimizers``), so it is safe in any environment. The remaining full optimizer-object
build + ``optimizer.step()`` requires CUDA (Megatron's DDP grad buffers allocate on
``torch.cuda.current_device()``), so that part is covered by the GPU Slurm smoke, not here.
"""

import dataclasses

import pytest


def _real_megatron_optimizer_config():
    """Return the installed Megatron OptimizerConfig, or skip if it has no Muon fields."""
    try:
        from megatron.core.optimizer import OptimizerConfig
    except Exception as exc:  # pragma: no cover - env dependent
        pytest.skip(f"megatron.core.optimizer not importable: {exc}")
    field_names = {f.name for f in dataclasses.fields(OptimizerConfig)}
    if "muon_momentum" not in field_names:
        pytest.skip("installed Megatron OptimizerConfig has no Muon fields (no emerging_optimizers)")
    return OptimizerConfig


def test_verl_wiring_builds_real_megatron_muon_config():
    OptimizerConfig = _real_megatron_optimizer_config()

    from verl.utils.megatron.optimizer import _MUON_ALGORITHMS, init_megatron_optim_config
    from verl.workers.config.optimizer import McoreOptimizerConfig

    vcfg = McoreOptimizerConfig(
        lr=0.01,
        optimizer="muon",
        weight_decay=0.01,
        clip_grad=1.0,
        muon_momentum=0.9,
        muon_num_ns_steps=6,
        muon_tp_mode="duplicated",
        muon_scalar_optimizer="lion",
        use_layer_wise_distributed_optimizer=True,
    )
    mcore_cfg = init_megatron_optim_config(vcfg, use_distributed_optimizer=False, fp16=False)

    # Real Megatron OptimizerConfig accepted every forwarded knob (field names/types compatible).
    assert isinstance(mcore_cfg, OptimizerConfig)
    assert mcore_cfg.optimizer == "muon"
    assert mcore_cfg.muon_momentum == 0.9
    assert mcore_cfg.muon_num_ns_steps == 6
    assert mcore_cfg.muon_tp_mode == "duplicated"
    assert mcore_cfg.muon_scalar_optimizer == "lion"
    assert mcore_cfg.use_layer_wise_distributed_optimizer is True

    # Exact predicate Megatron's get_megatron_optimizer uses to pick the emerging (Muon) path.
    assert mcore_cfg.optimizer not in ("adam", "sgd"), "muon config would fall back to Adam/SGD path!"
    assert str(vcfg.optimizer).lower() in _MUON_ALGORITHMS


def test_verl_adam_path_leaves_real_megatron_config_standard():
    OptimizerConfig = _real_megatron_optimizer_config()

    from verl.utils.megatron.optimizer import init_megatron_optim_config
    from verl.workers.config.optimizer import McoreOptimizerConfig

    adam_cfg = init_megatron_optim_config(
        McoreOptimizerConfig(lr=0.01, optimizer="adam", muon_momentum=0.42),
        use_distributed_optimizer=False,
        fp16=False,
    )
    assert isinstance(adam_cfg, OptimizerConfig)
    assert adam_cfg.optimizer == "adam"
    # verl did NOT forward the muon knob; Megatron keeps its own default.
    assert adam_cfg.muon_momentum == OptimizerConfig().muon_momentum
    assert adam_cfg.optimizer in ("adam", "sgd")
