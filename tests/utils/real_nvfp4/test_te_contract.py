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

from types import SimpleNamespace

import pytest

from verl.utils.real_nvfp4 import config


def _recipe(**overrides):
    qparams = SimpleNamespace(
        random_hadamard_transform=False,
        stochastic_rounding=False,
        fp4_2d_quantization=False,
    )
    values = {
        "backward_override": "dequantized",
        "row_scaled_activation": True,
        "disable_rht": True,
        "disable_stochastic_rounding": True,
        "disable_2d_quantization": True,
        "nvfp4_4over6": "none",
        "nvfp4_4over6_e4m3_use_256": "all",
        "nvfp4_4over6_err_mode": "MAE",
        "fp4_quant_fwd_inp": qparams,
        "fp4_quant_fwd_weight": qparams,
        "fp4_quant_bwd_grad": qparams,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_te_recipe_contract_accepts_exact_audited_semantics(monkeypatch):
    monkeypatch.setattr(config, "version", lambda _name: config.REAL_NVFP4_TE_VERSION)

    config.validate_real_nvfp4_te_recipe(
        _recipe(),
        backward_override="dequantized",
    )


@pytest.mark.parametrize("package", ["transformer-engine", "transformer-engine-cu13", "transformer-engine-torch"])
@pytest.mark.parametrize("bad_version", ["2.16.1", "2.18.0+e7c550c5", "2.19.0"])
def test_te_recipe_contract_rejects_mixed_or_unvalidated_release(monkeypatch, package, bad_version):
    monkeypatch.setattr(
        config, "version", lambda name: bad_version if name == package else config.REAL_NVFP4_TE_VERSION
    )

    with pytest.raises(RuntimeError, match="matched Transformer Engine release packages"):
        config.validate_real_nvfp4_te_recipe(
            _recipe(),
            backward_override="dequantized",
        )


def test_te_recipe_contract_rejects_4over6_train_rollout_mismatch(monkeypatch):
    monkeypatch.setattr(config, "version", lambda _name: config.REAL_NVFP4_TE_VERSION)

    with pytest.raises(RuntimeError, match="recipe drifted"):
        config.validate_real_nvfp4_te_recipe(
            _recipe(nvfp4_4over6="all"),
            backward_override="dequantized",
        )


def test_te_recipe_contract_rejects_quantizer_drift(monkeypatch):
    monkeypatch.setattr(config, "version", lambda _name: config.REAL_NVFP4_TE_VERSION)
    bad_qparams = SimpleNamespace(
        random_hadamard_transform=True,
        stochastic_rounding=False,
        fp4_2d_quantization=False,
    )

    with pytest.raises(RuntimeError, match="quantizer contract drifted"):
        config.validate_real_nvfp4_te_recipe(
            _recipe(fp4_quant_fwd_inp=bad_qparams),
            backward_override="dequantized",
        )
