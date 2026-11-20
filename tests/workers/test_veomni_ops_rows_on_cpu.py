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
"""Exercise the VeOmni engine's per-family ops-selection forwarding without VeOmni installed.

VeOmni is optional in CPU CI, so its import surfaces are stubbed the way the router-replay
engine tests do. What is checked here is how ``VeOmniEngineConfig`` selectors map onto the
installed VeOmni's ``OpsImplementationConfig``: every accepted field is forwarded, a field
the installed version does not know is skipped while it is still at the verl default, and
rejected when it was explicitly set.
"""

import sys
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

for _mod in (
    "veomni",
    "veomni.arguments",
    "veomni.distributed",
    "veomni.distributed.torch_parallelize",
    "veomni.models",
    "veomni.models.auto",
    "veomni.models.checkpoint",
    "veomni.optim",
    "veomni.utils",
    "veomni.utils.seqlen_pos_transform_utils",
):
    sys.modules.setdefault(_mod, MagicMock())

from verl.workers.config import VeOmniEngineConfig  # noqa: E402
from verl.workers.engine.veomni import transformer_impl  # noqa: E402


@dataclass
class _LegacyOpsImplementationConfig:
    """The selector surface before VeOmni moved QAT to per-family rows."""

    attn_implementation: str | None = None
    moe_implementation: str = "eager"
    swiglu_mlp_implementation: str = "eager"
    qat_implementation: str = "none"


@dataclass
class _RowOpsImplementationConfig:
    """The selector surface after VeOmni moved QAT to per-family rows."""

    attn_implementation: str | None = None
    moe_implementation: str = "eager"
    swiglu_mlp_implementation: str = "eager"
    linear_implementation: str = "eager"
    kv_round_implementation: str = "none"


def test_qat_rows_are_forwarded_per_family(monkeypatch):
    monkeypatch.setattr(transformer_impl, "OpsImplementationConfig", _RowOpsImplementationConfig)
    engine_config = VeOmniEngineConfig(
        linear_implementation="qat",
        swiglu_mlp_implementation="qat",
        moe_implementation="qat",
        kv_round_implementation="triton",
    )

    ops = transformer_impl._build_ops_implementation_config(engine_config)

    assert ops.linear_implementation == "qat"
    assert ops.swiglu_mlp_implementation == "qat"
    assert ops.moe_implementation == "qat"
    assert ops.kv_round_implementation == "triton"


def test_qat_rows_fail_fast_on_an_older_veomni(monkeypatch):
    monkeypatch.setattr(transformer_impl, "OpsImplementationConfig", _LegacyOpsImplementationConfig)
    engine_config = VeOmniEngineConfig(linear_implementation="qat")

    with pytest.raises(ValueError, match="linear_implementation"):
        transformer_impl._build_ops_implementation_config(engine_config)
