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
"""Export kwargs for the per-rank expert LoRA slice of the colocated rollout sync (rollout.lora_expert_slice)."""

from types import SimpleNamespace

import pytest

from verl.workers.engine_workers import _lora_expert_slice_kwargs


def _rollout(ep=None, tp=None):
    return SimpleNamespace(lora_expert_slice=lambda: ep, lora_expert_tp_slice=lambda: tp)


def _engine(ep=True, tp=True):
    return SimpleNamespace(supports_lora_expert_slice=ep, supports_lora_expert_tp_slice=tp)


def test_off_when_disabled():
    assert _lora_expert_slice_kwargs(_rollout(ep=(1, 8)), _engine(), False) == {}


def test_ep_slice():
    assert _lora_expert_slice_kwargs(_rollout(ep=(3, 8)), _engine(), True) == {"lora_expert_slice": (3, 8)}


def test_tp_slice_without_ep():
    assert _lora_expert_slice_kwargs(_rollout(tp=(5, 8)), _engine(), True) == {"lora_expert_tp_slice": (5, 8)}


def test_ep_slice_takes_precedence():
    assert _lora_expert_slice_kwargs(_rollout(ep=(1, 8), tp=(1, 8)), _engine(), True) == {"lora_expert_slice": (1, 8)}


@pytest.mark.parametrize(
    "rollout, engine",
    [
        (_rollout(ep=(1, 8)), _engine(ep=False)),  # engine cannot export EP blocks
        (_rollout(tp=(1, 8)), _engine(tp=False)),  # engine cannot export TP shards
        (_rollout(), _engine()),  # no EP, TP1, PP or PD rollout layouts
        (SimpleNamespace(), _engine()),  # rollout without placement queries
        (_rollout(ep=(1, 8)), SimpleNamespace()),  # engine without the capability flags
    ],
)
def test_whole_adapter_otherwise(rollout, engine):
    assert _lora_expert_slice_kwargs(rollout, engine, True) == {}
