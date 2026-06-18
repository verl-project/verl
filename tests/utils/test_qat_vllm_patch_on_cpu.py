# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import importlib
import sys
import types

import pytest


@pytest.mark.parametrize(
    ("vllm_version", "expects_is_a_8bit"),
    [
        ("0.11.1", False),
        ("0.12.0", True),
        ("0.13.0", True),
    ],
)
def test_gptq_marlin_repack_version_compatible_is_a_8bit(monkeypatch, vllm_version, expects_is_a_8bit):
    calls = []

    class FakeOps:
        @staticmethod
        def gptq_marlin_repack(**kwargs):
            calls.append(kwargs)
            return "packed"

    fake_vllm = types.SimpleNamespace(__version__=vllm_version)
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    vllm_patch = importlib.import_module("verl.utils.qat.vllm_patch")

    b_q_weight = object()
    perm = object()
    result = vllm_patch._gptq_marlin_repack(
        FakeOps,
        b_q_weight=b_q_weight,
        perm=perm,
        size_k=128,
        size_n=256,
        num_bits=4,
        is_a_8bit=False,
    )

    assert result == "packed"
    assert len(calls) == 1
    assert calls[0]["b_q_weight"] is b_q_weight
    assert calls[0]["perm"] is perm
    assert calls[0]["size_k"] == 128
    assert calls[0]["size_n"] == 256
    assert calls[0]["num_bits"] == 4
    assert ("is_a_8bit" in calls[0]) is expects_is_a_8bit
    if expects_is_a_8bit:
        assert calls[0]["is_a_8bit"] is False


@pytest.mark.parametrize(
    ("vllm_version", "expects_is_a_8bit"),
    [
        ("0.11.1", False),
        ("0.12.0", True),
        ("0.13.0", True),
    ],
)
def test_marlin_permute_scales_version_compatible_is_a_8bit(monkeypatch, vllm_version, expects_is_a_8bit):
    calls = []

    def fake_marlin_permute_scales(**kwargs):
        calls.append(kwargs)
        return "permuted"

    fake_vllm = types.SimpleNamespace(__version__=vllm_version)
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    vllm_patch = importlib.import_module("verl.utils.qat.vllm_patch")

    scale = object()
    result = vllm_patch._marlin_permute_scales(
        fake_marlin_permute_scales,
        s=scale,
        size_k=128,
        size_n=256,
        group_size=16,
        is_a_8bit=False,
    )

    assert result == "permuted"
    assert len(calls) == 1
    assert calls[0]["s"] is scale
    assert calls[0]["size_k"] == 128
    assert calls[0]["size_n"] == 256
    assert calls[0]["group_size"] == 16
    assert ("is_a_8bit" in calls[0]) is expects_is_a_8bit
    if expects_is_a_8bit:
        assert calls[0]["is_a_8bit"] is False
