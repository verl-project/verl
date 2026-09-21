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

"""Configuration and BF16 transport checks for native NVFP4 refit."""

from types import SimpleNamespace

import pytest
import torch

from verl.utils.real_nvfp4 import vllm_runtime
from verl.utils.real_nvfp4.bf16_transport import attest_real_nvfp4_bf16_transport


def test_native_quantization_configuration():
    vllm_runtime.require_vllm_native_nvfp4_per_token(
        SimpleNamespace(model_config=SimpleNamespace(quantization="nvfp4_per_token"))
    )
    with pytest.raises(RuntimeError, match="quantization drifted"):
        vllm_runtime.require_vllm_native_nvfp4_per_token(
            SimpleNamespace(model_config=SimpleNamespace(quantization="fp8"))
        )


def test_bf16_native_reload_refuses_mtp_before_ipc(monkeypatch):
    from verl.workers.rollout.vllm_rollout import bucketed_weight_transfer, utils

    monkeypatch.setenv("VERL_VLLM_NATIVE_RELOAD", "1")
    monkeypatch.setattr(torch.version, "hip", None)

    def forbidden_receiver(*args, **kwargs):
        pytest.fail("MTP must be rejected before opening an IPC receiver")

    monkeypatch.setattr(bucketed_weight_transfer, "BucketedWeightReceiver", forbidden_receiver)
    worker = SimpleNamespace(
        device=torch.device("cpu"),
        _is_real_nvfp4=False,
        _use_mtp_drafter_weight_sync=lambda: True,
    )
    with pytest.raises(NotImplementedError, match="MTP drafter"):
        utils.vLLMColocateWorkerExtension.update_weights_from_ipc(worker)


def _weights():
    return [
        (f"model.layers.{layer}.mlp.experts.{expert}.{projection}.weight", torch.ones(2, 16))
        for layer in range(2)
        for expert in range(2)
        for projection in ("gate_proj", "up_proj", "down_proj")
    ]


def _attest(weights):
    return list(
        attest_real_nvfp4_bf16_transport(
            iter(weights),
            expected_expert_weights=12,
            hf_config={"num_hidden_layers": 2, "num_experts": 2},
        )
    )


def test_expert_coverage_accepts_complete_reordered_stream():
    weights = list(reversed(_weights()))
    weights.insert(0, ("model.embed_tokens.weight", torch.ones(2, 16)))
    assert len(_attest(weights)) == 13


def test_expert_coverage_rejects_duplicate_replacing_missing_projection():
    weights = _weights()
    weights[1] = weights[0]
    with pytest.raises(RuntimeError, match="duplicate expert weight"):
        _attest(weights)


def test_expert_coverage_rejects_missing_projection():
    with pytest.raises(RuntimeError, match="missing 1 expert weights"):
        _attest(_weights()[:-1])


@pytest.mark.parametrize(
    "name",
    [
        "model.layers.2.mlp.experts.0.gate_proj.weight",
        "model.layers.0.mlp.experts.2.gate_proj.weight",
        "other.layers.0.mlp.experts.0.gate_proj.weight",
    ],
)
def test_expert_coverage_rejects_wrong_layer_expert_or_prefix(name):
    weights = _weights()
    weights[0] = name, weights[0][1]
    with pytest.raises(RuntimeError, match="unexpected expert weight"):
        _attest(weights)


def test_count_only_transport_still_rejects_duplicates():
    weight = _weights()[0]
    with pytest.raises(RuntimeError, match="duplicate expert weight"):
        list(attest_real_nvfp4_bf16_transport(iter([weight, weight]), expected_expert_weights=2))
