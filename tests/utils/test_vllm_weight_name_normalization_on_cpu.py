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

import torch

from verl.workers.rollout.vllm_rollout.utils import VLLM_LORA_INT_ID, vLLMColocateWorkerExtension


class _FakeMapper:
    def __init__(self, mapping: dict[str, str]):
        self.mapping = mapping

    def apply_list(self, names: list[str]) -> list[str]:
        return [self.mapping.get(name, name) for name in names]


class _FakeModel:
    def __init__(self):
        self.hf_to_vllm_mapper = _FakeMapper(
            {
                "model.language_model.layers.0.mlp.experts.base_layer.w13_weight": (
                    "language_model.model.layers.0.mlp.experts.base_layer.w13_weight"
                ),
                "model.language_model.layers.0.mlp.experts.base_layer.w2_weight": (
                    "language_model.model.layers.0.mlp.experts.base_layer.w2_weight"
                ),
                "model.language_model.layers.0.self_attn.qkv_proj.base_layer.weight": (
                    "language_model.model.layers.0.self_attn.qkv_proj.base_layer.weight"
                ),
            }
        )
        self.packed_modules_mapping = {
            "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        }

    def named_parameters(self, remove_duplicate: bool = False):
        del remove_duplicate
        yield "language_model.model.layers.0.mlp.experts.base_layer.w13_weight", torch.empty(0)
        yield "language_model.model.layers.0.mlp.experts.base_layer.w2_weight", torch.empty(0)
        yield "language_model.model.layers.0.self_attn.qkv_proj.base_layer.weight", torch.empty(0)


def _make_worker(model):
    worker = object.__new__(vLLMColocateWorkerExtension)
    worker.model_runner = SimpleNamespace(model=model)
    return worker


def test_normalize_base_sync_weight_names_preserves_expert_logical_aliases():
    worker = _make_worker(_FakeModel())
    tensor = torch.empty(0)

    normalized_weights = worker._normalize_base_sync_weight_names(
        [
            ("model.language_model.layers.0.mlp.experts.gate_up_proj", tensor),
            ("model.language_model.layers.0.mlp.experts.down_proj", tensor),
            ("model.language_model.layers.0.self_attn.q_proj.weight", tensor),
        ]
    )

    assert [name for name, _ in normalized_weights] == [
        "model.language_model.layers.0.mlp.experts.gate_up_proj",
        "model.language_model.layers.0.mlp.experts.down_proj",
        "model.language_model.layers.0.self_attn.q_proj.base_layer.weight",
    ]


def test_normalize_base_sync_weight_names_handles_bridge_inserted_base_layer_on_fused_experts():
    worker = _make_worker(_FakeModel())
    tensor = torch.empty(0)

    normalized_weights = worker._normalize_base_sync_weight_names(
        [
            ("model.language_model.layers.0.mlp.experts.base_layer.gate_up_proj", tensor),
            ("model.language_model.layers.0.mlp.experts.base_layer.down_proj", tensor),
        ]
    )

    assert [name for name, _ in normalized_weights] == [
        "model.language_model.layers.0.mlp.experts.gate_up_proj",
        "model.language_model.layers.0.mlp.experts.down_proj",
    ]


def test_normalize_base_sync_weight_names_handles_fused_expert_leaf_params():
    worker = _make_worker(_FakeModel())
    tensor = torch.empty(0)

    normalized_weights = worker._normalize_base_sync_weight_names(
        [
            ("model.language_model.layers.0.mlp.experts.w13_weight", tensor),
            ("model.language_model.layers.0.mlp.experts.base_layer.w2_weight", tensor),
        ]
    )

    assert [name for name, _ in normalized_weights] == [
        "model.language_model.layers.0.mlp.experts.base_layer.w13_weight",
        "model.language_model.layers.0.mlp.experts.base_layer.w2_weight",
    ]


def test_update_weights_from_ipc_accumulates_lora_tensors_across_buckets(monkeypatch):
    import verl.workers.rollout.vllm_rollout.bucketed_weight_transfer as bucketed_weight_transfer

    class _FakeBucketReceiver:
        def __init__(self, zmq_handle, device, use_shm):
            del zmq_handle, device, use_shm

        def receive_weights(self, on_bucket_received):
            on_bucket_received(
                [("layers.0.self_attn.q_proj.lora_A.weight", torch.ones(1))],
                is_last=False,
            )
            on_bucket_received(
                [("layers.0.self_attn.q_proj.lora_B.weight", torch.zeros(1))],
                is_last=True,
            )

    monkeypatch.setattr(bucketed_weight_transfer, "BucketedWeightReceiver", _FakeBucketReceiver)

    worker = _make_worker(_FakeModel())
    worker.model_runner.vllm_config = SimpleNamespace()
    worker.device = torch.device("cpu")
    worker.local_rank = 0
    worker._is_qat_model = False
    worker._get_zmq_handle = lambda: "ipc:///tmp/test-bucketed-lora.sock"

    removed_loras = []
    added_requests = []
    worker.remove_lora = removed_loras.append

    def _add_lora(lora_request):
        added_requests.append(lora_request)
        return True

    worker.add_lora = _add_lora

    worker.update_weights_from_ipc(peft_config={"r": 1}, base_sync_done=True)

    assert removed_loras == [VLLM_LORA_INT_ID]
    assert len(added_requests) == 1
    assert set(added_requests[0].lora_tensors) == {
        "layers.0.self_attn.q_proj.lora_A.weight",
        "layers.0.self_attn.q_proj.lora_B.weight",
    }
