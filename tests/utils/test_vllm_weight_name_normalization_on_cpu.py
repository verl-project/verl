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

    def named_buffers(self):
        return iter(())


class _FakeExperts:
    def __init__(self, expert_map):
        self._expert_map = torch.tensor(expert_map, dtype=torch.int32)


class _FakeExpertModel(_FakeModel):
    def __init__(self, expert_map, mapper=None):
        super().__init__()
        if mapper is not None:
            self.hf_to_vllm_mapper = mapper
        self.experts = _FakeExperts(expert_map)

    def named_modules(self):
        yield "", self
        yield "model.layers.0.mlp.experts", self.experts


def _make_worker(model):
    worker = object.__new__(vLLMColocateWorkerExtension)
    worker.model_runner = SimpleNamespace(model=model)
    return worker


def test_normalize_base_sync_weight_names_preserves_expert_logical_aliases():
    worker = _make_worker(_FakeModel())
    tensor = torch.empty(0)

    normalized_weights = list(
        worker._iter_normalized_base_sync_weights(
            [
                ("model.language_model.layers.0.mlp.experts.gate_up_proj", tensor),
                ("model.language_model.layers.0.mlp.experts.down_proj", tensor),
                ("model.language_model.layers.0.self_attn.q_proj.weight", tensor),
            ]
        )
    )

    assert [name for name, _ in normalized_weights] == [
        "model.language_model.layers.0.mlp.experts.gate_up_proj",
        "model.language_model.layers.0.mlp.experts.down_proj",
        "model.language_model.layers.0.self_attn.q_proj.base_layer.weight",
    ]


def test_normalize_base_sync_weight_names_handles_bridge_inserted_base_layer_on_fused_experts():
    worker = _make_worker(_FakeModel())
    tensor = torch.empty(0)

    normalized_weights = list(
        worker._iter_normalized_base_sync_weights(
            [
                ("model.language_model.layers.0.mlp.experts.base_layer.gate_up_proj", tensor),
                ("model.language_model.layers.0.mlp.experts.base_layer.down_proj", tensor),
            ]
        )
    )

    assert [name for name, _ in normalized_weights] == [
        "model.language_model.layers.0.mlp.experts.gate_up_proj",
        "model.language_model.layers.0.mlp.experts.down_proj",
    ]


def test_layerwise_reload_skips_nonlocal_expert_weights_before_clone():
    mapper = _FakeMapper(
        {
            "model.language_model.layers.0.mlp.experts.1.down_proj.weight": (
                "model.layers.0.mlp.experts.1.down_proj.weight"
            )
        }
    )
    worker = _make_worker(_FakeExpertModel([0, -1], mapper=mapper))
    local_expert = torch.tensor([1.0])
    nonlocal_expert = torch.tensor([2.0])
    dense_weight = torch.tensor([3.0])

    normalized_weights = list(
        worker._iter_normalized_base_sync_weights(
            [
                ("model.layers.0.mlp.experts.0.gate_proj.weight", local_expert),
                ("model.layers.0.mlp.experts.1.gate_proj.weight", nonlocal_expert),
                ("model.language_model.layers.0.mlp.experts.1.down_proj.weight", nonlocal_expert),
                ("model.layers.0.self_attn.q_proj.weight", dense_weight),
            ],
            clone_tensors=True,
        )
    )

    assert [name for name, _ in normalized_weights] == [
        "model.layers.0.mlp.experts.0.gate_proj.weight",
        "model.layers.0.self_attn.q_proj.weight",
    ]
    assert normalized_weights[0][1].data_ptr() != local_expert.data_ptr()
    assert normalized_weights[1][1].data_ptr() != dense_weight.data_ptr()


def test_layerwise_reload_expert_filter_fails_open_for_unknown_weights():
    worker = _make_worker(_FakeExpertModel([0, -1]))
    unknown_expert = torch.tensor([1.0])
    expert_scale = torch.tensor([2.0])

    normalized_weights = list(
        worker._iter_normalized_base_sync_weights(
            [
                ("model.layers.1.mlp.experts.1.gate_proj.weight", unknown_expert),
                ("model.layers.0.mlp.experts.1.gate_proj.weight_scale", expert_scale),
            ],
            clone_tensors=True,
        )
    )

    assert [name for name, _ in normalized_weights] == [
        "model.layers.1.mlp.experts.1.gate_proj.weight",
        "model.layers.0.mlp.experts.1.gate_proj.weight_scale",
    ]
    assert normalized_weights[0][1].data_ptr() != unknown_expert.data_ptr()
    assert normalized_weights[1][1].data_ptr() != expert_scale.data_ptr()


def test_nonlocal_expert_filter_only_applies_to_layerwise_clone_path():
    worker = _make_worker(_FakeExpertModel([0, -1]))
    nonlocal_expert = torch.tensor([1.0])

    normalized_weights = list(
        worker._iter_normalized_base_sync_weights(
            [("model.layers.0.mlp.experts.1.gate_proj.weight", nonlocal_expert)],
            clone_tensors=False,
        )
    )

    assert len(normalized_weights) == 1
    assert normalized_weights[0][1] is nonlocal_expert


def test_update_weights_from_ipc_accumulates_lora_tensors_across_buckets(monkeypatch):
    import verl.workers.rollout.vllm_rollout.bucketed_weight_transfer as bucketed_weight_transfer

    class _FakeBucketReceiver:
        def __init__(self, zmq_handle, device, use_shm):
            del zmq_handle, device, use_shm

        def receive_weights(self, on_bucket_received):
            shared_bucket_tensor = torch.ones(1)
            on_bucket_received([("layers.0.self_attn.q_proj.lora_A.weight", shared_bucket_tensor)])
            shared_bucket_tensor.fill_(7)
            on_bucket_received([("layers.0.self_attn.q_proj.lora_B.weight", torch.zeros(1))])

    monkeypatch.setattr(bucketed_weight_transfer, "BucketedWeightReceiver", _FakeBucketReceiver)

    worker = _make_worker(_FakeModel())
    worker.model_runner.vllm_config = SimpleNamespace(speculative_config=None)
    worker.device = torch.device("cpu")
    worker.local_rank = 0
    worker._is_qat_model = False
    worker._is_modelopt_qat = False
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
    torch.testing.assert_close(
        added_requests[0].lora_tensors["layers.0.self_attn.q_proj.lora_A.weight"],
        torch.ones(1),
    )


def test_maybe_reload_standard_weights_falls_back_for_mtp():
    class _UnusedReceiver:
        def iter_weights(self):
            raise AssertionError("MTP fallback must not consume the streaming receiver")

    worker = _make_worker(_FakeModel())
    worker.model_runner.vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(method="mtp", draft_model_config=object())
    )
    worker.model_runner.drafter = SimpleNamespace(model=_FakeModel())

    def _unexpected_reload_weights(**kwargs):
        raise AssertionError(f"MTP fallback must not call reload_weights: {kwargs}")

    worker.reload_weights = _unexpected_reload_weights

    assert worker._maybe_reload_standard_weights_from_ipc(_UnusedReceiver()) is False


def test_maybe_reload_standard_weights_falls_back_without_reload_api():
    """Platform workers without reload_weights (e.g. vllm-ascend NPUWorker) use the bucketed path."""

    class _UnusedReceiver:
        def iter_weights(self):
            raise AssertionError("fallback must not consume the streaming receiver")

    worker = _make_worker(_FakeModel())
    worker.model_runner.vllm_config = SimpleNamespace(speculative_config=None)
    assert not hasattr(worker, "reload_weights")

    assert worker._maybe_reload_standard_weights_from_ipc(_UnusedReceiver()) is False


def test_update_weights_normalizes_base_layer_names_before_fp8(monkeypatch):
    import verl.workers.rollout.vllm_rollout.utils as worker_utils

    worker = _make_worker(_FakeModel())
    worker.model_runner.vllm_config = SimpleNamespace(speculative_config=None, quant_config=object())
    loaded_weights = []

    monkeypatch.setattr(worker_utils, "is_fp8_model", lambda _config: True)

    def _load_quanted_weights(weights, _model_runner, **_kwargs):
        loaded_weights.extend(weights)
        return []

    monkeypatch.setattr(worker_utils, "load_quanted_weights", _load_quanted_weights)

    worker._update_weights(
        [("model.language_model.layers.0.self_attn.q_proj.weight", torch.empty(0))],
        peft_config={"r": 1},
        base_sync_done=False,
    )

    assert [name for name, _ in loaded_weights] == ["model.language_model.layers.0.self_attn.q_proj.base_layer.weight"]


def test_update_weights_from_ipc_uses_reload_weights_stream_for_standard_base_sync(monkeypatch):
    import verl.workers.rollout.vllm_rollout.bucketed_weight_transfer as bucketed_weight_transfer
    import verl.workers.rollout.vllm_rollout.utils as worker_utils

    class _StreamingBucketReceiver:
        def __init__(self, zmq_handle, device, use_shm):
            del zmq_handle, device, use_shm

        def iter_weights(self):
            shared_tensor = torch.ones(1)
            yield ("model.language_model.layers.0.self_attn.q_proj.weight", shared_tensor)
            shared_tensor.fill_(7)
            yield ("model.language_model.layers.0.mlp.experts.base_layer.down_proj", torch.zeros(1))

        def receive_weights(self, on_bucket_received):
            raise AssertionError("standard base sync should use reload_weights streaming path")

    monkeypatch.setattr(bucketed_weight_transfer, "BucketedWeightReceiver", _StreamingBucketReceiver)
    monkeypatch.setattr(worker_utils, "patch_vllm_moe_model_weight_loader", lambda model: None)

    worker = _make_worker(_FakeModel())
    worker.model_runner.vllm_config = SimpleNamespace(speculative_config=None)
    worker.device = torch.device("cpu")
    worker.local_rank = 0
    worker._is_qat_model = False
    worker._is_modelopt_qat = False
    worker._get_zmq_handle = lambda: "ipc:///tmp/test-streaming-reload.sock"

    reloaded_weights = []

    def _reload_weights(*, weights_iterator, is_checkpoint_format):
        reloaded_weights.extend(list(weights_iterator))
        assert is_checkpoint_format is True

    worker.reload_weights = _reload_weights

    worker.update_weights_from_ipc(peft_config=None, base_sync_done=True)

    assert [name for name, _ in reloaded_weights] == [
        "model.language_model.layers.0.self_attn.q_proj.base_layer.weight",
        "model.language_model.layers.0.mlp.experts.down_proj",
    ]
    torch.testing.assert_close(reloaded_weights[0][1], torch.ones(1))
    torch.testing.assert_close(reloaded_weights[1][1], torch.zeros(1))
