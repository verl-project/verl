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

import importlib.util
import sys
import types
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_weight_update_utils():
    module_path = _REPO_ROOT / "verl/workers/rollout/vllm_rollout/weight_update_utils.py"
    spec = importlib.util.spec_from_file_location("weight_update_utils", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


_weight_update_utils = _load_weight_update_utils()
apply_buffer_updates = _weight_update_utils.apply_buffer_updates
split_buffer_updates = _weight_update_utils.split_buffer_updates
refresh_weight_caches = _weight_update_utils.refresh_weight_caches


class Glm5NextLinearAttention(torch.nn.Module):
    def __init__(self, build_cache=True):
        super().__init__()
        for kind in "qkv":
            setattr(self, f"{kind}_conv1d", torch.nn.Conv1d(4, 4, 4, groups=4, bias=False))
        with torch.inference_mode():
            self._merged_conv_weight = torch.zeros(12, 4) if build_cache else None


def test_refresh_conv_cache_preserves_inference_storage():
    layer = Glm5NextLinearAttention()
    cached = layer._merged_conv_weight
    original_pointer = cached.data_ptr()
    for step in (1, 2):
        with torch.no_grad():
            for index, kind in enumerate("qkv"):
                getattr(layer, f"{kind}_conv1d").weight.fill_(step * (index + 1))
        assert refresh_weight_caches(layer) == 1
        assert layer._merged_conv_weight is cached
        assert cached.data_ptr() == original_pointer
        expected = torch.cat([torch.full((4, 4), step * index) for index in (1, 2, 3)]).float()
        torch.testing.assert_close(cached, expected)


def test_refresh_conv_cache_keeps_lazy_initialization():
    layer = Glm5NextLinearAttention(build_cache=False)
    assert refresh_weight_caches(layer) == 0
    assert layer._merged_conv_weight is None
    assert refresh_weight_caches(torch.nn.Linear(4, 4)) == 0


class Glm5NextMLAAttention(torch.nn.Module):
    def __init__(self, build_cache=True):
        super().__init__()
        self.indexer = torch.nn.Module()
        self.indexer.head_dim = 4
        self.indexer.wk_weights_proj = torch.nn.Linear(8, 6, bias=False, dtype=torch.bfloat16)
        with torch.inference_mode():
            self.indexer._wp_fp32 = torch.zeros(8, 2) if build_cache else None


def test_refresh_indexer_cache_preserves_fp32_inference_storage():
    layer = Glm5NextMLAAttention()
    cached = layer.indexer._wp_fp32
    original_pointer = cached.data_ptr()
    for step in (1, 2):
        weight = torch.arange(48, dtype=torch.bfloat16).reshape(6, 8) * step
        with torch.no_grad():
            layer.indexer.wk_weights_proj.weight.copy_(weight)
        assert refresh_weight_caches(layer) == 1
        assert layer.indexer._wp_fp32 is cached
        assert cached.data_ptr() == original_pointer
        assert cached.dtype == torch.float32
        torch.testing.assert_close(cached, torch.stack([weight[4], weight[5]], dim=1).float())


def test_refresh_indexer_cache_keeps_lazy_initialization():
    layer = Glm5NextMLAAttention(build_cache=False)
    assert refresh_weight_caches(layer) == 0
    assert layer.indexer._wp_fp32 is None
    layer.indexer = None
    assert refresh_weight_caches(layer) == 0


def _load_vllm_rollout_utils():
    """Load vllm_rollout/utils.py with heavyweight deps stubbed.

    Injected ``sys.modules`` entries are restored afterwards so the fakes do not
    leak into other tests; the loaded module keeps working since it binds the
    names it needs at import time.
    """
    module_name = "verl.workers.rollout.vllm_rollout.utils"
    module_path = _REPO_ROOT / "verl/workers/rollout/vllm_rollout/utils.py"

    fake_outputs = types.ModuleType("vllm.outputs")

    class _FakeRequestOutput:
        pass

    fake_outputs.RequestOutput = _FakeRequestOutput
    fake_vllm = types.ModuleType("vllm")
    fake_vllm.outputs = fake_outputs

    fake_vllm_third_party = types.ModuleType("verl.third_party.vllm")
    fake_vllm_third_party.VLLM_SLEEP_LEVEL = 1
    fake_vllm_third_party.get_version = lambda pkg: "0.8.0"

    fake_vllm_utils = types.ModuleType("verl.utils.vllm")

    class _FakeTensorLoRARequest:
        pass

    class _FakeVLLMHijack:
        @staticmethod
        def hijack():
            return None

    fake_vllm_utils.TensorLoRARequest = _FakeTensorLoRARequest
    fake_vllm_utils.VLLMHijack = _FakeVLLMHijack
    fake_vllm_utils.resolve_weight_name = lambda model, name, names: name

    fake_vllm_patch = types.ModuleType("verl.utils.vllm.patch")
    fake_vllm_patch.patch_vllm_moe_model_weight_loader = lambda model: None

    fake_vllm_quant = types.ModuleType("verl.utils.vllm.vllm_quant_utils")
    fake_vllm_quant.apply_vllm_quant_patches = lambda: None
    fake_vllm_quant.is_fp8_model = lambda config: False
    fake_vllm_quant.load_quanted_weights = lambda weights, runner, is_drafter=False: weights

    # NOTE: deliberately do NOT stub verl.plugin.platform. It is lightweight and
    # imports fine on CPU. verl.utils.device binds `get_platform` at import time, so
    # a `lambda: None` stub gets baked into verl.utils.device during this window;
    # because only the keys in `fakes` below are restored, that fake would leak
    # process-wide and later tests would crash in get_device_name() with
    # "'NoneType' object has no attribute 'device_name'".

    fakes = {
        "vllm": fake_vllm,
        "vllm.outputs": fake_outputs,
        "verl.third_party.vllm": fake_vllm_third_party,
        "verl.utils.vllm": fake_vllm_utils,
        "verl.utils.vllm.patch": fake_vllm_patch,
        "verl.utils.vllm.vllm_quant_utils": fake_vllm_quant,
        "verl.workers.rollout.vllm_rollout.weight_update_utils": _weight_update_utils,
    }

    saved = {name: sys.modules.get(name) for name in fakes}
    try:
        sys.modules.update(fakes)
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        spec.loader.exec_module(module)
    finally:
        for name, prev in saved.items():
            if prev is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prev
    return module


_vllm_rollout_utils = _load_vllm_rollout_utils()
vLLMColocateWorkerExtension = _vllm_rollout_utils.vLLMColocateWorkerExtension


def test_vllm_refreshes_derived_caches_after_all_weight_buckets(monkeypatch):
    model = torch.nn.Sequential(Glm5NextLinearAttention(), Glm5NextMLAAttention())
    worker = object.__new__(vLLMColocateWorkerExtension)
    worker.model_runner = _FakeModelRunner(model)
    worker.model_runner.vllm_config.model_config = object()
    worker.device = torch.device("cpu")
    worker._is_qat_model = False
    worker._is_modelopt_qat = False
    worker._get_zmq_handle = lambda: None
    cached_conv = model[0]._merged_conv_weight
    cached_gate = model[1].indexer._wp_fp32

    def load_weights(weights):
        with torch.no_grad():
            for name, source in weights:
                model.get_parameter(name).copy_(source)

    model.load_weights = load_weights

    class Receiver:
        def __init__(self, **kwargs):
            pass

        def receive_weights(self, on_bucket_received):
            params = list(model.named_parameters())
            for index, (name, parameter) in enumerate(params):
                on_bucket_received([(name, torch.full_like(parameter, index + 1))], index == len(params) - 1)
                assert torch.count_nonzero(cached_conv) == 0
                assert torch.count_nonzero(cached_gate) == 0

    post_load_calls = []

    def post_load(inner_model, config, device):
        post_load_calls.append(inner_model)
        assert torch.count_nonzero(cached_conv) == 0
        assert torch.count_nonzero(cached_gate) == 0

    transfer = types.ModuleType("verl.workers.rollout.vllm_rollout.bucketed_weight_transfer")
    transfer.BucketedWeightReceiver = Receiver
    loader = types.ModuleType("vllm.model_executor.model_loader.utils")
    loader.process_weights_after_loading = post_load
    monkeypatch.setitem(sys.modules, transfer.__name__, transfer)
    monkeypatch.setitem(sys.modules, loader.__name__, loader)

    worker.update_weights_from_ipc(peft_config=None, base_sync_done=True)

    assert post_load_calls == [model]
    assert model[0]._merged_conv_weight is cached_conv
    assert model[1].indexer._wp_fp32 is cached_gate
    expected_conv = torch.repeat_interleave(torch.arange(1, 4).float(), 4)[:, None].expand(12, 4)
    torch.testing.assert_close(cached_conv, expected_conv)
    torch.testing.assert_close(cached_gate, torch.full((8, 2), 4.0))


class _ToyBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4, bias=False)
        self.register_buffer("e_score_correction_bias", torch.zeros(4, dtype=torch.float32))


class _ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([_ToyBlock()])


class _FakeVllmConfig:
    def __init__(self, speculative_config=None):
        self.speculative_config = speculative_config


class _FakeModelRunner:
    def __init__(self, inner_model, speculative_config=None):
        self.model = inner_model
        self.vllm_config = _FakeVllmConfig(speculative_config=speculative_config)


def test_split_buffer_updates_routes_registered_buffers():
    model = _ToyModel()
    weights = [
        ("model.layers.0.linear.weight", torch.ones(4, 4, dtype=torch.float32)),
        ("model.layers.0.e_score_correction_bias", torch.arange(4, dtype=torch.float32)),
    ]

    param_updates, buffer_updates, named_buffers = split_buffer_updates(model, weights)

    assert [name for name, _ in param_updates] == ["model.layers.0.linear.weight"]
    assert [name for name, _ in buffer_updates] == ["model.layers.0.e_score_correction_bias"]
    assert "model.layers.0.e_score_correction_bias" in named_buffers


def test_apply_buffer_updates_copies_buffer_values():
    model = _ToyModel()
    updates = [("model.layers.0.e_score_correction_bias", torch.arange(4, dtype=torch.float32) + 1)]

    loaded = apply_buffer_updates(model, updates)

    assert loaded == 1
    torch.testing.assert_close(
        model.model.layers[0].e_score_correction_bias, torch.tensor([1, 2, 3, 4], dtype=torch.float32)
    )


def test_apply_buffer_updates_ignores_non_buffer_weights():
    model = _ToyModel()
    weights = [("model.layers.0.linear.weight", torch.ones(4, 4, dtype=torch.float32))]

    loaded = apply_buffer_updates(model, weights)

    assert loaded == 0
    assert torch.count_nonzero(model.model.layers[0].e_score_correction_bias) == 0


def test_vllm_update_weights_loads_params_and_buffers():
    model = _ToyModel()
    loaded_param_names = []
    apply_named_buffers = []

    def _fake_load_weights(weights):
        loaded_param_names.extend(name for name, _ in weights)

    model.load_weights = _fake_load_weights

    original_apply_buffer_updates = _vllm_rollout_utils.apply_buffer_updates

    def _spy_apply_buffer_updates(inner_model, buffer_updates, named_buffers=None):
        apply_named_buffers.append(named_buffers)
        return original_apply_buffer_updates(inner_model, buffer_updates, named_buffers=named_buffers)

    _vllm_rollout_utils.apply_buffer_updates = _spy_apply_buffer_updates

    worker = object.__new__(vLLMColocateWorkerExtension)
    worker.model_runner = _FakeModelRunner(model)

    weights = [
        ("model.layers.0.linear.weight", torch.ones(4, 4, dtype=torch.float32)),
        ("model.layers.0.e_score_correction_bias", torch.arange(4, dtype=torch.float32) + 5),
    ]

    try:
        worker._update_weights(weights, peft_config=None, base_sync_done=False)
    finally:
        _vllm_rollout_utils.apply_buffer_updates = original_apply_buffer_updates

    assert loaded_param_names == ["model.layers.0.linear.weight"]
    assert apply_named_buffers and apply_named_buffers[0] is not None
    torch.testing.assert_close(
        model.model.layers[0].e_score_correction_bias, torch.tensor([5, 6, 7, 8], dtype=torch.float32)
    )


def test_vllm_update_weights_syncs_buffers_to_mtp_drafter():
    """When an MTP drafter is synced, its registered buffers must be updated too."""
    main_model = _ToyModel()
    drafter_model = _ToyModel()
    main_model.load_weights = lambda weights: None
    drafter_model.load_weights = lambda weights: None

    class _SpecConfig:
        method = "mtp"
        draft_model_config = object()

    class _Drafter:
        def __init__(self, m):
            self.model = m

    worker = object.__new__(vLLMColocateWorkerExtension)
    worker.model_runner = _FakeModelRunner(main_model, speculative_config=_SpecConfig())
    worker.model_runner.drafter = _Drafter(drafter_model)

    weights = [
        ("model.layers.0.linear.weight", torch.ones(4, 4, dtype=torch.float32)),
        ("model.layers.0.e_score_correction_bias", torch.arange(4, dtype=torch.float32) + 5),
    ]

    worker._update_weights(weights, peft_config=None, base_sync_done=False)

    expected = torch.tensor([5, 6, 7, 8], dtype=torch.float32)
    torch.testing.assert_close(main_model.model.layers[0].e_score_correction_bias, expected)
    torch.testing.assert_close(drafter_model.model.layers[0].e_score_correction_bias, expected)
