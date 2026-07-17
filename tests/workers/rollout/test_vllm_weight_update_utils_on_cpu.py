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

import pytest
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

    fake_vllm_patch = types.ModuleType("verl.utils.vllm.patch")
    fake_vllm_patch.patch_vllm_moe_model_weight_loader = lambda model: None

    fake_vllm_fp8 = types.ModuleType("verl.utils.vllm.vllm_fp8_utils")
    fake_vllm_fp8.apply_vllm_fp8_patches = lambda: None
    fake_vllm_fp8.is_fp8_model = lambda config: False
    fake_vllm_fp8.load_quanted_weights = lambda weights, runner, is_drafter=False: weights

    fake_platform = types.ModuleType("verl.plugin.platform")
    fake_platform.get_platform = lambda: None

    fakes = {
        "vllm": fake_vllm,
        "vllm.outputs": fake_outputs,
        "verl.third_party.vllm": fake_vllm_third_party,
        "verl.utils.vllm": fake_vllm_utils,
        "verl.utils.vllm.patch": fake_vllm_patch,
        "verl.utils.vllm.vllm_fp8_utils": fake_vllm_fp8,
        "verl.plugin.platform": fake_platform,
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


def _run_with_fake_modules(fakes, fn):
    saved = {name: sys.modules.get(name) for name in fakes}
    try:
        sys.modules.update(fakes)
        return fn()
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


def _make_weight_reload_fakes(events, device_type="npu", initialize_error=None, reload_api_available=True):
    class _FakeBucketedWeightReceiver:
        def __init__(self, **kwargs):
            events.append("receiver:init")

        def receive_weights(self, on_bucket_received):
            on_bucket_received([("model.layers.0.linear.weight", torch.ones(4, 4))])

    fake_vllm = types.ModuleType("vllm")
    fake_platforms = types.ModuleType("vllm.platforms")
    fake_platforms.current_platform = types.SimpleNamespace(device_type=device_type)
    fake_model_executor = types.ModuleType("vllm.model_executor")
    fake_model_loader = types.ModuleType("vllm.model_executor.model_loader")
    fake_reload = types.ModuleType("vllm.model_executor.model_loader.reload")
    fake_loader_utils = types.ModuleType("vllm.model_executor.model_loader.utils")
    fake_bucket_transfer = types.ModuleType("verl.workers.rollout.vllm_rollout.bucketed_weight_transfer")

    fake_vllm.platforms = fake_platforms
    fake_vllm.model_executor = fake_model_executor
    fake_model_executor.model_loader = fake_model_loader
    fake_model_loader.reload = fake_reload
    fake_model_loader.utils = fake_loader_utils
    fake_bucket_transfer.BucketedWeightReceiver = _FakeBucketedWeightReceiver

    def _initialize(model):
        events.append(f"initialize:{model._test_reload_name}")
        if initialize_error is not None:
            raise initialize_error

    def _finalize(model, model_config):
        events.append(f"finalize:{model._test_reload_name}")

    if reload_api_available:
        fake_reload.initialize_layerwise_reload = _initialize
        fake_reload.finalize_layerwise_reload = _finalize
    fake_loader_utils.process_weights_after_loading = lambda model, model_config, device: events.append(
        f"postprocess:{model._test_reload_name}"
    )

    return {
        "vllm": fake_vllm,
        "vllm.platforms": fake_platforms,
        "vllm.model_executor": fake_model_executor,
        "vllm.model_executor.model_loader": fake_model_loader,
        "vllm.model_executor.model_loader.reload": fake_reload,
        "vllm.model_executor.model_loader.utils": fake_loader_utils,
        "verl.workers.rollout.vllm_rollout.bucketed_weight_transfer": fake_bucket_transfer,
    }


def _build_weight_reload_worker(target, drafter=None, draft_model_config="draft_config", enforce_eager=False):
    class _SpecConfig:
        method = "mtp"

        def __init__(self, model_config):
            self.draft_model_config = model_config

    class _Drafter:
        def __init__(self, model):
            self.model = model

    speculative_config = _SpecConfig(draft_model_config) if drafter is not None else None
    worker = object.__new__(vLLMColocateWorkerExtension)
    worker.model_runner = _FakeModelRunner(target, speculative_config=speculative_config)
    worker.model_runner.vllm_config.model_config = types.SimpleNamespace(enforce_eager=enforce_eager)
    target._test_reload_name = "target"
    if drafter is not None:
        worker.model_runner.drafter = _Drafter(drafter)
        drafter._test_reload_name = "drafter"
    worker.device = torch.device("cpu")
    worker.local_rank = 0
    worker._is_qat_model = False
    worker._is_modelopt_qat = False
    worker._get_zmq_handle = lambda: "ipc://cpu-test"
    worker.prepare_model_for_native_layerwise_reload = lambda model: None
    return worker


@pytest.mark.parametrize("enforce_eager", [False, True], ids=["graph", "eager"])
def test_npu_mtp_reload_lifecycle_is_independent_of_execution_mode(enforce_eager):
    events = []
    target = _ToyModel()
    drafter = _ToyModel()
    target.load_weights = lambda weights: events.append("load:target")
    drafter.load_weights = lambda weights: events.append("load:drafter")
    worker = _build_weight_reload_worker(target, drafter, enforce_eager=enforce_eager)
    worker.prepare_model_for_native_layerwise_reload = lambda model: events.append(f"prepare:{model._test_reload_name}")

    fakes = _make_weight_reload_fakes(events)
    _run_with_fake_modules(fakes, worker.update_weights_from_ipc)

    assert events == [
        "prepare:target",
        "prepare:drafter",
        "initialize:target",
        "initialize:drafter",
        "receiver:init",
        "load:target",
        "load:drafter",
        "finalize:target",
        "finalize:drafter",
    ]


def test_cuda_mtp_reload_keeps_direct_weight_loading():
    events = []
    target = _ToyModel()
    drafter = _ToyModel()
    target.load_weights = lambda weights: events.append("load:target")
    drafter.load_weights = lambda weights: events.append("load:drafter")
    worker = _build_weight_reload_worker(target, drafter)
    worker._native_npu_reload_failure = "a previous NPU-only failure"
    worker._uses_native_npu_mtp_reload = lambda *_args: pytest.fail("entered NPU-only dispatch")

    fakes = _make_weight_reload_fakes(events, device_type="cuda")
    _run_with_fake_modules(fakes, worker.update_weights_from_ipc)

    assert events == [
        "receiver:init",
        "load:target",
        "load:drafter",
        "postprocess:target",
        "postprocess:drafter",
    ]


@pytest.mark.parametrize("enforce_eager", [False, True], ids=["graph", "eager"])
def test_non_mtp_npu_reload_keeps_direct_weight_loading(enforce_eager):
    events = []
    target = _ToyModel()
    target.load_weights = lambda weights: events.append("load:target")
    worker = _build_weight_reload_worker(target, enforce_eager=enforce_eager)

    fakes = _make_weight_reload_fakes(events)
    _run_with_fake_modules(fakes, worker.update_weights_from_ipc)

    assert events == ["receiver:init", "load:target", "postprocess:target"]


def test_npu_mtp_reload_validates_draft_config_before_mutation():
    events = []
    worker = _build_weight_reload_worker(_ToyModel(), _ToyModel(), draft_model_config=None)
    worker.prepare_model_for_native_layerwise_reload = lambda model: events.append("prepare")

    fakes = _make_weight_reload_fakes(events)
    with pytest.raises(RuntimeError, match="draft_model_config is missing"):
        _run_with_fake_modules(fakes, worker.update_weights_from_ipc)

    assert events == []


def test_npu_mtp_reload_requires_backend_prepare_hook_before_mutation():
    events = []
    worker = _build_weight_reload_worker(_ToyModel(), _ToyModel())
    worker.prepare_model_for_native_layerwise_reload = None

    fakes = _make_weight_reload_fakes(events)
    with pytest.raises(RuntimeError, match="prepare_model_for_native_layerwise_reload"):
        _run_with_fake_modules(fakes, worker.update_weights_from_ipc)

    assert events == []


def test_npu_mtp_reload_requires_public_vllm_api_before_mutation():
    events = []
    worker = _build_weight_reload_worker(_ToyModel(), _ToyModel())

    fakes = _make_weight_reload_fakes(events, reload_api_available=False)
    with pytest.raises(RuntimeError, match="public layerwise reload API"):
        _run_with_fake_modules(fakes, worker.update_weights_from_ipc)

    assert events == []


def test_npu_mtp_reload_failure_makes_worker_fail_stop():
    events = []
    failure = RuntimeError("initialize failed")
    worker = _build_weight_reload_worker(_ToyModel(), _ToyModel())

    fakes = _make_weight_reload_fakes(events, initialize_error=failure)
    with pytest.raises(RuntimeError, match="initialize failed") as exc_info:
        _run_with_fake_modules(fakes, worker.update_weights_from_ipc)

    assert exc_info.value is failure
    assert events == ["initialize:target"]

    with pytest.raises(RuntimeError, match="Recreate the worker"):
        _run_with_fake_modules(fakes, worker.update_weights_from_ipc)

    assert events == ["initialize:target"]


@pytest.mark.parametrize(
    "has_drafter,use_standard_weight_load,is_qat_model,is_modelopt_qat,expected",
    [
        pytest.param(True, True, False, False, True, id="standard-mtp"),
        pytest.param(False, True, False, False, False, id="non-mtp"),
        pytest.param(True, False, False, False, False, id="non-standard-mtp"),
        pytest.param(True, True, True, False, False, id="qat-mtp"),
        pytest.param(True, True, False, True, False, id="modelopt-mtp"),
    ],
)
def test_native_npu_mtp_reload_selector_is_narrow(
    has_drafter, use_standard_weight_load, is_qat_model, is_modelopt_qat, expected
):
    drafter = _ToyModel() if has_drafter else None
    worker = _build_weight_reload_worker(_ToyModel(), drafter)
    worker._is_qat_model = is_qat_model
    worker._is_modelopt_qat = is_modelopt_qat

    assert worker._uses_native_npu_mtp_reload(use_standard_weight_load) is expected


def test_native_layerwise_loader_owns_reused_receiver_buffer():
    model = _ToyModel()
    retained_weights = []
    model.load_weights = lambda weights: retained_weights.extend(weights)
    reused_buffer = torch.tensor([1.0])

    vLLMColocateWorkerExtension._load_weights_with_native_layerwise([model], [("first.weight", reused_buffer.view(1))])
    reused_buffer.fill_(2.0)
    vLLMColocateWorkerExtension._load_weights_with_native_layerwise([model], [("second.weight", reused_buffer.view(1))])
    reused_buffer.fill_(3.0)

    assert [name for name, _ in retained_weights] == ["first.weight", "second.weight"]
    torch.testing.assert_close(retained_weights[0][1], torch.tensor([1.0]))
    torch.testing.assert_close(retained_weights[1][1], torch.tensor([2.0]))
    assert retained_weights[0][1].untyped_storage().data_ptr() != reused_buffer.untyped_storage().data_ptr()
