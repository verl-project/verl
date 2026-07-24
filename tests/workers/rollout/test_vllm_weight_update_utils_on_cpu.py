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


def _stub_fp8_layerwise_env(monkeypatch, *, receive_error=None, finalize_error=None):
    """Wire update_weights_from_ipc onto the FP8 layerwise path with all vLLM /
    fp8_utils touch-points faked. Returns the list that records abort() calls.

    ``receive_error`` / ``finalize_error`` inject a failure at the matching
    lifecycle stage; both None drives the happy path.
    """
    abort_calls: list[str] = []

    fake_config = types.ModuleType("vllm.config")

    class _CtxMgr:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return None

        def __exit__(self, *a):
            return False

    fake_config.set_current_vllm_config = _CtxMgr

    fake_platforms = types.ModuleType("vllm.platforms")

    class _Platform:
        device_type = "cpu"

    fake_platforms.current_platform = _Platform()

    fake_bwt = types.ModuleType("verl.workers.rollout.vllm_rollout.bucketed_weight_transfer")

    class _Receiver:
        def __init__(self, **kwargs):
            pass

        def receive_weights(self, on_bucket_received):
            if receive_error is not None:
                raise receive_error

    fake_bwt.BucketedWeightReceiver = _Receiver

    fake_fp8 = types.ModuleType("verl.utils.vllm.vllm_fp8_utils")
    fake_fp8.prepare_quanted_weights_for_loading = lambda runner: False
    fake_fp8._vllm_supports_layerwise_reload = lambda: True
    fake_fp8.is_mxfp8_vllm_ascend = lambda cfg: False
    fake_fp8.begin_fp8_layerwise_reload = lambda model, tag="main": True

    def _finalize(model, model_config, tag="main"):
        if finalize_error is not None:
            raise finalize_error

    fake_fp8.finalize_fp8_layerwise_reload = _finalize
    fake_fp8.abort_fp8_layerwise_reload = lambda tag="main": abort_calls.append(tag)

    monkeypatch.setitem(sys.modules, "vllm.config", fake_config)
    monkeypatch.setitem(sys.modules, "vllm.platforms", fake_platforms)
    monkeypatch.setitem(sys.modules, "verl.workers.rollout.vllm_rollout.bucketed_weight_transfer", fake_bwt)
    monkeypatch.setitem(sys.modules, "verl.utils.vllm.vllm_fp8_utils", fake_fp8)
    # is_fp8_model was bound into the utils module at import time.
    monkeypatch.setattr(_vllm_rollout_utils, "is_fp8_model", lambda cfg: True)
    return abort_calls


def _make_fp8_worker():
    worker = object.__new__(vLLMColocateWorkerExtension)
    worker.model_runner = _FakeModelRunner(_ToyModel())
    worker.model_runner.vllm_config.quant_config = object()
    worker.model_runner.vllm_config.model_config = object()
    worker.device = torch.device("cpu")
    worker.local_rank = 0
    worker._is_qat_model = False
    worker._is_modelopt_qat = False
    return worker


@pytest.mark.parametrize("stage", ["receive", "finalize"])
def test_update_weights_from_ipc_fp8_failure_fails_stop(monkeypatch, stage):
    """A failure in the FP8 layerwise reload (at receive OR finalize) must clear
    the reload flag, mark the worker fail-stop, and make a follow-on sync refuse
    instead of re-initializing a partially-updated model."""
    err = RuntimeError(f"boom in {stage}")
    abort_calls = _stub_fp8_layerwise_env(
        monkeypatch,
        receive_error=err if stage == "receive" else None,
        finalize_error=err if stage == "finalize" else None,
    )
    worker = _make_fp8_worker()

    with pytest.raises(RuntimeError, match=f"boom in {stage}"):
        worker.update_weights_from_ipc()

    assert worker._fp8_reload_failed is True
    assert abort_calls == ["main"]

    # The worker is now poisoned: any subsequent sync must refuse up front.
    with pytest.raises(RuntimeError, match="poisoned by a prior failed FP8"):
        worker.update_weights_from_ipc()


def test_update_weights_from_ipc_fp8_success_does_not_fail_stop(monkeypatch):
    """Negative control: a clean FP8 layerwise reload leaves no fail-stop marker
    and does not call abort."""
    abort_calls = _stub_fp8_layerwise_env(monkeypatch)
    worker = _make_fp8_worker()

    worker.update_weights_from_ipc()

    assert getattr(worker, "_fp8_reload_failed", False) is False
    assert abort_calls == []
