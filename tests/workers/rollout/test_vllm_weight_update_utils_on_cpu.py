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


def _vocab_parallel_embedding_stubs():
    """Stubs standing in for vLLM's ``VocabParallelEmbedding``, or {} if vLLM is real.

    ``weight_update_utils`` imports that class at module scope for tied-alias
    detection, so loading it on CPU needs a placeholder. This file only exercises
    the buffer helpers, which never touch it.
    """
    try:
        importlib.import_module("vllm.model_executor.layers.vocab_parallel_embedding")
    except ImportError:
        pass
    else:
        return {}

    names = [
        "vllm",
        "vllm.model_executor",
        "vllm.model_executor.layers",
        "vllm.model_executor.layers.vocab_parallel_embedding",
    ]
    stubs = {name: sys.modules.get(name) or types.ModuleType(name) for name in names}
    stubs[names[-1]].VocabParallelEmbedding = type("VocabParallelEmbedding", (), {})
    return stubs


def _load_weight_update_utils():
    module_path = _REPO_ROOT / "verl/workers/rollout/vllm_rollout/weight_update_utils.py"
    spec = importlib.util.spec_from_file_location("weight_update_utils", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None

    stubs = _vocab_parallel_embedding_stubs()
    saved = {name: sys.modules.get(name) for name in stubs}
    try:
        sys.modules.update(stubs)
        spec.loader.exec_module(module)
    finally:
        for name, prev in saved.items():
            if prev is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prev
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
    fake_vllm_utils.resolve_weight_name = lambda model, name, names: name

    fake_vllm_patch = types.ModuleType("verl.utils.vllm.patch")
    fake_vllm_patch.patch_vllm_moe_model_weight_loader = lambda model: None

    fake_vllm_quant = types.ModuleType("verl.utils.vllm.vllm_quant_utils")
    fake_vllm_quant.apply_vllm_quant_patches = lambda: None
    fake_vllm_quant.configure_fp8_ignored_layers_match_mode = lambda config: None
    fake_vllm_quant.is_quantized_model = lambda config: False
    fake_vllm_quant.refresh_rocm_attention_weight_caches = lambda model: None
    fake_vllm_quant.reset_uncalibrated_kv_cache_scales = lambda model: None

    fake_rocm_expert_map = types.ModuleType("verl.utils.vllm.rocm_vllm_moe_expert_map")
    fake_rocm_expert_map.restore_moe_expert_maps = lambda model: None

    fake_bucketed_transfer = types.ModuleType("verl.workers.rollout.vllm_rollout.bucketed_weight_transfer")
    fake_bucketed_transfer.BucketedWeightReceiver = None

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
        "verl.utils.vllm.rocm_vllm_moe_expert_map": fake_rocm_expert_map,
        "verl.workers.rollout.vllm_rollout.bucketed_weight_transfer": fake_bucketed_transfer,
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


@pytest.mark.parametrize("owner_attr", ["drafter", "speculator"])
def test_vllm_update_weights_syncs_buffers_to_mtp_drafter(owner_attr):
    """MTP buffers must stay in sync with either the V1 or V2 draft-model owner."""
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
    setattr(worker.model_runner, owner_attr, _Drafter(drafter_model))

    weights = [
        ("model.layers.0.linear.weight", torch.ones(4, 4, dtype=torch.float32)),
        ("model.layers.0.e_score_correction_bias", torch.arange(4, dtype=torch.float32) + 5),
    ]

    worker._update_weights(weights, peft_config=None, base_sync_done=False)

    expected = torch.tensor([5, 6, 7, 8], dtype=torch.float32)
    torch.testing.assert_close(main_model.model.layers[0].e_score_correction_bias, expected)
    torch.testing.assert_close(drafter_model.model.layers[0].e_score_correction_bias, expected)


@pytest.mark.parametrize("fail_receive", [False, True])
def test_bucketed_transfer_is_bracketed_by_vllm_reload(monkeypatch, fail_receive):
    events = []
    _install_fake_reload(monkeypatch, events)

    class Receiver:
        def __init__(self, **kwargs):
            pass

        def receive_weights(self, on_bucket_received):
            on_bucket_received([], False)
            if fail_receive:
                raise RuntimeError("transfer failed")
            on_bucket_received([], True)

    monkeypatch.setattr(_vllm_rollout_utils, "BucketedWeightReceiver", Receiver)
    worker = object.__new__(vLLMColocateWorkerExtension)
    worker.device = torch.device("cpu")
    worker.model_runner = _FakeModelRunner(_ToyModel())
    monkeypatch.setattr(worker, "_iter_all_models", lambda: [None])
    monkeypatch.setattr(worker, "_iter_all_models_with_config", lambda: [(None, None)])
    monkeypatch.setattr(worker, "_get_zmq_handle", lambda: "unused")
    monkeypatch.setattr(worker, "_update_weights", lambda *args, **kwargs: events.append("bucket"))

    if fail_receive:
        with pytest.raises(RuntimeError, match="transfer failed"):
            worker.update_weights_from_ipc()
        assert events == ["start_reload", "bucket"]
    else:
        worker.update_weights_from_ipc()
        assert events == ["start_reload", "bucket", "bucket", "finish_reload"]


def _install_fake_reload(monkeypatch, events):
    """Record vLLM's reload lifecycle calls (start_reload / finish_reload)."""
    reload = types.ModuleType("vllm.model_executor.model_loader.reload")
    reload.start_reload = lambda model: events.append("start_reload")
    reload.finish_reload = lambda model, model_config, device: events.append("finish_reload")
    monkeypatch.setitem(sys.modules, reload.__name__, reload)


def test_reload_skips_drafter_start_reload_when_draft_model_config_is_none(monkeypatch):
    started_models = []
    finished_models = []
    reload = types.ModuleType("vllm.model_executor.model_loader.reload")
    reload.start_reload = lambda model: started_models.append(model)
    reload.finish_reload = lambda model, model_config, device: finished_models.append(model)
    monkeypatch.setitem(sys.modules, reload.__name__, reload)

    class _Receiver:
        def __init__(self, **kwargs):
            pass

        def receive_weights(self, on_bucket_received):
            on_bucket_received([], True)

    monkeypatch.setattr(_vllm_rollout_utils, "BucketedWeightReceiver", _Receiver)
    monkeypatch.setattr(_vllm_rollout_utils, "is_npu_available", False)

    main_model = _ToyModel()
    drafter_model = _ToyModel()

    class _SpecConfig:
        method = "mtp"
        draft_model_config = None

    worker = object.__new__(vLLMColocateWorkerExtension)
    worker.device = torch.device("cpu")
    worker.model_runner = _FakeModelRunner(main_model, speculative_config=_SpecConfig())
    worker.model_runner.vllm_config.model_config = object()
    worker.model_runner.drafter = types.SimpleNamespace(model=drafter_model)
    monkeypatch.setattr(worker, "_get_zmq_handle", lambda: "unused")
    monkeypatch.setattr(worker, "_update_weights", lambda *args, **kwargs: None)

    worker.update_weights_from_ipc()
    assert started_models == [main_model]
    assert finished_models == [main_model]


def test_npu_quantized_and_mtp_update_resets_kv_scales_and_disables_mtp_check(monkeypatch):
    events = []

    class _MtpCtx:
        def __enter__(self):
            events.append("enter_mtp_disable")

        def __exit__(self, *args):
            events.append("exit_mtp_disable")

    mtp_mod = types.ModuleType("vllm.model_executor.model_loader.mtp_validation")
    mtp_mod.disable_mtp_completeness_check = _MtpCtx
    monkeypatch.setitem(sys.modules, mtp_mod.__name__, mtp_mod)

    loader_utils = types.ModuleType("vllm.model_executor.model_loader.utils")
    loader_utils.process_weights_after_loading = lambda model, cfg, dev: events.append(("post_load", model))
    monkeypatch.setitem(sys.modules, loader_utils.__name__, loader_utils)

    class _Receiver:
        def __init__(self, **kwargs):
            pass

        def receive_weights(self, on_bucket_received):
            events.append("receive")
            on_bucket_received([], True)

    monkeypatch.setattr(_vllm_rollout_utils, "BucketedWeightReceiver", _Receiver)
    monkeypatch.setattr(_vllm_rollout_utils, "is_npu_available", True)
    monkeypatch.setattr(_vllm_rollout_utils, "is_quantized_model", lambda cfg: True)
    monkeypatch.setattr(
        _vllm_rollout_utils, "reset_uncalibrated_kv_cache_scales", lambda m: events.append(("reset_kv", m))
    )

    main_model = _ToyModel()
    drafter_model = _ToyModel()

    class _SpecConfig:
        method = "mtp"
        draft_model_config = object()

    worker = object.__new__(vLLMColocateWorkerExtension)
    worker.device = torch.device("cpu")
    worker.model_runner = _FakeModelRunner(main_model, speculative_config=_SpecConfig())
    worker.model_runner.vllm_config.model_config = object()
    worker.model_runner.drafter = types.SimpleNamespace(model=drafter_model)
    monkeypatch.setattr(worker, "_get_zmq_handle", lambda: "unused")
    monkeypatch.setattr(worker, "_update_weights", lambda *args, **kwargs: None)

    worker.update_weights_from_ipc()
    assert events == [
        "enter_mtp_disable",
        "receive",
        "exit_mtp_disable",
        ("post_load", main_model),
        ("post_load", drafter_model),
        ("reset_kv", main_model),
        ("reset_kv", drafter_model),
    ]
