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

import contextlib
import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import patch

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

    with patch.dict(sys.modules, _vocab_parallel_embedding_stubs()):
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
    fake_vllm_utils.resolve_weight_name = lambda model, name, names: name

    fake_vllm_patch = types.ModuleType("verl.utils.vllm.patch")
    fake_vllm_patch.patch_vllm_moe_model_weight_loader = lambda model: None

    fake_vllm_quant = types.ModuleType("verl.utils.vllm.vllm_quant_utils")
    fake_vllm_quant.apply_vllm_quant_patches = lambda: None
    fake_vllm_quant.configure_fp8_ignored_layers_match_mode = lambda config: None
    fake_vllm_quant.is_quantized_model = lambda config: False
    fake_vllm_quant.load_quanted_weights = lambda weights, runner, is_drafter=False: weights
    fake_vllm_quant.prepare_quanted_weights_for_loading = lambda model: None
    fake_vllm_quant.process_quanted_weights_after_loading = lambda model, state: None

    fake_vllm_unquant = types.ModuleType("verl.utils.vllm.vllm_unquant_utils")
    fake_vllm_unquant.stage_unquantized_moe_params = lambda model: []
    fake_vllm_unquant.fold_unquantized_moe_params = lambda layers: contextlib.nullcontext()

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
        "verl.utils.vllm.vllm_unquant_utils": fake_vllm_unquant,
        "verl.utils.vllm.rocm_vllm_moe_expert_map": fake_rocm_expert_map,
        "verl.workers.rollout.vllm_rollout.bucketed_weight_transfer": fake_bucketed_transfer,
        "verl.workers.rollout.vllm_rollout.weight_update_utils": _weight_update_utils,
    }

    with patch.dict(sys.modules, fakes):
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        spec.loader.exec_module(module)
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


class _FakeModelRunner:
    def __init__(self, inner_model, speculative_config=None):
        self.model = inner_model
        self.vllm_config = types.SimpleNamespace(speculative_config=speculative_config)


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


@pytest.mark.parametrize("layerwise", [False, True])
def test_vllm_update_weights_loads_params_and_buffers(layerwise):
    model = _ToyModel()

    def load_weights(weights):
        with torch.no_grad():
            for name, value in weights:
                dict(model.named_parameters())[name].copy_(value)

    model.load_weights = load_weights

    worker = object.__new__(vLLMColocateWorkerExtension)
    worker.model_runner = _FakeModelRunner(model)

    weights = [
        ("model.layers.0.linear.weight", torch.ones(4, 4, dtype=torch.float32)),
        ("model.layers.0.e_score_correction_bias", torch.arange(4, dtype=torch.float32) + 5),
    ]

    worker._update_weights(weights, peft_config=None, base_sync_done=False, layerwise_reload=layerwise)
    torch.testing.assert_close(model.model.layers[0].linear.weight, torch.ones(4, 4))
    torch.testing.assert_close(
        model.model.layers[0].e_score_correction_bias, torch.tensor([5, 6, 7, 8], dtype=torch.float32)
    )


@pytest.mark.parametrize("layerwise", [False, True])
def test_vllm_update_weights_syncs_buffers_to_mtp_drafter(layerwise):
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

    worker._update_weights(weights, peft_config=None, base_sync_done=False, layerwise_reload=layerwise)

    expected = torch.tensor([5, 6, 7, 8], dtype=torch.float32)
    torch.testing.assert_close(main_model.model.layers[0].e_score_correction_bias, expected)
    torch.testing.assert_close(drafter_model.model.layers[0].e_score_correction_bias, expected)


def test_layerwise_reload_owns_weights_across_reused_buckets():
    pytest.importorskip("vllm.model_executor.model_loader.reload")
    from vllm.model_executor.model_loader.reload import (
        finalize_layerwise_reload,
        initialize_layerwise_reload,
        record_metadata_for_reloading,
    )

    model = torch.nn.Linear(3, 2)

    def load_weights(weights):
        for name, value in weights:
            parameter = getattr(model, name)
            parameter.weight_loader(parameter, value)

    model.load_weights = load_weights
    worker = object.__new__(vLLMColocateWorkerExtension)
    worker.model_runner = _FakeModelRunner(model)
    record_metadata_for_reloading(model)
    addresses = [parameter.data_ptr() for parameter in model.parameters()]
    bucket = torch.empty(6)

    for step in range(1, 6):
        initialize_layerwise_reload(model)
        bucket.fill_(step)
        worker._update_weights(
            [("weight", bucket.view(2, 3))], peft_config=None, base_sync_done=True, layerwise_reload=True
        )
        bucket.fill_(-step)
        worker._update_weights([("bias", bucket[:2])], peft_config=None, base_sync_done=True, layerwise_reload=True)
        bucket.zero_()
        finalize_layerwise_reload(model, None)

        torch.testing.assert_close(model.weight, torch.full((2, 3), float(step)))
        torch.testing.assert_close(model.bias, torch.full((2,), float(-step)))
        assert [parameter.data_ptr() for parameter in model.parameters()] == addresses


def test_layerwise_buffer_loader_retains_its_own_bucket_input():
    model = _ToyModel()
    source = torch.arange(4, dtype=torch.float32)
    retained = []
    target = model.model.layers[0].e_score_correction_bias
    target.weight_loader = lambda buffer, loaded: retained.append((buffer, loaded))
    worker = object.__new__(vLLMColocateWorkerExtension)
    worker.model_runner = _FakeModelRunner(model)
    worker._update_weights(
        [("model.layers.0.e_score_correction_bias", source)],
        peft_config=None,
        base_sync_done=True,
        layerwise_reload=True,
    )
    source.zero_()
    assert len(retained) == 1 and retained[0][0] is target
    torch.testing.assert_close(retained[0][1], torch.arange(4, dtype=torch.float32))


def test_layerwise_update_still_drops_tied_aliases(monkeypatch):
    model = _ToyModel()
    loaded = []
    model.load_weights = lambda weights: loaded.extend(weights)
    monkeypatch.setattr(_vllm_rollout_utils, "drop_tied_alias_updates", lambda model, updates: updates[:1])
    worker = object.__new__(vLLMColocateWorkerExtension)
    worker.model_runner = _FakeModelRunner(model)
    source = torch.ones(4, 4)
    worker._update_weights(
        [("model.layers.0.linear.weight", source), ("lm_head.weight", source)],
        peft_config=None,
        base_sync_done=False,
        layerwise_reload=True,
    )
    source.zero_()
    assert [name for name, _ in loaded] == ["model.layers.0.linear.weight"]
    torch.testing.assert_close(loaded[0][1], torch.ones(4, 4))


def test_native_layerwise_reload_updates_meta_buffers_and_mtp_drafter():
    pytest.importorskip("vllm.model_executor.model_loader.reload")
    from vllm.model_executor.model_loader.reload import (
        finalize_layerwise_reload,
        initialize_layerwise_reload,
        record_metadata_for_reloading,
    )

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(2))
            self.register_buffer("checkpoint_buffer", torch.zeros(2))
            self.register_buffer("e_score_correction_bias", torch.zeros(2))
            self.checkpoint_buffer.weight_loader = lambda param, source: param.copy_(source * 2)

        def load_weights(self, weights):
            for name, value in weights:
                param = dict(self.named_parameters())[name]
                param.weight_loader(param, value)

    main, draft = Model(), Model()
    spec = types.SimpleNamespace(method="mtp", draft_model_config=object())
    worker = object.__new__(vLLMColocateWorkerExtension)
    worker.model_runner = _FakeModelRunner(main, speculative_config=spec)
    worker.model_runner.drafter = types.SimpleNamespace(model=draft)
    models = [main, draft]
    addresses = []
    for model in models:
        record_metadata_for_reloading(model)
        addresses.append(
            [model.weight.data_ptr(), model.checkpoint_buffer.data_ptr(), model.e_score_correction_bias.data_ptr()]
        )
    for step in range(1, 4):
        for model in models:
            initialize_layerwise_reload(model)
            assert model.checkpoint_buffer.is_meta
            assert not model.e_score_correction_bias.is_meta
        bucket = torch.full((2,), float(step))
        worker._update_weights([("weight", bucket)], peft_config=None, base_sync_done=True, layerwise_reload=True)
        bucket.fill_(step + 2)
        worker._update_weights(
            [("checkpoint_buffer", bucket), ("e_score_correction_bias", bucket)],
            peft_config=None,
            base_sync_done=True,
            layerwise_reload=True,
        )
        bucket.zero_()
        for index, model in enumerate(models):
            finalize_layerwise_reload(model, None)
            torch.testing.assert_close(model.weight, torch.full((2,), float(step)))
            torch.testing.assert_close(model.checkpoint_buffer, torch.full((2,), float((step + 2) * 2)))
            torch.testing.assert_close(model.e_score_correction_bias, torch.full((2,), float(step + 2)))
            assert [
                model.weight.data_ptr(),
                model.checkpoint_buffer.data_ptr(),
                model.e_score_correction_bias.data_ptr(),
            ] == addresses[index]
