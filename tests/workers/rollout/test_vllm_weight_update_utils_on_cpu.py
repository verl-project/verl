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


@pytest.mark.parametrize("sync_mtp", [False, True], ids=["main", "main-and-mtp"])
def test_vllm_update_weights_loads_params_and_buffers(sync_mtp):
    models = [_ToyModel() for _ in range(2 if sync_mtp else 1)]
    addresses = []
    for model in models:
        addresses.append(
            (model.model.layers[0].linear.weight.data_ptr(), model.model.layers[0].e_score_correction_bias.data_ptr())
        )

        def load_weights(weights, model=model):
            with torch.no_grad():
                for name, value in weights:
                    dict(model.named_parameters())[name].copy_(value)

        model.load_weights = load_weights

    spec = types.SimpleNamespace(method="mtp", draft_model_config=object()) if sync_mtp else None
    worker = object.__new__(vLLMColocateWorkerExtension)
    worker.model_runner = _FakeModelRunner(models[0], speculative_config=spec)
    if sync_mtp:
        worker.model_runner.drafter = types.SimpleNamespace(model=models[1])
    bucket = torch.empty(16)
    inputs = torch.ones(1, 4)
    for version in (1, 2):
        bucket.fill_(version)
        worker._update_weights(
            [("model.layers.0.linear.weight", bucket.view(4, 4))], peft_config=None, base_sync_done=False
        )
        expected_buffer = torch.arange(4, dtype=torch.float32) + version * 5
        bucket[:4].copy_(expected_buffer)
        worker._update_weights(
            [("model.layers.0.e_score_correction_bias", bucket[:4])], peft_config=None, base_sync_done=False
        )
        bucket.fill_(-999)  # Standard loaders consume the receiver's view synchronously.
        for model, address in zip(models, addresses, strict=True):
            block = model.model.layers[0]
            assert (block.linear.weight.data_ptr(), block.e_score_correction_bias.data_ptr()) == address
            torch.testing.assert_close(block.linear.weight, torch.full((4, 4), float(version)))
            torch.testing.assert_close(block.e_score_correction_bias, expected_buffer)
            torch.testing.assert_close(
                block.linear(inputs) + block.e_score_correction_bias, 4 * version + expected_buffer[None]
            )
