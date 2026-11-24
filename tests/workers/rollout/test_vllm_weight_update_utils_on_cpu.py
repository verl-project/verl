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


def test_engram_key_matching_and_frozen_key_collection():
    from verl.utils.engram import collect_frozen_engram_keys, is_engram_safetensors_key, is_engram_table_param

    # Qwen3.8-Flash-Next PLE n-gram shards & buffers
    assert is_engram_table_param("model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight")
    assert is_engram_table_param("language_model.model.layers.1.ple.ple_embedding.ngram_embedding.weight")
    assert is_engram_table_param("language_model.model.layers.1.ple.ple_embedding.ngram_embedding.weight_scale")
    assert is_engram_safetensors_key("model.language_model.layers.1.ple.ple_embedding.layer_multipliers")
    assert is_engram_safetensors_key("model.language_model.layers.1.ple.ple_embedding.ngram_heads_offsets")
    assert is_engram_safetensors_key("model.language_model.layers.1.ple.ple_embedding.ngram_heads_vocab_sizes")

    # DeepSeek-V4.1-Flash Engram lookup table & companion scale
    assert is_engram_table_param("layers.1.engram.embed.weight")
    assert is_engram_table_param("layers.1.engram.embed.scale")
    assert is_engram_table_param("language_model.model.layers.14.engram.embed_tokens.weight")
    assert is_engram_table_param("language_model.model.layers.14.engram.embed_tokens.weight_scale_inv")

    # Trainable PLE / Engram projections & ordinary token embeddings must NEVER match
    for non_table_key in (
        "model.language_model.layers.1.ple.key_proj.weight",
        "model.language_model.layers.1.ple.value_proj.weight",
        "model.language_model.layers.1.ple.kv_proj.weight",
        "model.language_model.layers.1.ple.norm_key.weight",
        "model.language_model.layers.1.ple.conv1d.weight",
        "layers.1.engram.q_weight",
        "layers.1.engram.k_weight",
        "layers.1.engram.wkv.weight",
        "layers.1.engram.wkv.scale",
        "embed.weight",
        "model.embed_tokens.weight",
    ):
        assert not is_engram_table_param(non_table_key), non_table_key
        assert not is_engram_safetensors_key(non_table_key), non_table_key

    # Companion scale buffer under a frozen Engram scope is collected, while trainable sibling stays excluded.
    toy = torch.nn.Module()
    toy.engram = torch.nn.Module()
    toy.engram.embed = torch.nn.Module()
    toy.engram.embed.weight = torch.nn.Parameter(torch.ones(4, 4), requires_grad=False)
    toy.engram.embed.register_buffer("scale", torch.ones(4, 1, dtype=torch.uint8))
    toy.engram.q_weight = torch.nn.Parameter(torch.ones(4, 4), requires_grad=True)

    frozen = collect_frozen_engram_keys(toy, toy.state_dict().keys())
    assert "engram.embed.weight" in frozen
    assert "engram.embed.scale" in frozen
    assert "engram.q_weight" not in frozen


def _write_sharded_engram_checkpoint(ckpt_dir: Path) -> tuple[Path, Path]:
    import json

    from safetensors.torch import save_file

    # Shard 1 holds layer 1 Engram FP8 weight + ue8m0 scale; Shard 2 holds Qwen4-Exp PLE shard + hash buffer.
    # Shard 0 is listed in index.json for non-Engram weights and intentionally NOT created on disk to prove
    # rollout streaming never opens non-Engram safetensors shards.
    shard1 = ckpt_dir / "model-00001-of-00002.safetensors"
    shard2 = ckpt_dir / "model-00002-of-00002.safetensors"

    save_file(
        {
            "layers.1.engram.embed.weight": torch.full((4, 8), 2.0, dtype=torch.float8_e4m3fn),
            "layers.1.engram.embed.scale": torch.full((4, 1), 127, dtype=torch.uint8).view(torch.float8_e8m0fnu),
            "layers.1.engram.q_weight": torch.full((4, 8), 9.0, dtype=torch.bfloat16),
        },
        str(shard1),
    )
    save_file(
        {
            "model.language_model.layers.2.ple.ple_embedding.ngram_embedding.shard_0.weight": torch.full(
                (4, 8), 3.0, dtype=torch.bfloat16
            ),
            "model.language_model.layers.2.ple.ple_embedding.layer_multipliers": torch.tensor(
                [3, 5, 7, 11], dtype=torch.int64
            ),
        },
        str(shard2),
    )
    index = {
        "weight_map": {
            "layers.0.attn.wq.weight": "model-00000-of-00002.safetensors",
            "layers.1.engram.embed.weight": shard1.name,
            "layers.1.engram.embed.scale": shard1.name,
            "layers.1.engram.q_weight": shard1.name,
            "model.language_model.layers.2.ple.ple_embedding.ngram_embedding.shard_0.weight": shard2.name,
            "model.language_model.layers.2.ple.ple_embedding.layer_multipliers": shard2.name,
        }
    }
    (ckpt_dir / "model.safetensors.index.json").write_text(json.dumps(index))
    return shard1, shard2


class _FakeEngramRolloutModel(torch.nn.Module):
    def __init__(self, *, include_layer1: bool = True, include_layer2: bool = True):
        super().__init__()
        self.loaded_batches: list[list[tuple[str, torch.Tensor]]] = []
        self.layers = torch.nn.ModuleDict()
        if include_layer1:
            l1 = torch.nn.Module()
            l1.engram = torch.nn.Module()
            l1.engram.embed_tokens = torch.nn.Module()
            l1.engram.embed_tokens.weight = torch.nn.Parameter(
                torch.zeros((4, 8), dtype=torch.float8_e4m3fn), requires_grad=False
            )
            l1.engram.embed_tokens.weight_scale_inv = torch.nn.Parameter(
                torch.zeros((4, 1), dtype=torch.uint8), requires_grad=False
            )
            l1.engram.q_weight = torch.nn.Parameter(torch.zeros((4, 8), dtype=torch.bfloat16), requires_grad=False)
            self.layers["1"] = l1
        if include_layer2:
            l2 = torch.nn.Module()
            l2.ple = torch.nn.Module()
            l2.ple.ple_embedding = torch.nn.Module()
            l2.ple.ple_embedding.ngram_embedding = torch.nn.Module()
            l2.ple.ple_embedding.ngram_embedding.weight = torch.nn.Parameter(
                torch.zeros((4, 8), dtype=torch.bfloat16), requires_grad=False
            )
            l2.ple.ple_embedding.register_buffer("layer_multipliers", torch.zeros(4, dtype=torch.int64))
            self.layers["2"] = l2

    def load_weights(self, weights):
        batch = list(weights)
        self.loaded_batches.append(batch)
        for name, tensor in batch:
            if name == "layers.1.engram.embed.weight" and "1" in self.layers:
                self.layers["1"].engram.embed_tokens.weight.data.copy_(tensor)
            elif name == "layers.1.engram.embed.scale" and "1" in self.layers:
                self.layers["1"].engram.embed_tokens.weight_scale_inv.data.copy_(tensor.view(torch.uint8))
            elif name == "layers.1.engram.q_weight" and "1" in self.layers:
                self.layers["1"].engram.q_weight.data.copy_(tensor)
            elif name.endswith("ngram_embedding.shard_0.weight") and "2" in self.layers:
                self.layers["2"].ple.ple_embedding.ngram_embedding.weight.data.copy_(tensor)
            elif name.endswith("ple_embedding.layer_multipliers") and "2" in self.layers:
                self.layers["2"].ple.ple_embedding.layer_multipliers.copy_(tensor)


def _make_engram_worker(model: torch.nn.Module, ckpt_dir: Path, *, load_format: str = "dummy"):
    class _BaseWorker:
        def __init__(self):
            self.sleep_levels: list[int] = []

        def sleep(self, level: int = 1):
            self.sleep_levels.append(level)

    class _ComposedWorker(_BaseWorker, vLLMColocateWorkerExtension):
        pass

    model_cfg = types.SimpleNamespace(
        model=str(ckpt_dir),
        model_weights=None,
        revision=None,
        sleep_preserve_parameter_names=(),
    )
    load_cfg = types.SimpleNamespace(load_format=load_format, download_dir=None)
    vllm_cfg = types.SimpleNamespace(
        model_config=model_cfg,
        load_config=load_cfg,
        speculative_config=None,
        weight_transfer_config=None,
    )
    worker = _ComposedWorker.__new__(_ComposedWorker, vllm_config=vllm_cfg)
    _BaseWorker.__init__(worker)
    worker.device = torch.device("cpu")
    worker.local_rank = 0
    worker.model_runner = types.SimpleNamespace(model=model, vllm_config=vllm_cfg)
    worker._get_zmq_handle = lambda: "unused"
    return worker


def test_rollout_streams_engram_once_for_cpu_offload_and_skips_non_engram_shards(monkeypatch, tmp_path):
    _write_sharded_engram_checkpoint(tmp_path)
    loader_utils = types.ModuleType("vllm.model_executor.model_loader.utils")
    post_events: list[str] = []
    loader_utils.process_weights_after_loading = lambda *args: post_events.append("post")
    monkeypatch.setitem(sys.modules, loader_utils.__name__, loader_utils)

    ipc_payload: list[tuple[str, torch.Tensor]] = [
        ("layers.1.engram.q_weight", torch.full((4, 8), 5.0, dtype=torch.bfloat16))
    ]

    class _Receiver:
        def __init__(self, **kwargs):
            pass

        def receive_weights(self, on_bucket_received):
            on_bucket_received(ipc_payload, True)

    monkeypatch.setattr(_vllm_rollout_utils, "BucketedWeightReceiver", _Receiver)

    model = _FakeEngramRolloutModel()
    worker = _make_engram_worker(model, tmp_path, load_format="dummy")

    # Initial sync under dummy format loads IPC trainable weights + streams Engram/PLE from safetensors.
    worker.update_weights_from_ipc()
    loaded_names = [name for batch in model.loaded_batches for name, _ in batch]
    assert loaded_names == [
        "layers.1.engram.q_weight",
        "layers.1.engram.embed.weight",
        "layers.1.engram.embed.scale",
        "model.language_model.layers.2.ple.ple_embedding.ngram_embedding.shard_0.weight",
        "model.language_model.layers.2.ple.ple_embedding.layer_multipliers",
    ]
    assert torch.all(model.layers["1"].engram.embed_tokens.weight_scale_inv == 127)
    assert torch.all(model.layers["2"].ple.ple_embedding.ngram_embedding.weight == 3.0)
    assert model.layers["2"].ple.ple_embedding.layer_multipliers.tolist() == [3, 5, 7, 11]
    assert post_events == ["post"]

    # CPU-offloaded Engram/PLE (param.is_cpu == True) survives both sleep(level=1) and sleep(level=2):
    # subsequent weight syncs must NOT re-read safetensors.
    model.loaded_batches.clear()
    worker.sleep(level=1)
    worker.sleep(level=2)
    worker.update_weights_from_ipc()
    assert [name for batch in model.loaded_batches for name, _ in batch] == ["layers.1.engram.q_weight"]


def test_rollout_reloads_device_resident_engram_only_after_level2_sleep(monkeypatch, tmp_path):
    _write_sharded_engram_checkpoint(tmp_path)
    loader_utils = types.ModuleType("vllm.model_executor.model_loader.utils")
    loader_utils.process_weights_after_loading = lambda *args: None
    monkeypatch.setitem(sys.modules, loader_utils.__name__, loader_utils)

    class _Receiver:
        def __init__(self, **kwargs):
            pass

        def receive_weights(self, on_bucket_received):
            on_bucket_received([], True)

    monkeypatch.setattr(_vllm_rollout_utils, "BucketedWeightReceiver", _Receiver)

    # Simulate device-resident Engram/PLE parameters on CPU CI by overriding Tensor.is_cpu -> False on the PLE weight.
    class _DeviceResidentFakeParam(torch.nn.Parameter):
        @property
        def is_cpu(self):
            return False

    model = _FakeEngramRolloutModel(include_layer1=False, include_layer2=True)
    model.layers["2"].ple.ple_embedding.ngram_embedding.weight = _DeviceResidentFakeParam(
        torch.zeros((4, 8), dtype=torch.bfloat16), requires_grad=False
    )

    # With load_format="safetensors", initial sync skips safetensors reload when no level-2 sleep occurred.
    worker = _make_engram_worker(model, tmp_path, load_format="safetensors")
    worker.update_weights_from_ipc()
    assert model.loaded_batches == []

    # Level-1 sleep preserves device weights -> still no safetensors reload.
    worker.sleep(level=1)
    worker.update_weights_from_ipc()
    assert model.loaded_batches == []

    # Level-2 sleep discards unpreserved device weights -> next sync reloads ONLY layer 2's shard!
    # (Also verifies PP layer filtering: shard 1 for missing layer 1 is NOT opened.)
    (tmp_path / "model-00001-of-00002.safetensors").unlink()
    worker.sleep(level=2)
    worker.update_weights_from_ipc()
    loaded_names = [name for batch in model.loaded_batches for name, _ in batch]
    assert loaded_names == [
        "model.language_model.layers.2.ple.ple_embedding.ngram_embedding.shard_0.weight",
        "model.language_model.layers.2.ple.ple_embedding.layer_multipliers",
    ]


def test_rollout_does_not_overwrite_trainable_engram_sent_over_ipc(monkeypatch, tmp_path):
    _write_sharded_engram_checkpoint(tmp_path)
    loader_utils = types.ModuleType("vllm.model_executor.model_loader.utils")
    loader_utils.process_weights_after_loading = lambda *args: None
    monkeypatch.setitem(sys.modules, loader_utils.__name__, loader_utils)

    trained_ple = torch.full((4, 8), 42.0, dtype=torch.bfloat16)

    class _Receiver:
        def __init__(self, **kwargs):
            pass

        def receive_weights(self, on_bucket_received):
            on_bucket_received(
                [("model.language_model.layers.2.ple.ple_embedding.ngram_embedding.shard_0.weight", trained_ple)],
                True,
            )

    monkeypatch.setattr(_vllm_rollout_utils, "BucketedWeightReceiver", _Receiver)

    model = _FakeEngramRolloutModel(include_layer1=False, include_layer2=True)
    worker = _make_engram_worker(model, tmp_path, load_format="dummy")
    worker.update_weights_from_ipc()

    # Checkpoint has 3.0; trainer IPC sent 42.0 -> must stay 42.0 (not overwritten by safetensors).
    assert torch.all(model.layers["2"].ple.ple_embedding.ngram_embedding.weight == 42.0)


def test_veomni_engine_and_shard_export_skip_frozen_engram(monkeypatch, tmp_path):
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Shard, distribute_tensor

    fake_ps = types.SimpleNamespace(ep_enabled=False, ep_size=1, ep_rank=0)
    fake_veomni = types.ModuleType("veomni")
    fake_veomni_args = types.ModuleType("veomni.arguments")
    for cls_name in ("AcceleratorConfig", "FSDPConfig", "MixedPrecisionConfig", "OpsImplementationConfig"):
        setattr(fake_veomni_args, cls_name, type(cls_name, (), {}))
    fake_veomni_dist = types.ModuleType("veomni.distributed")
    fake_veomni_ps = types.ModuleType("veomni.distributed.parallel_state")
    fake_veomni_ps.get_parallel_state = lambda: fake_ps
    fake_veomni_dist.parallel_state = fake_veomni_ps
    fake_veomni_tp = types.ModuleType("veomni.distributed.torch_parallelize")
    fake_veomni_tp.build_parallelize_model = lambda *a, **kw: None
    fake_veomni_models = types.ModuleType("veomni.models")
    fake_veomni_auto = types.ModuleType("veomni.models.auto")
    fake_veomni_auto.build_foundation_model = lambda *a, **kw: None
    fake_veomni_ckpt = types.ModuleType("veomni.models.checkpoint")
    fake_veomni_ckpt.get_checkpoint_tensor_converter = lambda m: getattr(m, "_converter", None)
    fake_veomni_optim = types.ModuleType("veomni.optim")
    fake_veomni_optim.build_lr_scheduler = lambda *a, **kw: None
    fake_veomni_optim.build_optimizer = lambda *a, **kw: None
    fake_veomni_utils = types.ModuleType("veomni.utils")
    fake_veomni_seqlen = types.ModuleType("veomni.utils.seqlen_pos_transform_utils")
    fake_veomni_seqlen.prepare_fa_kwargs_from_position_ids = lambda *a, **kw: None

    for mod_name, mod_obj in {
        "veomni": fake_veomni,
        "veomni.arguments": fake_veomni_args,
        "veomni.distributed": fake_veomni_dist,
        "veomni.distributed.parallel_state": fake_veomni_ps,
        "veomni.distributed.torch_parallelize": fake_veomni_tp,
        "veomni.models": fake_veomni_models,
        "veomni.models.auto": fake_veomni_auto,
        "veomni.models.checkpoint": fake_veomni_ckpt,
        "veomni.optim": fake_veomni_optim,
        "veomni.utils": fake_veomni_utils,
        "veomni.utils.seqlen_pos_transform_utils": fake_veomni_seqlen,
    }.items():
        monkeypatch.setitem(sys.modules, mod_name, mod_obj)

    from verl.workers.engine.veomni.transformer_impl import VeOmniEngine
    from verl.workers.engine.veomni.utils import veomni_shard_export

    monkeypatch.setattr("verl.workers.engine.veomni.transformer_impl.get_device_id", lambda: "cpu")

    dist.init_process_group("gloo", init_method=f"file://{tmp_path / 'veomni_rendezvous'}", rank=0, world_size=1)
    try:
        mesh = init_device_mesh("cpu", (1,))
        module = torch.nn.Module()
        module.config = types.SimpleNamespace(model_type="qwen4_exp")
        module.ple = torch.nn.Module()
        module.ple.ple_embedding = torch.nn.Module()
        module.ple.ple_embedding.ngram_embedding = torch.nn.ModuleDict()
        shard0 = torch.nn.Module()
        raw = torch.arange(24, dtype=torch.bfloat16).reshape(6, 4)
        shard0.weight = torch.nn.Parameter(distribute_tensor(raw, mesh, [Shard(0)]), requires_grad=False)
        module.ple.ple_embedding.ngram_embedding["shard_0"] = shard0
        module.ple.key_proj = torch.nn.Linear(4, 4, bias=False)
        trainable_shard = torch.nn.Module()
        trainable_shard.weight = torch.nn.Parameter(
            distribute_tensor(torch.ones((4, 4), dtype=torch.bfloat16), mesh, [Shard(0)]),
            requires_grad=True,
        )
        module.ple_trainable = trainable_shard
        module._fqn2spec_info = {
            "ple.ple_embedding.ngram_embedding.shard_0.weight": types.SimpleNamespace(
                para_name="ple", placement=Shard(0)
            ),
            "ple_trainable.weight": types.SimpleNamespace(para_name="ple", placement=Shard(0)),
        }
        module._converter = types.SimpleNamespace(
            can_handle=lambda name: False,
            convert=lambda name, t: None,
            crop_hf_export_tensor=lambda name, t: t[:5] if "ngram_embedding.shard_0.weight" in name else t,
        )
        module._create_checkpoint_tensor_converter = lambda m: m._converter

        orig_full_tensor = torch.distributed.tensor.DTensor.full_tensor
        ple_full_tensor_calls = 0

        def _guarded_full_tensor(self, *args, **kwargs):
            nonlocal ple_full_tensor_calls
            if tuple(self.shape) == (6, 4):
                ple_full_tensor_calls += 1
            return orig_full_tensor(self, *args, **kwargs)

        monkeypatch.setattr(torch.distributed.tensor.DTensor, "full_tensor", _guarded_full_tensor)

        engine = object.__new__(VeOmniEngine)
        engine.module = module
        engine._uses_fsdp2_cpu_offload_policy = True
        engine._is_offload_param = False
        engine._rollout_weight_quantizer = None

        # 1. Rollout weight sync (get_per_tensor_param) skips frozen PLE without calling full_tensor().
        rollout_weights, _ = engine.get_per_tensor_param()
        rollout_dict = dict(rollout_weights)
        assert ple_full_tensor_calls == 0
        assert "ple.ple_embedding.ngram_embedding.shard_0.weight" not in rollout_dict
        assert "ple.key_proj.weight" in rollout_dict

        # 2. Checkpoint hf_model export (_get_hf_export_per_tensor_param) keeps and crops PLE.
        ckpt_weights, _ = engine._get_hf_export_per_tensor_param()
        ckpt_dict = dict(ckpt_weights)
        assert ple_full_tensor_calls == 1
        assert ckpt_dict["ple.ple_embedding.ngram_embedding.shard_0.weight"].shape == (5, 4)

        # 3. Delta shard export skips frozen PLE and treats para_name='ple' Shard(0) as dense (not MoE expert stack).
        shard_gen, _ = veomni_shard_export(module)
        shard_items = {name: (flat, spec) for name, flat, spec in shard_gen}
        assert "ple.ple_embedding.ngram_embedding.shard_0.weight" not in shard_items
        assert "ple_trainable.weight" in shard_items
        assert shard_items["ple_trainable.weight"][1].to_hf_chunk is None

        # 4. Default weight-loading flags and PLE validation.
        from verl.workers.config.engine import VeOmniEngineConfig

        default_cfg = VeOmniEngineConfig()
        assert default_cfg.broadcast_model_weights_from_rank0 is False
        assert default_cfg.ep_sharded_stream_load is True
        with pytest.raises(ValueError, match="ple_parallel_size > 1 requires"):
            VeOmniEngineConfig(ple_parallel_size=2, broadcast_model_weights_from_rank0=True)
        with pytest.raises(ValueError, match="ple_parallel_size > 1 requires"):
            VeOmniEngineConfig(ple_parallel_size=2, ep_sharded_stream_load=False)
    finally:
        dist.destroy_process_group()
