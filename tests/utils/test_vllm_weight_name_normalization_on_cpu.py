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

"""CPU tests for the receiver-side weight-name normalization in
``vLLMColocateWorkerExtension``.

With LoRA enabled, vLLM wraps every linear layer (exposing base weights under
``.base_layer``), while Megatron only LoRA-wraps the user's target_modules and
exports plain names for the rest (e.g. the Qwen3.5-VL vision merger). The
receiver reconciles each name against the live vLLM namespace rather than a
hard-coded list. Imports only from ``vllm_rollout/utils`` (deps stubbed) to
prove the receiver is decoupled from ``megatron_peft_utils``.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_weight_update_utils():
    module_path = _REPO_ROOT / "verl/workers/rollout/vllm_rollout/weight_update_utils.py"
    spec = importlib.util.spec_from_file_location("weight_update_utils", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_vllm_rollout_utils():
    """Load ``vllm_rollout/utils.py`` with heavyweight deps stubbed (no GPU/vLLM needed)."""
    module_name = "verl.workers.rollout.vllm_rollout.utils"
    module_path = _REPO_ROOT / "verl/workers/rollout/vllm_rollout/utils.py"

    fake_outputs = types.ModuleType("vllm.outputs")

    class _FakeRequestOutput:
        pass

    fake_outputs.RequestOutput = _FakeRequestOutput
    fake_vllm = types.ModuleType("vllm")
    fake_vllm.outputs = fake_outputs

    # Stub ``vllm.lora.layers.base.BaseLayerWithLoRA`` WITHOUT ``load_weights``
    # so the module-level probe falls to False, matching released vLLM. Tests
    # that exercise the newer-vLLM path monkeypatch ``_HAS_LORA_LOAD_WEIGHTS``.
    fake_lora_base = types.ModuleType("vllm.lora.layers.base")

    class _FakeBaseLayerWithLoRA:
        pass

    fake_lora_base.BaseLayerWithLoRA = _FakeBaseLayerWithLoRA
    fake_lora_layers = types.ModuleType("vllm.lora.layers")
    fake_lora_layers.base = fake_lora_base
    fake_lora = types.ModuleType("vllm.lora")
    fake_lora.layers = fake_lora_layers
    fake_vllm.lora = fake_lora

    fake_vllm_third_party = types.ModuleType("verl.third_party.vllm")
    fake_vllm_third_party.VLLM_SLEEP_LEVEL = 1
    fake_vllm_third_party.get_version = lambda pkg: "0.8.0"

    fake_vllm_utils = types.ModuleType("verl.utils.vllm")

    class _FakeTensorLoRARequest:
        def __init__(self, *args, **kwargs):
            self.lora_name = kwargs.get("lora_name")
            self.lora_int_id = kwargs.get("lora_int_id")
            self.lora_path = kwargs.get("lora_path")
            self.peft_config = kwargs.get("peft_config")
            self.lora_tensors = kwargs.get("lora_tensors")

    class _FakeVLLMHijack:
        @staticmethod
        def hijack():
            return None

    fake_vllm_utils.TensorLoRARequest = _FakeTensorLoRARequest
    fake_vllm_utils.VLLMHijack = _FakeVLLMHijack

    fake_vllm_patch = types.ModuleType("verl.utils.vllm.patch")
    fake_vllm_patch.patch_vllm_moe_model_weight_loader = lambda model: None

    fake_vllm_quant = types.ModuleType("verl.utils.vllm.vllm_quant_utils")
    fake_vllm_quant.apply_vllm_quant_patches = lambda: None
    fake_vllm_quant.is_fp8_model = lambda config: False
    fake_vllm_quant.load_quanted_weights = lambda *a, **k: []

    # NOTE: deliberately do NOT stub verl.plugin.platform. It is lightweight and
    # imports fine on CPU. verl.utils.device binds `get_platform` at import time, so
    # a `lambda: None` stub gets baked into verl.utils.device during this window;
    # because only the keys in `fakes` below are restored, that fake would leak
    # process-wide and later tests would crash in get_device_name() with
    # "'NoneType' object has no attribute 'device_name'".

    # Minimal stubs for the vllm.lora.* imports pulled in by
    # ``verl/utils/vllm/utils.py`` (which also defines ``resolve_weight_name``).
    fake_lora_request = types.ModuleType("vllm.lora.request")
    fake_lora_request.LoRARequest = type("LoRARequest", (), {})
    fake_lora_utils = types.ModuleType("vllm.lora.utils")
    fake_lora_utils.get_adapter_absolute_path = lambda p: p
    fake_lora_worker_manager = types.ModuleType("vllm.lora.worker_manager")
    fake_lora_worker_manager.LRUCacheWorkerLoRAManager = type("LRUCacheWorkerLoRAManager", (), {})
    fake_lora_model = types.ModuleType("vllm.lora.lora_model")
    fake_lora_model.LoRAModel = type("LoRAModel", (), {})
    fake_lora_models = types.ModuleType("vllm.lora.models")
    fake_lora_models.LoRAModel = type("LoRAModel", (), {})

    fakes = {
        "vllm": fake_vllm,
        "vllm.outputs": fake_outputs,
        "vllm.lora": fake_lora,
        "vllm.lora.layers": fake_lora_layers,
        "vllm.lora.layers.base": fake_lora_base,
        "vllm.lora.request": fake_lora_request,
        "vllm.lora.utils": fake_lora_utils,
        "vllm.lora.worker_manager": fake_lora_worker_manager,
        "vllm.lora.lora_model": fake_lora_model,
        "vllm.lora.models": fake_lora_models,
        "verl.third_party.vllm": fake_vllm_third_party,
        "verl.utils.vllm": fake_vllm_utils,
        "verl.utils.vllm.patch": fake_vllm_patch,
        "verl.utils.vllm.vllm_quant_utils": fake_vllm_quant,
        "verl.workers.rollout.vllm_rollout.weight_update_utils": _weight_update_utils,
    }

    saved = {name: sys.modules.get(name) for name in fakes}
    try:
        sys.modules.update(fakes)
        # Load the real ``verl/utils/vllm/utils.py`` under the stubbed deps so
        # ``vllm_rollout/utils.py``'s ``from verl.utils.vllm import resolve_weight_name``
        # resolves to the genuine function (with ``_HAS_LORA_LOAD_WEIGHTS`` False
        # since the stub ``BaseLayerWithLoRA`` has no load_weights).
        utils_path = _REPO_ROOT / "verl/utils/vllm/utils.py"
        utils_spec = importlib.util.spec_from_file_location("verl.utils.vllm.utils_real", utils_path)
        utils_module = importlib.util.module_from_spec(utils_spec)
        assert utils_spec is not None and utils_spec.loader is not None
        utils_spec.loader.exec_module(utils_module)
        fake_vllm_utils.resolve_weight_name = utils_module.resolve_weight_name
        fake_vllm_utils._HAS_LORA_LOAD_WEIGHTS = utils_module._HAS_LORA_LOAD_WEIGHTS
        # The stubbed vllm has no real RoutedExperts, so the probe falls to
        # False (older vLLM). Expose it so tests can flip it via monkeypatch.
        fake_vllm_utils._HAS_LORA_BASE_LAYER_PREFIX = getattr(utils_module, "_HAS_LORA_BASE_LAYER_PREFIX", False)

        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        spec.loader.exec_module(module)
        module._vllm_utils_real = utils_module
    finally:
        for name, prev in saved.items():
            if prev is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prev
    return module


_weight_update_utils = _load_weight_update_utils()
_vllm_rollout_utils = _load_vllm_rollout_utils()
vLLMColocateWorkerExtension = _vllm_rollout_utils.vLLMColocateWorkerExtension
# The real ``verl/utils/vllm/utils.py`` module hosting ``resolve_weight_name``
# (loaded under stubbed deps); monkeypatch its ``_HAS_LORA_LOAD_WEIGHTS`` to flip
# the vLLM-version branch under test.
_vllm_utils_real = _vllm_rollout_utils._vllm_utils_real
resolve_weight_name = _vllm_utils_real.resolve_weight_name


class _FakeMapper:
    """Minimal stand-in for vLLM ``WeightsMapper``.

    Substring-based ``orig_to_new_substr`` / ``orig_to_new_stacked`` mapping:
    each key is a substring replaced (once) in the name, matching how the real
    mapper rewrites ``in_proj_qkv`` -> ``in_proj_qkvz``.
    """

    def __init__(self, mapping: dict[str, str]):
        self.mapping = mapping

    def apply_list(self, names: list[str]) -> list[str]:
        out = []
        for name in names:
            mapped = name
            for substr, new in self.mapping.items():
                if substr in mapped:
                    mapped = mapped.replace(substr, new, 1)
            out.append(mapped)
        return out


class _FakeModel:
    """A fake vLLM model exposing only the introspection surface the resolver uses."""

    def __init__(self, params: dict[str, torch.Tensor], *, mapper=None, packed=None, routed_experts=()):
        # name -> tensor (for named_parameters / named_buffers)
        self._params = dict(params)
        self.hf_to_vllm_mapper = mapper
        self.packed_modules_mapping = packed
        # Stand-ins for vLLM ``RoutedExperts`` submodules.
        self._routed_experts = list(routed_experts)

    def modules(self):
        yield self
        yield from self._routed_experts

    def named_parameters(self, remove_duplicate: bool = False):
        del remove_duplicate
        yield from self._params.items()

    def named_buffers(self):
        return iter(())

    def load_weights(self, weights):
        # Record resolved names for assertions.
        self.loaded = [name for name, _ in weights]


class _FakeVllmConfig:
    def __init__(self, speculative_config=None, quant_config=None):
        self.speculative_config = speculative_config
        self.quant_config = quant_config
        self.model_config = types.SimpleNamespace()


def _make_worker(model):
    worker = object.__new__(vLLMColocateWorkerExtension)
    worker.model_runner = types.SimpleNamespace(model=model, vllm_config=_FakeVllmConfig(), drafter=None)
    return worker


def _resolve(worker, model, name):
    del worker  # resolve_weight_name is a pure function (no worker state)
    names = {n for n, _ in model.named_parameters(remove_duplicate=False)}
    names.update(n for n, _ in model.named_buffers())
    return resolve_weight_name(model, name, names)


def _mapped(worker, model, name):
    mapper = getattr(model, "hf_to_vllm_mapper", None)
    if mapper is None:
        return name
    out = mapper.apply_list([name])
    return out[0] if out else name


# ---------------------------------------------------------------------------
# The exact nohup.out failure: vision merger leaf param with .base_layer must
# resolve to the unwrapped name.
# ---------------------------------------------------------------------------


def test_vision_merger_base_layer_is_stripped():
    """Qwen3.5-VL ``merger.linear_fc1.base_layer.weight`` -> ``merger.linear_fc1.weight``."""
    model = _FakeModel(
        {
            "merger.linear_fc1.weight": torch.empty(0),
            "merger.linear_fc1.bias": torch.empty(0),
        }
    )
    worker = _make_worker(model)

    assert _resolve(worker, model, "merger.linear_fc1.base_layer.weight") == "merger.linear_fc1.weight"
    assert _resolve(worker, model, "merger.linear_fc1.base_layer.bias") == "merger.linear_fc1.bias"


# ---------------------------------------------------------------------------
# Mixed LoRA model: language layers are LoRA-wrapped (have .base_layer), but
# the vision merger is a plain linear (no base_layer child). On vLLM 0.26.0
# the suffix must be stripped ONLY for the merger, not kept model-wide.
# ---------------------------------------------------------------------------


def test_mixed_model_vision_merger_strips_on_released_vllm():
    """vLLM 0.26.0 (_HAS_LORA_LOAD_WEIGHTS=False) with a mixed model: the
    vision merger's ``.base_layer.`` is stripped (no real base_layer child),
    while a coexisting LoRA-wrapped language leaf keeps the suffix."""
    # _HAS_LORA_LOAD_WEIGHTS defaults to False here (stub base layer), matching
    # vLLM 0.26.0.
    model = _FakeModel(
        {
            "merger.linear_fc1.weight": torch.empty(0),
            "merger.linear_fc1.bias": torch.empty(0),
            "language_model.model.layers.0.self_attn.q_proj.base_layer.weight": torch.empty(0),
        }
    )
    worker = _make_worker(model)

    assert _resolve(worker, model, "merger.linear_fc1.base_layer.weight") == "merger.linear_fc1.weight"
    assert _resolve(worker, model, "merger.linear_fc1.base_layer.bias") == "merger.linear_fc1.bias"
    # The language leaf keeps .base_layer on vLLM 0.26.0.
    name = "language_model.model.layers.0.self_attn.q_proj.base_layer.weight"
    assert _resolve(worker, model, name) == name


def test_mixed_model_with_prefix_rewriting_mapper():
    """The Qwen3.5-VL regression: the Bridge exports under ``model.language_model.``
    / ``model.visual.`` while the live vLLM namespace is ``language_model.model.``
    / ``visual.``. The parent-has-base-layer check must consult the
    *mapper-rewritten* name, else the LoRA-wrapped language leaf's real
    ``base_layer`` is invisible under the Bridge prefix and gets wrongly stripped,
    crashing ``AutoWeightsLoader``."""
    mapper = _FakeMapper(
        {
            # substring replacements approximating WeightsMapper orig_to_new_prefix
            "model.visual.": "visual.",
            "model.language_model.": "language_model.model.",
        }
    )
    model = _FakeModel(
        {
            "language_model.model.layers.0.mlp.gate.base_layer.weight": torch.empty(0),
            "visual.merger.linear_fc1.weight": torch.empty(0),
            "visual.merger.linear_fc1.bias": torch.empty(0),
        },
        mapper=mapper,
    )
    worker = _make_worker(model)

    # Language leaf: incoming Bridge prefix differs from vLLM's; the real
    # base_layer child exists under the rewritten prefix -> KEEP the suffix.
    name = "model.language_model.layers.0.mlp.gate.base_layer.weight"
    assert _resolve(worker, model, name) == name
    # Merger: no base_layer child under either prefix -> STRIP.
    assert _resolve(worker, model, "model.visual.merger.linear_fc1.base_layer.weight") == (
        "model.visual.merger.linear_fc1.weight"
    )
    assert _resolve(worker, model, "model.visual.merger.linear_fc1.base_layer.bias") == (
        "model.visual.merger.linear_fc1.bias"
    )


# ---------------------------------------------------------------------------
# LoRA-wrapped language params KEEP .base_layer (the suffix is real there).
# ---------------------------------------------------------------------------


def test_lora_wrapped_base_layer_gets_stripped(monkeypatch):
    """A LoRA-wrapped leaf with .base_layer is stripped: BaseLayerWithLoRA.load_weights
    delegates to AutoWeightsLoader(base_layer) which expects the inner name."""
    monkeypatch.setattr(_vllm_utils_real, "_HAS_LORA_LOAD_WEIGHTS", True)
    model = _FakeModel(
        {
            "language_model.model.layers.0.self_attn.q_proj.base_layer.weight": torch.empty(0),
        }
    )
    worker = _make_worker(model)

    name = "language_model.model.layers.0.self_attn.q_proj.base_layer.weight"
    assert _resolve(worker, model, name) == ("language_model.model.layers.0.self_attn.q_proj.weight")


def test_missing_base_layer_not_added_on_newer_vllm(monkeypatch):
    """On newer vLLM, an incoming name without .base_layer is NOT suffixed —
    BaseLayerWithLoRA.load_weights delegates to AutoWeightsLoader(base_layer)
    expecting inner names."""
    monkeypatch.setattr(_vllm_utils_real, "_HAS_LORA_LOAD_WEIGHTS", True)
    model = _FakeModel(
        {
            "language_model.model.layers.0.self_attn.q_proj.base_layer.weight": torch.empty(0),
        }
    )
    worker = _make_worker(model)

    name = "language_model.model.layers.0.self_attn.q_proj.weight"
    assert _resolve(worker, model, name) == name


def test_missing_base_layer_is_added_on_released_vllm():
    """On released vLLM (no BaseLayerWithLoRA.load_weights), an incoming leaf
    without ``.base_layer`` is suffixed so load_weights recurses into the
    ``base_layer`` child and matches the real param."""
    model = _FakeModel(
        {
            "language_model.model.layers.0.self_attn.q_proj.base_layer.weight": torch.empty(0),
        }
    )
    worker = _make_worker(model)

    name = "language_model.model.layers.0.self_attn.q_proj.weight"
    assert _resolve(worker, model, name) == ("language_model.model.layers.0.self_attn.q_proj.base_layer.weight")


def test_suffixed_leaf_is_kept_on_released_vllm():
    """On released vLLM, an incoming ``.base_layer.`` leaf is kept as-is
    (the loader recurses and matches the suffixed param)."""
    model = _FakeModel(
        {
            "language_model.model.layers.0.self_attn.q_proj.base_layer.weight": torch.empty(0),
        }
    )
    worker = _make_worker(model)

    name = "language_model.model.layers.0.self_attn.q_proj.base_layer.weight"
    assert _resolve(worker, model, name) == name


def test_packed_shard_leaf_suffixed_on_strict_released_vllm():
    """Regression: vLLM 0.24.0 (strict ``params_dict[name]`` loader, no
    ``BaseLayerWithLoRA.load_weights``) + Llama-arch + merge=False LoRA. The
    Bridge exports the *shard* leaf (``gate_proj``/``q_proj``); vLLM's
    ``stacked_params_mapping`` fuses it to the packed owner
    (``gate_up_proj``/``qkv_proj``) and indexes ``params_dict`` *after* the
    rewrite. The live param lives under ``.base_layer.`` (LoRA-wrapped), so the
    shard leaf must be suffixed before vLLM fuses it -- otherwise the fused name
    ``gate_up_proj.weight`` (no ``.base_layer``) KeyErrors. The packed-owner
    lookup must probe the wrapped form (``gate_up_proj.base_layer.weight``),
    not just the bare owner."""
    # _HAS_LORA_LOAD_WEIGHTS is False by default (released vLLM); _strict_outer
    # gives a strict inner loader (params_dict[name], no AutoWeightsLoader).
    model = _strict_outer(
        {
            "layers.0.self_attn.qkv_proj.base_layer.weight": torch.empty(0),
            "layers.0.mlp.gate_up_proj.base_layer.weight": torch.empty(0),
        }
    )
    model.packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }
    worker = _make_worker(model)

    # Shard leaves (what the Bridge exports) -> suffixed so vLLM fuses to the
    # wrapped owner key.
    assert _resolve(worker, model, "layers.0.mlp.gate_proj.weight") == ("layers.0.mlp.gate_proj.base_layer.weight")
    assert _resolve(worker, model, "layers.0.self_attn.q_proj.weight") == (
        "layers.0.self_attn.q_proj.base_layer.weight"
    )
    # Owner leaves are suffixed directly.
    assert _resolve(worker, model, "layers.0.mlp.gate_up_proj.weight") == (
        "layers.0.mlp.gate_up_proj.base_layer.weight"
    )


# ---------------------------------------------------------------------------
# Non-leaf fused-MoE expert alias: strip Bridge-inserted .base_layer.
# ---------------------------------------------------------------------------


def test_expert_alias_base_layer_gets_stripped():
    """``experts.base_layer.gate_up_proj`` (non-leaf) -> ``experts.gate_up_proj`` when that exists."""
    model = _FakeModel(
        {
            "language_model.model.layers.0.mlp.experts.gate_up_proj": torch.empty(0),
        }
    )
    worker = _make_worker(model)

    name = "language_model.model.layers.0.mlp.experts.base_layer.gate_up_proj"
    assert _resolve(worker, model, name) == "language_model.model.layers.0.mlp.experts.gate_up_proj"


def test_packed_routed_expert_alias_not_routed_with_lora():
    """Qwen3.5 routed experts: Bridge exports packed ``experts.gate_up_proj``
    (non-leaf). On vLLM with ``BaseLayerWithLoRA.load_weights`` (``_HAS_LORA_LOAD_WEIGHTS``),
    ``FusedMoEWithLoRA.load_weights`` delegates to ``MoERunner.load_weights`` which
    handles the packed alias directly — injecting ``.base_layer.`` would make the
    name ``experts.base_layer.gate_up_proj``, which ``MoERunner`` can't match (it
    expects bare ``gate_up_proj``), silently dropping every expert. The ``isdigit``
    guard keeps the packed alias unchanged even when ``base_layer.routed_experts``
    is live in the namespace."""
    model = _FakeModel(
        {
            "language_model.model.layers.0.mlp.experts.base_layer.routed_experts.w13_weight": torch.empty(0),
            "language_model.model.layers.0.mlp.experts.base_layer.routed_experts.w2_weight": torch.empty(0),
        }
    )
    worker = _make_worker(model)

    incoming = "language_model.model.layers.0.mlp.experts.gate_up_proj"
    assert _resolve(worker, model, incoming) == incoming
    assert _resolve(worker, model, "language_model.model.layers.0.mlp.experts.down_proj") == (
        "language_model.model.layers.0.mlp.experts.down_proj"
    )


def test_packed_routed_expert_alias_not_routed_without_lora():
    """Without a LoRA experts.base_layer, packed aliases stay unchanged
    (AutoWeightsLoader dispatches directly to the MoERunner)."""
    model = _FakeModel({"language_model.model.layers.0.mlp.experts.gate_up_proj": torch.empty(0)})
    worker = _make_worker(model)

    incoming = "language_model.model.layers.0.mlp.experts.gate_up_proj"
    assert _resolve(worker, model, incoming) == incoming


def test_per_expert_leaf_routed_through_base_layer(monkeypatch):
    """Flat-loader (Qwen3Moe) per-expert routed leaves emit the *leaf* form
    ``experts.<id>.<proj>.base_layer.<leaf>`` on vLLM carrying
    ``lora_base_layer_prefix`` (#31104). Both Bridge-exported forms normalize
    to it: the HF-native leaf gains the suffix, the leaf form passes through.
    """
    model = _FakeModel(
        {
            "model.layers.0.mlp.experts.base_layer.routed_experts.w13_weight": torch.empty(0),
            "model.layers.0.mlp.experts.base_layer.routed_experts.w2_weight": torch.empty(0),
            "model.layers.0.mlp.experts.base_layer.gate.weight": torch.empty(0),
        }
    )
    worker = _make_worker(model)
    monkeypatch.setattr(_vllm_utils_real, "_HAS_LORA_BASE_LAYER_PREFIX", True)

    for sub in ("gate_proj", "up_proj", "down_proj"):
        # Leaf form passes through.
        incoming = f"model.layers.0.mlp.experts.0.{sub}.base_layer.weight"
        assert _resolve(worker, model, incoming) == incoming
        # HF-native form gains the suffix at the leaf.
        incoming = f"model.layers.0.mlp.experts.0.{sub}.weight"
        assert _resolve(worker, model, incoming) == (f"model.layers.0.mlp.experts.0.{sub}.base_layer.weight")


def test_per_expert_leaf_plain_on_vllm_without_lora_base_layer_prefix(monkeypatch):
    """Pre-#31104 vLLM: per-expert routed leaves carry NO ``base_layer`` segment.

    Unreachable on the installed vLLM, so pin it by flipping the flag --
    otherwise it is untested compat code.
    """
    model = _FakeModel(
        {
            "model.layers.0.mlp.experts.base_layer.routed_experts.w13_weight": torch.empty(0),
            "model.layers.0.mlp.experts.base_layer.routed_experts.w2_weight": torch.empty(0),
            "model.layers.0.mlp.experts.base_layer.gate.weight": torch.empty(0),
        }
    )
    worker = _make_worker(model)
    monkeypatch.setattr(_vllm_utils_real, "_HAS_LORA_BASE_LAYER_PREFIX", False)

    for sub in ("gate_proj", "up_proj", "down_proj"):
        plain = f"model.layers.0.mlp.experts.0.{sub}.weight"
        # Leaf-suffix form gets the injected segment removed ...
        assert _resolve(worker, model, f"model.layers.0.mlp.experts.0.{sub}.base_layer.weight") == plain
        # ... and the HF-native form passes through untouched.
        assert _resolve(worker, model, plain) == plain


# ---------------------------------------------------------------------------
# Packed/stacked names: the resolver toggles ``.base_layer`` on the UNMAPPED
# name and returns the unmapped form so vLLM's own ``load_weights`` does the
# stacking (returning the mapped name would double-map, e.g. ``in_proj_qkvz``
# -> ``in_proj_qkvzz`` with a wrong shard_id).
# ---------------------------------------------------------------------------


def test_unpacked_proj_exists_via_packed_owner():
    """An unpacked alias whose packed owner exists is returned unchanged."""
    model = _FakeModel(
        {"language_model.model.layers.0.self_attn.qkv_proj.weight": torch.empty(0)},
        packed={"qkv_proj": ["q_proj", "k_proj", "v_proj"]},
    )
    worker = _make_worker(model)

    incoming = "language_model.model.layers.0.self_attn.q_proj.weight"
    assert _resolve(worker, model, incoming) == incoming


def test_stacked_gdn_in_proj_not_double_mapped(monkeypatch):
    """GDN ``in_proj_qkv`` is NOT suffixed with ``.base_layer`` on newer vLLM --
    the name stays unmapped so vLLM's mapper stacks it cleanly."""
    monkeypatch.setattr(_vllm_utils_real, "_HAS_LORA_LOAD_WEIGHTS", True)
    mapper = _FakeMapper({".in_proj_qkv.": ".in_proj_qkvz."})
    model = _FakeModel(
        {"language_model.model.layers.0.linear_attn.in_proj_qkvz.base_layer.weight": torch.empty(0)},
        mapper=mapper,
    )
    worker = _make_worker(model)

    incoming = "language_model.model.layers.0.linear_attn.in_proj_qkv.weight"
    resolved = _resolve(worker, model, incoming)
    # No .base_layer added on newer vLLM.
    assert resolved == incoming
    # vLLM's mapper then maps it cleanly (single z):
    assert _mapped(_make_worker(model), model, resolved) == (
        "language_model.model.layers.0.linear_attn.in_proj_qkvz.weight"
    )


def test_shared_expert_packed_owner_no_base_layer_on_newer_vllm(monkeypatch):
    """On newer vLLM, shared expert gate_proj.weight is NOT suffixed with
    .base_layer -- vLLM's mapper stacks gate_proj -> gate_up_proj and
    BaseLayerWithLoRA.load_weights handles the inner name."""
    monkeypatch.setattr(_vllm_utils_real, "_HAS_LORA_LOAD_WEIGHTS", True)
    mapper = _FakeMapper({".shared_expert.gate_proj.": ".shared_expert.gate_up_proj."})
    model = _FakeModel(
        {"language_model.model.layers.0.mlp.shared_expert.gate_up_proj.base_layer.weight": torch.empty(0)},
        mapper=mapper,
        packed={"gate_up_proj": ["gate_proj", "up_proj"]},
    )
    worker = _make_worker(model)

    incoming = "language_model.model.layers.0.mlp.shared_expert.gate_proj.weight"
    resolved = _resolve(worker, model, incoming)
    assert resolved == incoming
    assert _mapped(_make_worker(model), model, resolved) == (
        "language_model.model.layers.0.mlp.shared_expert.gate_up_proj.weight"
    )


def test_mapper_alias_left_to_vllm_mapper():
    """A logical alias absent on the model is returned unchanged (not pre-mapped)."""
    model = _FakeModel(
        {"language_model.model.layers.0.mlp.experts.base_layer.w13_weight": torch.empty(0)},
        mapper=_FakeMapper(
            {
                "language_model.model.layers.0.mlp.experts.gate_up_proj": (
                    "language_model.model.layers.0.mlp.experts.base_layer.w13_weight"
                )
            }
        ),
    )
    worker = _make_worker(model)

    incoming = "language_model.model.layers.0.mlp.experts.gate_up_proj"
    assert _resolve(worker, model, incoming) == incoming


# ---------------------------------------------------------------------------
# Already-correct names are a no-op (the common case for non-VLM LoRA).
# ---------------------------------------------------------------------------


def test_already_matching_name_is_noop():
    model = _FakeModel({"language_model.model.layers.0.self_attn.q_proj.weight": torch.empty(0)})
    worker = _make_worker(model)

    assert _resolve(worker, model, "language_model.model.layers.0.self_attn.q_proj.weight") == (
        "language_model.model.layers.0.self_attn.q_proj.weight"
    )


def test_truly_unknown_name_strips_base_layer(monkeypatch):
    """Even unknown names have ``.base_layer.`` stripped on newer vLLM."""
    monkeypatch.setattr(_vllm_utils_real, "_HAS_LORA_LOAD_WEIGHTS", True)
    model = _FakeModel({"a.base_layer.weight": torch.empty(0)})
    worker = _make_worker(model)

    assert _resolve(worker, model, "does.not.exist.base_layer.weight") == "does.not.exist.weight"


def test_drafter_drops_base_layer_wholesale():
    """A model with no ``.base_layer`` params (e.g. the MTP drafter) has the
    suffix stripped wholesale from every name, regardless of whether the
    stripped form exists."""
    model = _FakeModel({"a.weight": torch.empty(0)})
    worker = _make_worker(model)

    assert _resolve(worker, model, "does.not.exist.base_layer.weight") == "does.not.exist.weight"
    assert _resolve(worker, model, "fc.base_layer.weight") == "fc.weight"


# ---------------------------------------------------------------------------
# End-to-end through _update_weights: load_weights receives resolved names.
# ---------------------------------------------------------------------------


def test_update_weights_resolves_names_before_load(monkeypatch):
    # LoRA base-sync phase (peft_config set, base_sync_done=False): the vLLM model
    # is LoRA-wrapped (has ``.base_layer``), so the resolver reconciles incoming
    # names against it. (``peft_config=None`` is the non-LoRA/merge path where the
    # model is never wrapped and the resolver is skipped — can't reach here.)
    monkeypatch.setattr(_vllm_utils_real, "_HAS_LORA_LOAD_WEIGHTS", True)
    model = _FakeModel(
        {
            "merger.linear_fc1.weight": torch.empty(0),
            "language_model.model.layers.0.self_attn.q_proj.base_layer.weight": torch.empty(0),
        }
    )
    worker = _make_worker(model)

    weights = [
        ("merger.linear_fc1.base_layer.weight", torch.empty(0)),
        ("language_model.model.layers.0.self_attn.q_proj.base_layer.weight", torch.empty(0)),
    ]

    worker._update_weights(weights, peft_config={"r": 1}, base_sync_done=False)

    assert model.loaded == [
        "merger.linear_fc1.weight",
        "language_model.model.layers.0.self_attn.q_proj.weight",
    ]


def test_base_sync_emits_leaf_keeps_lora_prefix(monkeypatch):
    """Base-sync emits the leaf form and does NOT mutate
    ``lora_base_layer_prefix`` -- the vLLM-side ``get_expert_mapping`` patch
    handles that; the receiver only emits the name."""
    monkeypatch.setattr(_vllm_utils_real, "_HAS_LORA_BASE_LAYER_PREFIX", True)
    routed = types.SimpleNamespace(lora_base_layer_prefix="base_layer.")
    model = _FakeModel(
        {"model.layers.0.mlp.experts.base_layer.routed_experts.w13_weight": torch.empty(0)},
        routed_experts=[routed],
    )
    worker = _make_worker(model)

    worker._update_weights(
        [("model.layers.0.mlp.experts.0.gate_proj.weight", torch.empty(0))],
        peft_config={"r": 1},
        base_sync_done=False,
    )

    assert model.loaded == ["model.layers.0.mlp.experts.0.gate_proj.base_layer.weight"]
    assert routed.lora_base_layer_prefix == "base_layer."


def test_lora_adapter_path_skips_normalization():
    """LoRA adapter tensors (peft_config + base_sync_done) keep .base_layer — vLLM expects it."""
    model = _FakeModel({"language_model.model.layers.0.self_attn.q_proj.base_layer.weight": torch.empty(0)})

    added = []

    def _add_lora(request):
        added.append(set(request.lora_tensors))

    worker = _make_worker(model)
    worker.add_lora = _add_lora

    # These are LoRA tensors (lora_A/lora_B), not base weights; the .base_layer here
    # is irrelevant to normalization and must be passed through verbatim.
    weights = [("language_model.model.layers.0.self_attn.q_proj.lora_A.weight", torch.ones(1))]
    worker._update_weights(weights, peft_config={"r": 1}, base_sync_done=True)

    assert added == [{"language_model.model.layers.0.self_attn.q_proj.lora_A.weight"}]


# ---------------------------------------------------------------------------
# Multi-bucket LoRA accumulation: add_lora is called exactly once, with the
# complete adapter, after is_last — never per-bucket.
# ---------------------------------------------------------------------------


class _FakeBucketReceiver:
    """Drives the on_bucket_received callback with a scripted bucket sequence."""

    def __init__(self, buckets):
        # buckets: list of (weights_list, is_last)
        self._buckets = buckets

    def receive_weights(self, on_bucket_received):
        for weights, is_last in self._buckets:
            on_bucket_received(weights, is_last)


def test_update_weights_from_ipc_accumulates_lora_across_buckets(monkeypatch):
    """A LoRA adapter split across two buckets yields one add_lora with all tensors."""
    # Unlike the resolver tests above (which fully stub sys.modules), this drives the
    # real update_weights_from_ipc path and imports the actual bucketed_weight_transfer
    # module, whose package __init__ hard-requires vllm. vllm isn't in the `cpu` extra,
    # so this is skipped under cpu_unit_tests and run in vllm.yml (the vllm venv).
    pytest.importorskip("vllm")
    import verl.workers.rollout.vllm_rollout.bucketed_weight_transfer as bwt

    monkeypatch.setattr(
        bwt,
        "BucketedWeightReceiver",
        lambda *a, **k: _FakeBucketReceiver(
            [
                ([("lora.A.weight", torch.ones(1))], False),
                ([("lora.B.weight", torch.zeros(1))], True),
            ]
        ),
    )

    model = _FakeModel({"q.base_layer.weight": torch.empty(0)})

    added = []

    def _add_lora(request):
        added.append(dict(request.lora_tensors))

    worker = _make_worker(model)
    worker.add_lora = _add_lora
    worker.remove_lora = lambda lora_id: None
    worker.device = torch.device("cpu")
    worker.local_rank = 0
    worker._is_qat_model = False
    worker._is_modelopt_qat = False
    worker._get_zmq_handle = lambda: "ipc:///tmp/test-bucketed-lora.sock"

    worker.update_weights_from_ipc(peft_config={"r": 1}, base_sync_done=True)

    # Exactly one add_lora, holding both buckets' tensors (accumulated, not
    # overwritten) with the first bucket's tensor preserved (cloned, not a view
    # into a reused buffer).
    assert len(added) == 1
    assert set(added[0]) == {"lora.A.weight", "lora.B.weight"}
    torch.testing.assert_close(added[0]["lora.A.weight"], torch.ones(1))


def test_update_weights_from_ipc_standard_loads_per_bucket(monkeypatch):
    """Standard (non-LoRA) base sync loads every bucket immediately (no accumulation)."""
    # See the note above: needs the real bucketed_weight_transfer (vllm-backed), which
    # the `cpu` extra can't provide, so it is skipped here and run in vllm.yml.
    pytest.importorskip("vllm")
    import verl.workers.rollout.vllm_rollout.bucketed_weight_transfer as bwt

    monkeypatch.setattr(
        bwt,
        "BucketedWeightReceiver",
        lambda *a, **k: _FakeBucketReceiver(
            [
                ([("q.weight", torch.ones(1))], False),
                ([("k.weight", torch.zeros(1))], True),
            ]
        ),
    )

    model = _FakeModel({"q.weight": torch.empty(0), "k.weight": torch.empty(0)})
    loaded = []
    model.load_weights = lambda weights: loaded.extend(name for name, _ in weights)

    worker = _make_worker(model)
    worker.device = torch.device("cpu")
    worker.local_rank = 0
    worker._is_qat_model = False
    worker._is_modelopt_qat = False
    worker._get_zmq_handle = lambda: "ipc:///tmp/test-bucketed-std.sock"

    # Bypass the post-load process_weights_after_loading (needs real vLLM).
    fake_loader_utils = types.ModuleType("vllm.model_executor.model_loader.utils")
    fake_loader_utils.process_weights_after_loading = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "vllm.model_executor.model_loader.utils", fake_loader_utils)

    worker.update_weights_from_ipc(peft_config=None, base_sync_done=False)

    assert loaded == ["q.weight", "k.weight"]


def test_update_weights_base_sync_strips_per_expert_leaves(monkeypatch):
    """merge=False base-sync emits the leaf form for flat-loader (Qwen3Moe)
    per-expert weights; strict-loader leaves are untouched."""
    # Needs the real bucketed_weight_transfer (vllm-backed) update_weights_from_ipc
    # path, which can't run in the cpu-only/no-vllm CI env.
    pytest.importorskip("vllm")
    import verl.workers.rollout.vllm_rollout.bucketed_weight_transfer as bwt

    monkeypatch.setattr(_vllm_utils_real, "_HAS_LORA_BASE_LAYER_PREFIX", True)

    # A flat (Qwen3Moe) live namespace: routed experts live under
    # ``experts.base_layer.routed_experts.{w13,w2}_weight`` and the LoRA-leaf
    # ``experts.<id>.<proj>.base_layer.weight`` segment IS present (so
    # resolve_weight_name's ``any("mlp.experts.base_layer." in n ...)`` triggers
    # and it routes the incoming leaf under base_layer).
    live = {
        "model.layers.0.mlp.experts.base_layer.routed_experts.w13_weight": torch.empty(0),
        "model.layers.0.mlp.experts.base_layer.routed_experts.w2_weight": torch.empty(0),
        "model.layers.0.mlp.experts.base_layer.gate.weight": torch.empty(0),
    }
    model = _FakeModel(live)
    loaded = []
    model.load_weights = lambda weights: loaded.extend(name for name, _ in weights)

    monkeypatch.setattr(
        bwt,
        "BucketedWeightReceiver",
        lambda *a, **k: _FakeBucketReceiver(
            [
                (
                    [
                        ("model.layers.0.mlp.experts.0.gate_proj.base_layer.weight", torch.ones(1)),
                        ("model.layers.0.mlp.experts.0.up_proj.base_layer.weight", torch.ones(1)),
                        ("model.layers.0.mlp.experts.0.down_proj.base_layer.weight", torch.ones(1)),
                        ("model.layers.0.mlp.experts.1.gate_proj.base_layer.weight", torch.ones(1)),
                    ],
                    True,
                )
            ]
        ),
    )

    worker = _make_worker(model)
    worker.device = torch.device("cpu")
    worker.local_rank = 0
    worker._is_qat_model = False
    worker._is_modelopt_qat = False
    worker._get_zmq_handle = lambda: "ipc:///tmp/test-bucketed-mf.sock"

    fake_loader_utils = types.ModuleType("vllm.model_executor.model_loader.utils")
    fake_loader_utils.process_weights_after_loading = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "vllm.model_executor.model_loader.utils", fake_loader_utils)

    worker.update_weights_from_ipc(peft_config={"r": 1}, base_sync_done=False)

    assert loaded == [
        "model.layers.0.mlp.experts.0.gate_proj.base_layer.weight",
        "model.layers.0.mlp.experts.0.up_proj.base_layer.weight",
        "model.layers.0.mlp.experts.0.down_proj.base_layer.weight",
        "model.layers.0.mlp.experts.1.gate_proj.base_layer.weight",
    ]


# ---------------------------------------------------------------------------
# Loader-style-aware strip: strict inner loaders (DeepseekV2 ``params_dict[name]``
# lookup) need ``.base_layer.`` KEPT; flat loaders (Llama/Qwen3.5
# ``AutoWeightsLoader`` recursion) need it STRIPPED. Probed at runtime from the
# inner ``load_weights`` source, not a model_type allowlist.
# ---------------------------------------------------------------------------


class _FakeStrictInner:
    """Stand-in for a strict inner transformer (DeepseekV2Model-style).

    Its ``load_weights`` source contains ``params_dict[`` and NOT
    ``AutoWeightsLoader``, so the probe returns True. At module scope so
    ``inspect.getsource`` can read it.
    """

    def load_weights(self, weights):
        loaded = set()
        # Strict loaders (DeepseekV2) do ``params_dict[name]`` lookups; the
        # probe matches the ``params_dict[`` token in this source. The line is
        # never executed (weights is empty in tests), it just needs to be
        # visible to ``inspect.getsource``.
        params_dict = {"": None}  # noqa: F841
        for name, _ in weights:
            _probe = params_dict[name]  # noqa: F841
            loaded.add(name)
        return loaded


class _FakeFlatInner:
    """Stand-in for a flat inner transformer (LlamaModel/Qwen3_5Model-style).

    Its ``load_weights`` source contains ``AutoWeightsLoader``, so the probe
    returns False (flat -> strip).
    """

    def load_weights(self, weights):
        loader = AutoWeightsLoader(self)  # noqa: F821 - source signature only
        return loader.load_weights(weights)


class _OuterModel:
    """Wraps an inner transformer under ``.model`` (the ForCausalLM layout)."""

    def __init__(self, inner, params):
        self.model = inner
        self._params = dict(params)

    def named_parameters(self, remove_duplicate: bool = False):
        del remove_duplicate
        yield from self._params.items()

    def named_buffers(self):
        return iter(())


def _strict_outer(params):
    _vllm_utils_real._inner_load_weights_is_strict_cache.clear()
    return _OuterModel(_FakeStrictInner(), params)


def _flat_outer(params):
    _vllm_utils_real._inner_load_weights_is_strict_cache.clear()
    return _OuterModel(_FakeFlatInner(), params)


def test_strict_loader_keeps_base_layer_suffix_on_newer_vllm(monkeypatch):
    """A *strict* loader (DeepseekV2 params_dict[name] lookup) receives the
    SUFFIXED name unchanged — the unconditional strip flat loaders need would
    break strict loaders (KeyError on the stripped name)."""
    monkeypatch.setattr(_vllm_utils_real, "_HAS_LORA_LOAD_WEIGHTS", True)
    key = "language_model.model.layers.0.self_attn.q_proj.base_layer.weight"
    model = _strict_outer({key: torch.empty(0)})
    worker = _make_worker(model)
    assert _resolve(worker, model, key) == key  # KEPT, not stripped


def test_strict_loader_re_suffixes_plain_name_on_newer_vllm(monkeypatch):
    """A *strict* loader: an incoming PLAIN name (no .base_layer.) that doesn't
    match the live suffixed key is re-suffixed so params_dict[name] finds it
    (mirrors released-vLLM behavior for strict loaders)."""
    monkeypatch.setattr(_vllm_utils_real, "_HAS_LORA_LOAD_WEIGHTS", True)
    live = "language_model.model.layers.0.self_attn.q_proj.base_layer.weight"
    model = _strict_outer({live: torch.empty(0)})
    worker = _make_worker(model)
    plain = "language_model.model.layers.0.self_attn.q_proj.weight"
    assert _resolve(worker, model, plain) == live  # re-suffixed


def test_flat_loader_strips_base_layer_suffix_on_newer_vllm(monkeypatch):
    """A *flat* loader (AutoWeightsLoader recursion into base_layer): the
    SUFFIXED name is stripped — loader-style-gated, not a blanket newer-vLLM strip."""
    monkeypatch.setattr(_vllm_utils_real, "_HAS_LORA_LOAD_WEIGHTS", True)
    key = "language_model.model.layers.0.self_attn.q_proj.base_layer.weight"
    model = _flat_outer({key: torch.empty(0)})
    worker = _make_worker(model)
    assert _resolve(worker, model, key) == ("language_model.model.layers.0.self_attn.q_proj.weight")  # STRIPPED


def test_strict_loader_per_expert_leaf_kept_verbatim(monkeypatch):
    """A *strict* loader (DeepseekV2/Moonlight) per-expert leaf-suffix name is
    KEPT verbatim (``experts.<id>.<proj>.base_layer.<leaf>``) — strict loaders do
    ``params_dict[name]`` with a suffixed key, so routing it under base_layer
    (the flat-loader fix) would break them. Guards the 6/8 passing combos."""
    monkeypatch.setattr(_vllm_utils_real, "_HAS_LORA_LOAD_WEIGHTS", False)
    live = "model.layers.0.mlp.experts.0.gate_proj.base_layer.weight"
    model = _strict_outer(
        {
            "model.layers.0.mlp.experts.base_layer.routed_experts.w13_weight": torch.empty(0),
            live: torch.empty(0),
        }
    )
    worker = _make_worker(model)
    assert _resolve(worker, model, live) == live  # KEPT verbatim, not routed


def test_strict_loader_per_expert_leaf_plain_re_suffixed(monkeypatch):
    """A *strict* loader: a PLAIN per-expert leaf (no ``.base_layer.``) matching a
    live suffixed key is re-suffixed so ``params_dict[name]`` finds it."""
    monkeypatch.setattr(_vllm_utils_real, "_HAS_LORA_LOAD_WEIGHTS", False)
    live = "model.layers.0.mlp.experts.0.gate_proj.base_layer.weight"
    model = _strict_outer(
        {
            "model.layers.0.mlp.experts.base_layer.routed_experts.w13_weight": torch.empty(0),
            live: torch.empty(0),
        }
    )
    worker = _make_worker(model)
    plain = "model.layers.0.mlp.experts.0.gate_proj.weight"
    assert _resolve(worker, model, plain) == live  # re-suffixed, not routed


def test_strict_probe_is_cached_per_class():
    """The loader-style probe runs once per inner-model class (getsource is
    slow); a second lookup is a cache hit."""
    _vllm_utils_real._inner_load_weights_is_strict_cache.clear()
    model = _strict_outer({"q.base_layer.weight": torch.empty(0)})
    probe = _vllm_utils_real._inner_load_weights_is_strict
    # First call introspects and caches.
    first = probe(model)
    assert first is True
    # The cache is populated keyed by the outer-model class.
    assert type(model) in _vllm_utils_real._inner_load_weights_is_strict_cache
    # Second call is a cache hit (no re-introspection).
    second = probe(model)
    assert second is True


def test_strict_loader_vision_merger_still_strips(monkeypatch):
    """Even under a strict inner loader, a non-LoRA-wrapped module (e.g. the
    Qwen3.5-VL vision merger) whose ``.base_layer.`` isn't a real child still
    gets stripped — the per-name ``parent_has_base_layer`` check is unchanged."""
    monkeypatch.setattr(_vllm_utils_real, "_HAS_LORA_LOAD_WEIGHTS", True)
    model = _strict_outer(
        {
            # Language leaf has a real base_layer child; merger does not.
            "language_model.model.layers.0.self_attn.q_proj.base_layer.weight": torch.empty(0),
            "model.visual.merger.linear_fc1.weight": torch.empty(0),
        }
    )
    worker = _make_worker(model)
    # Merger .base_layer. has no live child -> stripped (per-name, loader-independent).
    assert _resolve(worker, model, "model.visual.merger.linear_fc1.base_layer.weight") == (
        "model.visual.merger.linear_fc1.weight"
    )
    # Language leaf keeps the suffix (strict loader).
    assert (
        _resolve(worker, model, "language_model.model.layers.0.self_attn.q_proj.base_layer.weight")
        == "language_model.model.layers.0.self_attn.q_proj.base_layer.weight"
    )


def test_strict_loader_non_lora_param_on_wrapped_module_re_suffixed(monkeypatch):
    """A non-LoRA parameter living on a LoRA-wrapped module (e.g. DSV4 router's
    ``gate.tid2eid`` / ``gate.e_score_correction_bias``) moves under
    ``.base_layer.`` when the module is wrapped. The resolver re-adds the
    suffix for both leaf and non-leaf params (the inner ``load_weights`` no
    longer re-adds, so the resolver owns re-add for everything whose live key
    is suffixed), gated by ``_exists(alt)`` so a plain name on an unwrapped
    module is left alone."""
    monkeypatch.setattr(_vllm_utils_real, "_HAS_LORA_LOAD_WEIGHTS", False)
    live_tid = "model.layers.0.mlp.gate.base_layer.tid2eid"
    live_bias = "model.layers.0.mlp.gate.base_layer.e_score_correction_bias"
    model = _strict_outer({live_tid: torch.empty(0), live_bias: torch.empty(0)})
    worker = _make_worker(model)
    # tid2eid: non-leaf, but live key is suffixed -> re-suffixed by the resolver.
    assert _resolve(worker, model, "model.layers.0.mlp.gate.tid2eid") == live_tid
    # e_score_correction_bias: ends in _bias, live key suffixed -> re-suffixed.
    assert _resolve(worker, model, "model.layers.0.mlp.gate.e_score_correction_bias") == live_bias


def test_flat_loader_non_lora_param_on_wrapped_module_not_suffixed(monkeypatch):
    """A flat loader (AutoWeightsLoader recursion) reaches the param through the
    ``base_layer`` child, so the unwrapped name is left alone -- no suffix
    re-added (mirrors the leaf branch's loader-style gate)."""
    monkeypatch.setattr(_vllm_utils_real, "_HAS_LORA_LOAD_WEIGHTS", True)
    live = "model.layers.0.mlp.gate.base_layer.tid2eid"
    model = _flat_outer({live: torch.empty(0)})
    worker = _make_worker(model)
    assert _resolve(worker, model, "model.layers.0.mlp.gate.tid2eid") == ("model.layers.0.mlp.gate.tid2eid")


def test_strict_loader_multi_segment_shard_strips_to_unfused_leaf(monkeypatch):
    """DSV4 ``compressor.fused_wkv_wgate`` is an *unwrappable* MergedColumnParallelLinear
    (no live ``.base_layer.``). Under merge=False LoRA, if the sender exports the
    constituent shard names (``compressor.wkv``/``compressor.wgate``) with the
    LoRA ``.base_layer.`` suffix, the ``shard_to_owner`` strip branch rewrites
    in-place on the full incoming name (prefix-safe) and returns the shard leaf
    (``compressor.wkv.weight``); vLLM's ``stacked_params_mapping`` then fuses
    it to the live ``compressor.fused_wkv_wgate.weight``. Without this branch
    the general ``_exists(stripped)`` path drops the ``model.layers.N.`` prefix
    on multi-segment shards and misses, so the suffixed name leaks through and
    the strict loader KeyErrors."""
    monkeypatch.setattr(_vllm_utils_real, "_HAS_LORA_LOAD_WEIGHTS", True)
    live = "model.layers.0.attn.compressor.fused_wkv_wgate.weight"
    model = _strict_outer({live: torch.empty(0)})
    model.packed_modules_mapping = {
        "compressor.fused_wkv_wgate": ["compressor.wkv", "compressor.wgate"],
    }
    worker = _make_worker(model)
    assert (
        _resolve(worker, model, "model.layers.0.attn.compressor.wkv.base_layer.weight")
        == "model.layers.0.attn.compressor.wkv.weight"
    )
    assert (
        _resolve(worker, model, "model.layers.0.attn.compressor.wgate.base_layer.weight")
        == "model.layers.0.attn.compressor.wgate.weight"
    )


def test_strict_loader_unprefixed_shard_against_model_prefixed_live(monkeypatch):
    """DSV4 crash repro: a shard-form base-sync name in HF checkpoint style
    WITHOUT a ``model.`` prefix (``layers.0.attn.compressor.wgate.base_layer.weight``),
    while vLLM's ``ForCausalLM.named_parameters`` reports the live param WITH the
    ``model.`` prefix (``model.layers.0.attn.compressor.fused_wkv_wgate.weight``).
    The ``shard_to_owner`` strip branch matches the shard substring on the full
    incoming name (prefix-agnostic) and strips the suffix, so the unprefixed
    shard leaf (``layers.0.attn.compressor.wgate.weight``) is returned for
    vLLM's ``stacked_params_mapping`` to fuse (``compressor.wgate`` ->
    ``compressor.fused_wkv_wgate``).

    The live fused target is unwrappable (no ``.base_layer.`` child), so the
    resolver strips the LoRA suffix and keeps the unprefixed shard leaf."""
    monkeypatch.setattr(_vllm_utils_real, "_HAS_LORA_LOAD_WEIGHTS", True)
    live = "model.layers.0.attn.compressor.fused_wkv_wgate.weight"
    model = _strict_outer({live: torch.empty(0)})
    # DSV4 ``hf_to_vllm_mapper``: ``layers.`` -> ``model.layers.``.
    model.hf_to_vllm_mapper = _FakeMapper({"layers.": "model.layers."})
    model.packed_modules_mapping = {
        "compressor.fused_wkv_wgate": ["compressor.wkv", "compressor.wgate"],
    }
    worker = _make_worker(model)
    assert (
        _resolve(worker, model, "layers.0.attn.compressor.wkv.base_layer.weight")
        == "layers.0.attn.compressor.wkv.weight"
    )
    assert (
        _resolve(worker, model, "layers.0.attn.compressor.wgate.base_layer.weight")
        == "layers.0.attn.compressor.wgate.weight"
    )


def test_packed_shard_keeps_suffix_when_fused_target_is_wrapped(monkeypatch):
    """A packed shard whose fused owner is LoRA-wrapped must arrive SUFFIXED.

    vLLM rewrites shard -> fused owner and then indexes ``params_dict``, which
    holds ``gate_up_proj.base_layer.weight``; stripping is a KeyError. Loaders
    that re-add the suffix themselves (DSV4) guard on ``not in params_dict``, so
    the suffixed form is right for them too.
    """
    monkeypatch.setattr(_vllm_utils_real, "_HAS_LORA_LOAD_WEIGHTS", True)
    live = "model.layers.0.mlp.gate_up_proj.base_layer.weight"
    model = _strict_outer({live: torch.empty(0)})
    model.packed_modules_mapping = {"gate_up_proj": ["gate_proj", "up_proj"]}
    worker = _make_worker(model)
    for shard in ("gate_proj", "up_proj"):
        name = f"model.layers.0.mlp.{shard}.base_layer.weight"
        assert _resolve(worker, model, name) == name


def test_packed_shard_keeps_suffix_on_released_vllm(monkeypatch):
    """vLLM 0.24 (no ``BaseLayerWithLoRA.load_weights``): a LoRA-wrapped packed
    shard must keep the suffix so ``AutoWeightsLoader`` recurses into the
    ``base_layer`` child after the shard -> fused rewrite.

    Regression for CI ``e2e_ppo_trainer_megatron-deepseek`` (Llama-arch
    deepseek-coder-1.3b, ``target_modules`` includes ``linear_fc1``), which
    died with ``KeyError: 'layers.0.mlp.gate_up_proj.weight'``.
    """
    monkeypatch.setattr(_vllm_utils_real, "_HAS_LORA_LOAD_WEIGHTS", False)
    live = {
        "model.layers.0.mlp.gate_up_proj.base_layer.weight": torch.empty(0),
        "model.layers.0.self_attn.qkv_proj.base_layer.weight": torch.empty(0),
    }
    for outer in (_strict_outer, _flat_outer):
        model = outer(live)
        model.packed_modules_mapping = {
            "qkv_proj": ["q_proj", "k_proj", "v_proj"],
            "gate_up_proj": ["gate_proj", "up_proj"],
        }
        worker = _make_worker(model)
        for shard in ("mlp.gate_proj", "mlp.up_proj", "self_attn.q_proj"):
            name = f"model.layers.0.{shard}.base_layer.weight"
            assert _resolve(worker, model, name) == name
