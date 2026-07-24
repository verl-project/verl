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

"""CPU regression tests for the vLLM >= 0.20 FP8 layerwise-reload weight resync.

Covers three defect classes:

1. Buffer ownership: verl's bucketed IPC receiver reuses ONE backing buffer
   for every bucket, while vLLM's layerwise reload buffers streamed tensors
   until an entire layer has arrived (potentially across buckets). A tensor
   yielded as a view into the shared buffer and retained past a bucket
   boundary is silently overwritten by the next bucket. ``quant_weights``
   must clone non-quantized tensors while a layerwise reload is active.

2. Lifecycle: ``begin_fp8_layerwise_reload`` / ``finalize_fp8_layerwise_reload``
   must be called exactly once per sync (double-begin and finalize-without-
   begin raise), and ``load_quanted_weights`` fails closed when a bucket
   arrives on a reload-capable vLLM without an active reload.

3. Version gate: the resync path is validated on vLLM 0.20.x only. Newer
   lines must NOT be silently opted in — begin is a no-op and the per-bucket
   loader raises an explicit version error.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_vllm_fp8_utils():
    """Load verl/utils/vllm/vllm_fp8_utils.py with vLLM and heavy deps stubbed.

    Injected ``sys.modules`` entries are restored afterwards so the fakes do
    not leak into other tests; the loaded module keeps working since it binds
    the names it needs at import time.
    """
    module_name = "verl.utils.vllm.vllm_fp8_utils_under_test"
    module_path = _REPO_ROOT / "verl/utils/vllm/vllm_fp8_utils.py"

    class _FakeFusedMoE(torch.nn.Module):
        pass

    class _FakeLinearBase(torch.nn.Module):
        pass

    fake_fused_moe_layer = types.ModuleType("vllm.model_executor.layers.fused_moe.layer")
    fake_fused_moe_layer.FusedMoE = _FakeFusedMoE
    fake_linear = types.ModuleType("vllm.model_executor.layers.linear")
    fake_linear.LinearBase = _FakeLinearBase

    fake_fp8_mod = types.ModuleType("vllm.model_executor.layers.quantization.fp8")

    class _FakeFp8Config:
        pass

    fake_fp8_mod.Fp8Config = _FakeFp8Config
    fake_fp8_mod.replace_parameter = lambda *a, **k: None

    # The layerwise reload entry points, importable so the availability probe
    # passes; behavior is irrelevant for these tests (never actually driven).
    fake_reload = types.ModuleType("vllm.model_executor.model_loader.reload")
    fake_reload.initialize_layerwise_reload = lambda model: None
    fake_reload.finalize_layerwise_reload = lambda model, cfg: None

    fake_config = types.ModuleType("vllm.config")

    class _FakeSetCurrentVllmConfig:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return None

        def __exit__(self, *a):
            return False

    fake_config.set_current_vllm_config = _FakeSetCurrentVllmConfig

    fake_vllm = types.ModuleType("vllm")
    fake_vllm.__version__ = "0.20.2"

    fake_kernel = types.ModuleType("verl.utils.kernel.fp8_kernel")

    def _fake_scaled_fp8_blockwise(t, weight_block_size=None):
        q = t.to(torch.float32)
        scale = torch.ones(1, 1, 1)
        return q, scale

    fake_kernel.scaled_fp8_blockwise = _fake_scaled_fp8_blockwise

    fake_dsv4 = types.ModuleType("verl.utils.vllm.vllm_dsv4_fp8_utils")
    fake_dsv4.cache_deepseek_v4_dense_fp8_scales = lambda model, weights: None
    fake_dsv4.is_deepseek_v4_model = lambda model: False
    fake_dsv4.iter_deepseek_v4_weights = lambda weights: iter(weights)
    fake_dsv4.prepare_deepseek_v4_weights_for_loading = lambda model, copy_fn: False
    fake_dsv4.process_deepseek_v4_weights_after_loading = lambda model, state: None
    fake_dsv4.reload_deepseek_v4_dense_fp8_scales = lambda model: None

    fakes = {
        "vllm": fake_vllm,
        "vllm.config": fake_config,
        "vllm.model_executor": types.ModuleType("vllm.model_executor"),
        "vllm.model_executor.layers": types.ModuleType("vllm.model_executor.layers"),
        "vllm.model_executor.layers.fused_moe": types.ModuleType("vllm.model_executor.layers.fused_moe"),
        "vllm.model_executor.layers.fused_moe.layer": fake_fused_moe_layer,
        "vllm.model_executor.layers.linear": fake_linear,
        "vllm.model_executor.layers.quantization": types.ModuleType("vllm.model_executor.layers.quantization"),
        "vllm.model_executor.layers.quantization.fp8": fake_fp8_mod,
        "vllm.model_executor.model_loader": types.ModuleType("vllm.model_executor.model_loader"),
        "vllm.model_executor.model_loader.reload": fake_reload,
        "verl.utils.kernel.fp8_kernel": fake_kernel,
        "verl.utils.vllm.vllm_dsv4_fp8_utils": fake_dsv4,
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


fp8_utils = _load_vllm_fp8_utils()


def _make_fake_reload_module():
    fake_reload = types.ModuleType("vllm.model_executor.model_loader.reload")
    fake_reload.initialize_layerwise_reload = lambda model: None
    fake_reload.finalize_layerwise_reload = lambda model, cfg: None
    return fake_reload


def _make_fake_config_module():
    fake_config = types.ModuleType("vllm.config")

    class _FakeSetCurrentVllmConfig:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return None

        def __exit__(self, *a):
            return False

    fake_config.set_current_vllm_config = _FakeSetCurrentVllmConfig
    return fake_config


@pytest.fixture(autouse=True)
def _fake_lazy_vllm_imports(monkeypatch):
    """The module under test lazily imports the vLLM reload entry points at
    call time; pin fakes into sys.modules so the tests do not depend on the
    host's installed vLLM shipping (or not shipping) the reload protocol."""
    monkeypatch.setitem(sys.modules, "vllm.model_executor.model_loader.reload", _make_fake_reload_module())
    monkeypatch.setitem(sys.modules, "vllm.config", _make_fake_config_module())
    yield


class _ToyModel(torch.nn.Module):
    """No FP8 modules: every streamed tensor takes the non-quantized path."""

    packed_modules_mapping: dict = {}

    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.norm = torch.nn.LayerNorm(4)


class _FakeQuantConfig:
    weight_block_size = (128, 128)


@pytest.fixture(autouse=True)
def _reset_fp8_state(monkeypatch):
    fp8_utils.fp8_state.seen_params.clear()
    fp8_utils.fp8_state.fp8_param_names.clear()
    fp8_utils.fp8_state.layerwise_active.clear()
    yield
    fp8_utils.fp8_state.layerwise_active.clear()


def _pin_version(monkeypatch, ver: str):
    from packaging import version as _v

    monkeypatch.setattr(fp8_utils, "_get_vllm_version", lambda: _v.parse(ver))


def _bucket_views_from_buffer(buffer: torch.Tensor, specs):
    """Build (name, tensor) views into a shared uint8 buffer, mirroring
    BucketedWeightReceiver.receive_weights."""
    weights = []
    for name, shape, offset in specs:
        size = 4 * int(torch.tensor(shape).prod())  # float32 elements
        t = buffer[offset : offset + size].view(dtype=torch.float32).view(shape)
        weights.append((name, t))
    return weights


def test_layerwise_reload_clones_bucket_backed_tensors(monkeypatch):
    """Two buckets share one reused backing buffer; the layer only completes
    after bucket 2 has overwritten the storage. The bytes handed to vLLM for
    bucket-1 tensors must come from the ORIGINAL bucket-1 payload."""
    _pin_version(monkeypatch, "0.20.2")
    model = _ToyModel()
    buffer = torch.zeros(256, dtype=torch.uint8)

    # --- bucket 1 arrives: a non-FP8 tensor viewed into the shared buffer ---
    (name, view1) = _bucket_views_from_buffer(buffer, [("model.norm.weight", (4,), 0)])[0]
    view1.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    original = view1.clone()

    fp8_utils.fp8_state.layerwise_active.add("main")
    buffered = list(fp8_utils.quant_weights([(name, view1)], model, _FakeQuantConfig()))
    assert len(buffered) == 1
    held_name, held = buffered[0]
    assert held_name == name

    # The layerwise reload holds `held` past the bucket boundary (delayed
    # layer completion). Bucket 2 now reuses the same backing buffer:
    buffer.fill_(0xFF)

    # Layer "completes" now — the held tensor must still carry bucket-1 bytes.
    assert torch.equal(held, original), (
        "bucket-backed tensor was overwritten by bucket 2 before the layer "
        "completed — quant_weights must clone non-quantized tensors while a "
        "layerwise reload is active"
    )
    assert held.data_ptr() != view1.data_ptr()


def test_without_active_reload_views_alias_the_buffer(monkeypatch):
    """Negative control: outside a layerwise reload the non-quantized path
    yields the view unchanged (zero-copy fast path preserved)."""
    _pin_version(monkeypatch, "0.20.2")
    model = _ToyModel()
    buffer = torch.zeros(256, dtype=torch.uint8)
    (name, view1) = _bucket_views_from_buffer(buffer, [("model.norm.weight", (4,), 0)])[0]
    view1.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))

    assert not fp8_utils.fp8_state.layerwise_active
    buffered = list(fp8_utils.quant_weights([(name, view1)], model, _FakeQuantConfig()))
    assert buffered[0][1].data_ptr() == view1.data_ptr()


def test_begin_finalize_lifecycle_guards(monkeypatch):
    _pin_version(monkeypatch, "0.20.2")
    model = _ToyModel()

    assert fp8_utils.begin_fp8_layerwise_reload(model, tag="main") is True
    with pytest.raises(RuntimeError, match="already active"):
        fp8_utils.begin_fp8_layerwise_reload(model, tag="main")

    fp8_utils.finalize_fp8_layerwise_reload(model, model_config=None, tag="main")
    with pytest.raises(RuntimeError, match="without an active begin"):
        fp8_utils.finalize_fp8_layerwise_reload(model, model_config=None, tag="main")


def test_begin_is_noop_below_020(monkeypatch):
    _pin_version(monkeypatch, "0.19.0")
    assert fp8_utils.begin_fp8_layerwise_reload(_ToyModel(), tag="main") is False
    assert not fp8_utils.fp8_state.layerwise_active


def test_abort_clears_state_and_allows_rebegin(monkeypatch):
    """After a failed sync, abort_fp8_layerwise_reload clears the active flag so
    the next begin succeeds instead of raising the stale 'already active'
    lifecycle error (the caller separately fail-stops the worker)."""
    _pin_version(monkeypatch, "0.20.2")
    model = _ToyModel()

    assert fp8_utils.begin_fp8_layerwise_reload(model, tag="main") is True
    assert "main" in fp8_utils.fp8_state.layerwise_active

    fp8_utils.abort_fp8_layerwise_reload("main")
    assert "main" not in fp8_utils.fp8_state.layerwise_active

    # A fresh sync can begin again — abort did not leave the tag poisoned.
    assert fp8_utils.begin_fp8_layerwise_reload(model, tag="main") is True


class _FakeModelConfig:
    dtype = torch.bfloat16


class _FakeVllmConfig:
    quant_config = _FakeQuantConfig()
    model_config = _FakeModelConfig()


class _FakeModelRunner:
    def __init__(self, model):
        self.model = model
        self.vllm_config = _FakeVllmConfig()


def test_load_quanted_weights_fails_closed_without_begin(monkeypatch):
    """A bucket on reload-capable vLLM without an active reload must raise,
    not stream checkpoint-format weights into kernel-format params."""
    _pin_version(monkeypatch, "0.20.2")
    runner = _FakeModelRunner(_ToyModel())
    with pytest.raises(RuntimeError, match="without an active layerwise reload"):
        fp8_utils.load_quanted_weights([("model.norm.weight", torch.zeros(4))], runner)


def test_load_quanted_weights_fails_closed_on_unvalidated_vllm(monkeypatch):
    """vLLM >= 0.21 ships the reload module but is not validated: begin must
    be a no-op and the per-bucket loader must raise an explicit version
    error instead of silently opting the new line in."""
    _pin_version(monkeypatch, "0.21.0")
    assert fp8_utils._vllm_layerwise_reload_available() is True
    assert fp8_utils._vllm_supports_layerwise_reload() is False
    assert fp8_utils.begin_fp8_layerwise_reload(_ToyModel(), tag="main") is False

    runner = _FakeModelRunner(_ToyModel())
    with pytest.raises(RuntimeError, match="validated on vLLM 0.20.x only"):
        fp8_utils.load_quanted_weights([("model.norm.weight", torch.zeros(4))], runner)
