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
    def _clear():
        fp8_utils.fp8_state.seen_params.clear()
        fp8_utils.fp8_state.fp8_param_names.clear()
        fp8_utils.fp8_state.layerwise_active.clear()
        fp8_utils.fp8_state.layerwise_begin_attempted.clear()
        fp8_utils.fp8_state.layerwise_poisoned.clear()

    _clear()
    yield
    _clear()


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


class _FakeModelConfig:
    dtype = torch.bfloat16


class _FakeVllmConfig:
    quant_config = _FakeQuantConfig()
    model_config = _FakeModelConfig()


class _FakeModelRunner:
    """Stand-in for vLLM's ``GPUModelRunner`` with the two serving entry points.

    ``execute_model`` / ``_dummy_run`` are the methods
    ``vllm/v1/worker/gpu_worker.py`` ``Worker.execute_model`` /
    ``Worker.execute_dummy_batch`` delegate to at every audited tag, so they are
    what the poison guard must shadow.
    """

    def __init__(self, model):
        self.model = model
        self.vllm_config = _FakeVllmConfig()
        self.served = 0

    def execute_model(self, scheduler_output=None, intermediate_tensors=None):
        self.served += 1
        return "output"

    def _dummy_run(self, num_tokens=1, **kwargs):
        self.served += 1
        return "output"


def test_load_quanted_weights_fails_closed_without_begin(monkeypatch):
    """A bucket on reload-capable vLLM without an active reload must raise,
    not stream checkpoint-format weights into kernel-format params."""
    _pin_version(monkeypatch, "0.20.2")
    runner = _FakeModelRunner(_ToyModel())
    with pytest.raises(RuntimeError, match="without an active layerwise reload"):
        fp8_utils.load_quanted_weights([("model.norm.weight", torch.zeros(4))], runner)


def test_validated_interval_covers_the_audited_vllm_tags(monkeypatch):
    """The gate must cover every vLLM line whose reload semantics were audited.

    verl's CI images pin vllm023.dev1 and the dense/MoE FP8 E2Es exercise this
    path, so a gate that excludes 0.23 turns fail-closed into a red CI. The
    upper bound is the first line whose layer-completion accounting changed
    (0.25.0, which re-derives load_numel_total on every load).
    """
    for validated in ("0.20.0", "0.20.2", "0.21.0", "0.22.1", "0.23.0", "0.24.0"):
        _pin_version(monkeypatch, validated)
        assert fp8_utils._vllm_supports_layerwise_reload() is True, validated

    for unvalidated in ("0.25.0", "0.26.0"):
        _pin_version(monkeypatch, unvalidated)
        assert fp8_utils._vllm_layerwise_reload_available() is True, unvalidated
        assert fp8_utils._vllm_supports_layerwise_reload() is False, unvalidated


def test_load_quanted_weights_fails_closed_on_unvalidated_vllm(monkeypatch):
    """Above the validated interval the reload module still imports, but the
    lifecycle semantics are unverified: begin must be a no-op and the
    per-bucket loader must raise an explicit version error instead of silently
    opting the new line in."""
    _pin_version(monkeypatch, "0.25.0")
    assert fp8_utils._vllm_layerwise_reload_available() is True
    assert fp8_utils._vllm_supports_layerwise_reload() is False
    assert fp8_utils.begin_fp8_layerwise_reload(_ToyModel(), tag="main") is False

    runner = _FakeModelRunner(_ToyModel())
    with pytest.raises(RuntimeError, match=r"validated on vLLM >= 0\.20, < 0\.25\.0"):
        fp8_utils.load_quanted_weights([("model.norm.weight", torch.zeros(4))], runner)


# ---------------------------------------------------------------------------
# Fault injection: a begin that dies part-way through initialize_layerwise_reload
# leaves the model half-converted (early layers on meta, the rest real), so the
# worker must be fail-stopped rather than reused.
# ---------------------------------------------------------------------------


def _pin_failing_initialize(monkeypatch, layers_converted: int):
    """Make initialize_layerwise_reload convert N layers, then raise."""
    fake_reload = types.ModuleType("vllm.model_executor.model_loader.reload")
    converted = []

    def _initialize_layerwise_reload(model):
        for index, module in enumerate(model.modules()):
            if index >= layers_converted:
                raise RuntimeError("synthetic vLLM failure mid-initialize (layer swap to meta)")
            converted.append(module)

    fake_reload.initialize_layerwise_reload = _initialize_layerwise_reload
    fake_reload.finalize_layerwise_reload = lambda model, cfg: None
    monkeypatch.setitem(sys.modules, "vllm.model_executor.model_loader.reload", fake_reload)
    return converted


def test_begin_records_attempt_before_calling_into_vllm(monkeypatch):
    """`attempted` must be recorded BEFORE the vLLM call, otherwise a failure
    part-way through initialize leaves no durable evidence that the model was
    touched (`layerwise_active` is never reached)."""
    _pin_version(monkeypatch, "0.23.0")
    converted = _pin_failing_initialize(monkeypatch, layers_converted=1)
    model = _ToyModel()

    with pytest.raises(RuntimeError, match="synthetic vLLM failure mid-initialize"):
        fp8_utils.begin_fp8_layerwise_reload(model, tag="main")

    assert converted, "the fault injection must convert at least one layer before raising"
    assert "main" in fp8_utils.fp8_state.layerwise_begin_attempted
    # NOT active: the reload never completed.
    assert "main" not in fp8_utils.fp8_state.layerwise_active


def test_begin_failure_poisons_the_worker(monkeypatch):
    """A begin that raises mid-initialize must fail-stop the worker: later
    syncs raise, and a generation request on the half-converted model raises
    instead of returning possibly corrupt output."""
    _pin_version(monkeypatch, "0.23.0")
    _pin_failing_initialize(monkeypatch, layers_converted=1)
    model = _ToyModel()
    runner = _FakeModelRunner(model)

    with pytest.raises(RuntimeError, match="synthetic vLLM failure mid-initialize"):
        fp8_utils.begin_fp8_layerwise_reload(model, tag="main", model_runner=runner)

    assert "main" in fp8_utils.fp8_state.layerwise_poisoned

    # 1. A later weight sync must refuse rather than load into the damaged model.
    monkeypatch.setitem(sys.modules, "vllm.model_executor.model_loader.reload", _make_fake_reload_module())
    with pytest.raises(RuntimeError, match="fail-stopped"):
        fp8_utils.begin_fp8_layerwise_reload(model, tag="main", model_runner=runner)

    # 2. The pre-IPC config validation must also refuse (before any socket exists).
    with pytest.raises(RuntimeError, match="fail-stopped"):
        fp8_utils.validate_fp8_layerwise_reload_config(runner.vllm_config, uses_mtp_drafter=False)

    # 3. A per-bucket load must refuse.
    with pytest.raises(RuntimeError, match="fail-stopped"):
        fp8_utils.load_quanted_weights([("model.norm.weight", torch.zeros(4))], runner)

    # 4. A generation request must raise instead of running on meta weights —
    #    at the WORKER level: vLLM's Worker.execute_model /
    #    Worker.execute_dummy_batch delegate to exactly these two runner
    #    methods, so guarding them refuses the request before the forward pass.
    with pytest.raises(RuntimeError, match="fail-stopped"):
        runner.execute_model(scheduler_output=None)
    with pytest.raises(RuntimeError, match="fail-stopped"):
        runner._dummy_run(1)
    assert runner.served == 0, "a poisoned runner still executed the model"
    # 5. And the model-level backstop still holds for callers that hold the
    #    model object directly.
    with pytest.raises(RuntimeError, match="fail-stopped"):
        model.forward(torch.zeros(1))


def test_poison_guard_names_the_real_vllm_worker_entry_points():
    """The guarded runner methods must be the ones vLLM's Worker actually calls.

    Pins the invariant to vLLM's own delegation rather than to a name we chose:
    if a future vLLM line renames ``execute_model`` / ``_dummy_run``, or verl's
    tuple drifts from them, this fails instead of silently leaving a poisoned
    worker able to serve.

    Reads ``vllm/v1/worker/gpu_worker.py`` as TEXT from the installed package
    directory rather than importing it: importing that module pulls in the whole
    engine/config chain, which is not available in a CPU-only test environment
    (and fails with an ImportError rather than a skippable ModuleNotFoundError
    on a source checkout). Skipped when vLLM is absent entirely, with the tuple
    still asserted non-empty.
    """
    assert fp8_utils._MODEL_RUNNER_SERVING_ENTRY_POINTS, "the guard must name at least one entry point"

    vllm_spec = importlib.util.find_spec("vllm")
    if vllm_spec is None or not vllm_spec.submodule_search_locations:
        pytest.skip("needs an installed vLLM to read Worker's delegation from source")
    source_path = Path(next(iter(vllm_spec.submodule_search_locations))) / "v1" / "worker" / "gpu_worker.py"
    if not source_path.is_file():
        pytest.skip(f"installed vLLM has no {source_path}")
    source = source_path.read_text()

    for method_name, entry_point in (
        ("execute_model", "execute_model"),
        ("execute_dummy_batch", "_dummy_run"),
    ):
        marker = f"    def {method_name}("
        start = source.find(marker)
        assert start != -1, f"vLLM's Worker no longer defines {method_name} in {source_path}"
        # Method body = up to the next same-indentation def.
        end = source.find("\n    def ", start + len(marker))
        body = source[start:] if end == -1 else source[start:end]
        assert f"self.model_runner.{entry_point}(" in body, (
            f"vLLM's Worker.{method_name} no longer delegates to model_runner.{entry_point}; "
            "the FP8 fail-stop guard must be re-pointed at the current entry point"
        )
        assert entry_point in fp8_utils._MODEL_RUNNER_SERVING_ENTRY_POINTS, (
            f"model_runner.{entry_point} serves requests but is not fail-stop guarded"
        )


def test_finalize_failure_poisons_the_worker(monkeypatch):
    """finalize also restores layers one at a time, so a failure there leaves
    the same partially-restored model and must fail-stop too — including the
    worker/runner serving entry points when the runner is passed in."""
    _pin_version(monkeypatch, "0.23.0")
    fake_reload = types.ModuleType("vllm.model_executor.model_loader.reload")
    fake_reload.initialize_layerwise_reload = lambda model: None

    def _finalize(model, cfg):
        raise RuntimeError("synthetic vLLM failure mid-finalize")

    fake_reload.finalize_layerwise_reload = _finalize
    monkeypatch.setitem(sys.modules, "vllm.model_executor.model_loader.reload", fake_reload)

    model = _ToyModel()
    runner = _FakeModelRunner(model)
    assert fp8_utils.begin_fp8_layerwise_reload(model, tag="main") is True
    with pytest.raises(RuntimeError, match="synthetic vLLM failure mid-finalize"):
        fp8_utils.finalize_fp8_layerwise_reload(model, model_config=None, tag="main", model_runner=runner)

    assert "main" in fp8_utils.fp8_state.layerwise_poisoned
    assert "main" not in fp8_utils.fp8_state.layerwise_active
    with pytest.raises(RuntimeError, match="fail-stopped"):
        runner.execute_model(scheduler_output=None)
    with pytest.raises(RuntimeError, match="fail-stopped"):
        runner._dummy_run(1)
    assert runner.served == 0, "a poisoned runner still executed the model"


def test_validate_config_rejects_mtp_and_unvalidated_version(monkeypatch):
    """The pre-IPC validation gate must reject exactly the configurations the
    in-loop raises used to reject, so no unsupported sync ever reaches IPC."""
    runner = _FakeModelRunner(_ToyModel())

    _pin_version(monkeypatch, "0.23.0")
    # Supported configuration: must not raise.
    fp8_utils.validate_fp8_layerwise_reload_config(runner.vllm_config, uses_mtp_drafter=False)

    with pytest.raises(NotImplementedError, match="MTP drafter"):
        fp8_utils.validate_fp8_layerwise_reload_config(runner.vllm_config, uses_mtp_drafter=True)

    _pin_version(monkeypatch, "0.25.0")
    with pytest.raises(RuntimeError, match=r"validated on vLLM >= 0\.20, < 0\.25\.0"):
        fp8_utils.validate_fp8_layerwise_reload_config(runner.vllm_config, uses_mtp_drafter=False)

    # Below 0.20 the reload module is absent: this path does not apply at all.
    _pin_version(monkeypatch, "0.19.0")
    fp8_utils.validate_fp8_layerwise_reload_config(runner.vllm_config, uses_mtp_drafter=True)
