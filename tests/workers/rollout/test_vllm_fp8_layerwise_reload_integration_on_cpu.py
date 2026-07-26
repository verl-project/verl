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

"""CPU integration tests driving the REAL vLLM layerwise-reload API.

Role split with ``test_vllm_fp8_layerwise_lifecycle_on_cpu.py``: that suite
stubs the vLLM reload entry points and tests verl's own lifecycle, ownership
and gate logic in isolation (it must run on a host with any vLLM, or none).
THIS suite imports the installed vLLM and calls
``record_metadata_for_reloading`` / ``initialize_layerwise_reload`` /
``finalize_layerwise_reload`` for real, so it pins the two vLLM behaviours
verl's FP8 resync path actually depends on:

  (i) initialize is NOT idempotent across buckets — a layer that completes
      during streaming ends with ``info.reset()``, so re-initializing swaps it
      back to meta and finalize then treats ``0 < load_numel < total`` as
      "delayed processing"; verl must therefore call initialize/finalize once
      per sync, outside the per-bucket loop;
  (ii) streamed tensors are BUFFERED (retained in ``info.loaded_weights``)
      until a layer is complete, so a tensor aliasing verl's reused IPC
      receive buffer must be cloned before being handed to load_weights.

These are the properties ``_vllm_supports_layerwise_reload``'s validated
interval asserts. Running this file on a new vLLM line is the cheap way to
find out whether the gate may be widened to it.

Everything here is CPU-only and model-free (a hand-written toy module whose
``load_weights`` mirrors vLLM's contract), so it runs in verl's
``*_on_cpu.py`` CI collection.
"""

import pytest
import torch

vllm_reload = pytest.importorskip(
    "vllm.model_executor.model_loader.reload",
    reason="installed vLLM does not ship the layerwise reload protocol (vLLM < 0.20)",
)

from vllm.model_executor.model_loader.reload import (  # noqa: E402
    finalize_layerwise_reload,
    initialize_layerwise_reload,
    record_metadata_for_reloading,
)
from vllm.model_executor.model_loader.reload.layerwise import get_layerwise_info  # noqa: E402

from verl.utils.vllm.vllm_fp8_utils import (  # noqa: E402
    _get_vllm_version,
    _vllm_layerwise_reload_available,
    _vllm_supports_layerwise_reload,
    begin_fp8_layerwise_reload,
    finalize_fp8_layerwise_reload,
    fp8_state,
    validate_fp8_layerwise_reload_config,
)


class _ToyLayer(torch.nn.Module):
    """One 'layer' with two params, so a bucket boundary can split it."""

    def __init__(self, size: int = 8):
        super().__init__()
        self.a = torch.nn.Parameter(torch.zeros(size, size), requires_grad=False)
        self.b = torch.nn.Parameter(torch.zeros(size, size), requires_grad=False)


class _ToyModel(torch.nn.Module):
    """Minimal stand-in for a vLLM model on the reload path.

    ``load_weights`` mirrors vLLM's contract: look the destination param up by
    name and drive its ``weight_loader`` when one is installed. Under an active
    layerwise reload vLLM replaces that attribute with its
    ``online_process_loader`` wrapper, which is exactly the machinery under
    test.
    """

    def __init__(self, size: int = 8):
        super().__init__()
        self.layer0 = _ToyLayer(size)
        self.layer1 = _ToyLayer(size)

    def load_weights(self, weights):
        params = dict(self.named_parameters(remove_duplicate=False))
        loaded = set()
        for name, tensor in weights:
            param = params[name]
            loader = getattr(param, "weight_loader", None)
            if loader is None:
                param.data.copy_(tensor)
            else:
                loader(param, tensor)
            loaded.add(name)
        return loaded


class _FakeModelConfig:
    dtype = torch.float32


@pytest.fixture(autouse=True)
def _reset_fp8_state():
    def _clear():
        fp8_state.layerwise_active.clear()
        fp8_state.layerwise_begin_attempted.clear()
        fp8_state.layerwise_poisoned.clear()

    _clear()
    yield
    _clear()


def _reference_weights(model: _ToyModel):
    """Deterministic non-zero target values, one per parameter."""
    return [
        (name, torch.full_like(param.data, float(index + 1)))
        for index, (name, param) in enumerate(model.named_parameters())
    ]


def _assert_loaded(model: _ToyModel, expected):
    actual = dict(model.named_parameters())
    for name, want in expected:
        got = actual[name].data
        assert not got.is_meta, f"{name} left on meta after finalize"
        torch.testing.assert_close(got, want, msg=f"{name} did not receive its streamed value")


def test_installed_vllm_is_inside_the_validated_interval():
    """Self-check for the rest of this file: it asserts the semantics of the
    validated interval, so it is only meaningful when the installed vLLM is in
    it. A FAIL here on a newly bumped image is the intended signal that the
    gate needs re-auditing before it is widened."""
    assert _vllm_layerwise_reload_available() is True
    assert _vllm_supports_layerwise_reload() is True, (
        f"installed vLLM {_get_vllm_version()} is outside the interval this resync path is "
        "validated on; audit the reload/ sources and widen "
        "_VLLM_LAYERWISE_RELOAD_VALIDATED_BELOW before enabling it"
    )


def test_real_reload_lifecycle_loads_every_bucket():
    """The full verl shape against the real API: ONE begin, N bucketed
    load_weights calls, ONE finalize — every streamed value must land."""
    model = _ToyModel()
    record_metadata_for_reloading(model)
    weights = _reference_weights(model)

    assert begin_fp8_layerwise_reload(model, tag="main") is True
    assert "main" in fp8_state.layerwise_active
    # Real vLLM swapped the params to meta placeholders.
    assert all(param.is_meta for _, param in model.named_parameters())

    # Stream one parameter per "bucket" so layer0 spans two buckets.
    for name, tensor in weights:
        model.load_weights([(name, tensor)])

    with torch.device("cpu"):
        finalize_fp8_layerwise_reload(model, _FakeModelConfig(), tag="main")

    assert "main" not in fp8_state.layerwise_active
    _assert_loaded(model, weights)


def test_real_initialize_is_not_idempotent_across_buckets():
    """Pins property (i): re-initializing after a layer has completed swaps
    that layer BACK to meta. This is why verl's begin/finalize live in
    update_weights_from_ipc and not in the per-bucket loader — a per-bucket
    initialize would resurrect completed layers and let finalize process
    uninitialized memory into kernel storage.
    """
    model = _ToyModel()
    record_metadata_for_reloading(model)
    weights = _reference_weights(model)

    initialize_layerwise_reload(model)

    # Complete layer0 (both of its params), leave layer1 untouched.
    layer0 = [(name, tensor) for name, tensor in weights if name.startswith("layer0.")]
    with torch.device("cpu"):
        model.load_weights(layer0)

    info0 = get_layerwise_info(model.layer0)
    assert info0.can_load() is False, "a completed layer must have been reset by _layerwise_process"
    assert not model.layer0.a.is_meta, "a completed layer must hold real data again"

    # A second initialize (what a per-bucket initialize would do) re-converts it.
    initialize_layerwise_reload(model)
    assert model.layer0.a.is_meta, (
        "re-initialize did NOT swap the completed layer back to meta: the "
        "non-idempotence this path is designed around no longer holds on "
        f"vLLM {_get_vllm_version()} — re-audit before trusting the gate"
    )

    with torch.device("cpu"):
        finalize_fp8_layerwise_reload_unmanaged(model)


def finalize_layerwise_reload_unmanaged(model):
    finalize_layerwise_reload(model, _FakeModelConfig())


# Alias kept explicit so the test above reads as "clean up the raw vLLM state"
# rather than exercising verl's tag bookkeeping (no begin_ was called).
finalize_fp8_layerwise_reload_unmanaged = finalize_layerwise_reload_unmanaged


def test_real_reload_buffers_streamed_tensors_until_layer_completes():
    """Pins property (ii): vLLM RETAINS the caller's tensor until the layer is
    complete. verl's IPC receiver reuses one buffer for every bucket, so a
    tensor that is a view into it must be cloned; this test proves the hazard
    is real by mutating the caller's tensor after handing it over.
    """
    model = _ToyModel()
    record_metadata_for_reloading(model)
    initialize_layerwise_reload(model)

    good = torch.full((8, 8), 7.0)
    # Emulate the reused IPC receive buffer: same storage for both yields.
    reused = torch.full((8, 8), 3.0)

    with torch.device("cpu"):
        model.load_weights([("layer0.a", reused)])
        # layer0 is NOT complete yet (layer0.b outstanding), so vLLM is holding
        # a reference to `reused`. The next bucket overwrites that storage.
        reused.fill_(99.0)
        model.load_weights([("layer0.b", good)])

    assert not model.layer0.a.is_meta, "layer0 should have completed once both params arrived"
    torch.testing.assert_close(
        model.layer0.a.data,
        torch.full((8, 8), 99.0),
        msg=(
            "vLLM no longer retains the caller's tensor across the bucket boundary; "
            "the clone in quant_weights would be unnecessary — re-audit ownership "
            f"on vLLM {_get_vllm_version()}"
        ),
    )

    # And the fix: cloning before handing over makes the value stable.
    model2 = _ToyModel()
    record_metadata_for_reloading(model2)
    initialize_layerwise_reload(model2)
    reused2 = torch.full((8, 8), 3.0)
    with torch.device("cpu"):
        model2.load_weights([("layer0.a", reused2.clone())])
        reused2.fill_(99.0)
        model2.load_weights([("layer0.b", good)])
    torch.testing.assert_close(model2.layer0.a.data, torch.full((8, 8), 3.0))

    with torch.device("cpu"):
        finalize_layerwise_reload_unmanaged(model)
        finalize_layerwise_reload_unmanaged(model2)


def test_real_double_begin_is_rejected_before_touching_vllm():
    """verl's lifecycle guard must fire on a second begin, so the
    non-idempotence above can never be reached through verl's own API."""
    model = _ToyModel()
    record_metadata_for_reloading(model)

    assert begin_fp8_layerwise_reload(model, tag="main") is True
    with pytest.raises(RuntimeError, match="already active"):
        begin_fp8_layerwise_reload(model, tag="main")

    with torch.device("cpu"):
        finalize_fp8_layerwise_reload(model, _FakeModelConfig(), tag="main")


def test_real_begin_failure_poisons_the_worker(monkeypatch):
    """Fault injection against the REAL API: make vLLM's initialize raise after
    it has already converted part of the model, and assert verl fail-stops
    instead of leaving a half-converted model reusable."""
    model = _ToyModel()
    record_metadata_for_reloading(model)

    real_initialize = vllm_reload.initialize_layerwise_reload

    def _initialize_then_fail(target):
        # Convert layer0 for real, then die — the exact half-converted shape.
        real_initialize(target.layer0)
        raise RuntimeError("synthetic failure after partially moving layers to meta")

    monkeypatch.setattr(vllm_reload, "initialize_layerwise_reload", _initialize_then_fail)

    class _Runner:
        def __init__(self, m):
            self.model = m
            self.vllm_config = type("_Cfg", (), {"quant_config": None})()
            self.served = 0

        # The two methods vLLM's Worker delegates serving to (verified in
        # vllm/v1/worker/gpu_worker.py at every audited tag).
        def execute_model(self, scheduler_output=None, intermediate_tensors=None):
            self.served += 1
            return "output"

        def _dummy_run(self, num_tokens=1, **kwargs):
            self.served += 1
            return "output"

    runner = _Runner(model)
    with pytest.raises(RuntimeError, match="synthetic failure after partially moving layers"):
        begin_fp8_layerwise_reload(model, tag="main", model_runner=runner)

    # The model really is half-converted: layer0 on meta, layer1 not.
    assert model.layer0.a.is_meta
    assert not model.layer1.a.is_meta

    # Attempt recorded before the vLLM call, worker poisoned, and every entry
    # point into the resync path now refuses.
    assert "main" in fp8_state.layerwise_begin_attempted
    assert "main" in fp8_state.layerwise_poisoned
    assert "main" not in fp8_state.layerwise_active

    monkeypatch.setattr(vllm_reload, "initialize_layerwise_reload", real_initialize)
    with pytest.raises(RuntimeError, match="fail-stopped"):
        begin_fp8_layerwise_reload(model, tag="main", model_runner=runner)
    with pytest.raises(RuntimeError, match="fail-stopped"):
        validate_fp8_layerwise_reload_config(runner.vllm_config, uses_mtp_drafter=False)
    # Serving is refused at the worker/runner level, ahead of the forward pass,
    # after a REAL partial initialize_layerwise_reload.
    with pytest.raises(RuntimeError, match="fail-stopped"):
        runner.execute_model(scheduler_output=None)
    with pytest.raises(RuntimeError, match="fail-stopped"):
        runner._dummy_run(1)
    assert runner.served == 0, "a poisoned runner still executed the model"
    with pytest.raises(RuntimeError, match="fail-stopped"):
        model.forward(torch.zeros(1))


def test_validate_config_runs_before_any_ipc_resource():
    """The pre-IPC gate must be callable with nothing but a config — that is
    what lets update_weights_from_ipc reject an unsupported sync before a
    socket or shared buffer exists (and therefore before the un-timed ACK
    handshake the sender is about to enter)."""
    config = type("_Cfg", (), {"quant_config": None})()

    validate_fp8_layerwise_reload_config(config, uses_mtp_drafter=False)

    with pytest.raises(NotImplementedError, match="MTP drafter"):
        validate_fp8_layerwise_reload_config(config, uses_mtp_drafter=True)
