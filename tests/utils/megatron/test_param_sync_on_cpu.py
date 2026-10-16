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

"""Exercise parameter publication without CUDA/MCore; consumers use their actual source.

The fake Work publishes new parameter bytes only when waited on, allowing these
tests to detect stale numerical results as well as the next-step handle assertion.
This is CPU lifecycle coverage, not a distributed GPU execution test.
"""

import ast
import gc
import importlib.util
import sys
import types
import weakref
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("_test_verl_param_sync", ROOT / "verl/utils/megatron/param_sync.py")
param_sync = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(param_sync)


class _Work:
    def __init__(self, bucket):
        self.bucket = bucket

    def wait(self):
        self.bucket.waits += 1
        with torch.no_grad():
            for param in self.bucket.params:
                param.copy_(self.bucket.target)


class _Bucket:
    def __init__(self, *params):
        self.params = params
        self.param_gather_dispatched = False
        self.param_gather_handle = None
        self.calls = []
        self.waits = 0
        self.target = torch.ones_like(params[0])

    def start_param_sync(self):
        assert self.param_gather_handle is None, "previous parameter gather still in flight"
        self.param_gather_handle = _Work(self)
        self.param_gather_dispatched = True

    def finish_param_sync(self, skip_next_bucket_dispatch=False):
        self.calls.append(skip_next_bucket_dispatch)
        if not self.param_gather_dispatched:
            self.start_param_sync()
        if self.param_gather_handle is not None:
            self.param_gather_handle.wait()
            self.param_gather_handle = None


class _DDP:
    def __init__(self, *params, overlap=True, align=False, optimizer_overlap=False):
        self.module = torch.nn.ParameterList(params)
        self.ddp_config = SimpleNamespace(overlap_param_gather=overlap, align_param_gather=align)
        self.overlap_param_gather_with_optimizer_step = optimizer_overlap
        self.remove_forward_pre_hook_handles = {"output_layer": object()}
        self.bucket = _Bucket(*params)
        self.param_to_bucket_group = dict.fromkeys(params, self.bucket)


@pytest.fixture(autouse=True)
def legacy_mcore(monkeypatch):
    # Only stub the dependency used at registration; no global package replacement.
    distributed = types.ModuleType("megatron.core.distributed")
    distributed.DistributedDataParallel = _DDP
    monkeypatch.setitem(sys.modules, "megatron.core.distributed", distributed)
    monkeypatch.setattr(param_sync, "_mcore_ensure_params_ready", None)
    monkeypatch.setattr(param_sync, "PARAM_READY_CALLBACK_ATTR", None)
    monkeypatch.setattr(param_sync, "_is_graph_capturing", lambda: False)


@pytest.mark.parametrize("align,optimizer_overlap", [(False, False), (True, False), (False, True), (True, True)])
def test_legacy_callback_preserves_ddp_prefetch_policy(align, optimizer_overlap):
    weight = torch.nn.Parameter(torch.zeros(3, 2))
    ddp = _DDP(weight, align=align, optimizer_overlap=optimizer_overlap)
    param_sync.register_ddp_param_ready_callbacks([ddp])
    param_sync.ensure_fused_weight_ready(weight)
    param_sync.ensure_fused_weight_ready(weight)
    assert ddp.bucket.calls == [align or optimizer_overlap]
    assert ddp.bucket.waits == 1
    torch.testing.assert_close(weight, ddp.bucket.target)


def test_compatibility_uses_newer_ddp_finish_method(monkeypatch):
    weight = torch.nn.Parameter(torch.zeros(3, 2))
    ddp = _DDP(weight)
    calls = []

    def finish(bucket):
        calls.append(bucket)
        bucket.finish_param_sync(skip_next_bucket_dispatch=True)

    ddp._finish_param_sync_for_bucket_group = finish
    param_sync.register_ddp_param_ready_callbacks(ddp)
    param_sync.ensure_fused_weight_ready(weight)
    assert calls == [ddp.bucket]
    assert ddp.bucket.calls == [True]


@pytest.mark.parametrize("in_flight", [False, True])
def test_external_schedule_only_drains_an_existing_gather(in_flight):
    weight = torch.nn.Parameter(torch.zeros(3, 2))
    ddp = _DDP(weight)
    ddp.remove_forward_pre_hook_handles.clear()
    if in_flight:
        ddp.bucket.start_param_sync()
    param_sync.register_ddp_param_ready_callbacks(ddp)
    param_sync.ensure_fused_weight_ready(weight)
    assert ddp.bucket.calls == ([True] if in_flight else [])
    assert ddp.bucket.param_gather_handle is None
    assert ddp.bucket.param_gather_dispatched is in_flight


def test_graph_capture_does_not_capture_a_new_collective(monkeypatch):
    weight = torch.nn.Parameter(torch.zeros(3, 2))
    ddp = _DDP(weight)
    param_sync.register_ddp_param_ready_callbacks(ddp)
    monkeypatch.setattr(param_sync, "_is_graph_capturing", lambda: True)
    param_sync.ensure_fused_weight_ready(weight)
    assert not ddp.bucket.param_gather_dispatched
    assert ddp.bucket.calls == []


def test_native_mcore_callback_takes_precedence(monkeypatch):
    weight = torch.nn.Parameter(torch.zeros(3, 2))
    ddp = _DDP(weight)
    calls = []
    weight.native_ready = lambda: calls.append("native")
    monkeypatch.setattr(param_sync, "PARAM_READY_CALLBACK_ATTR", "native_ready")
    monkeypatch.setattr(param_sync, "_mcore_ensure_params_ready", lambda params: params[0].native_ready())
    param_sync.register_ddp_param_ready_callbacks(ddp)
    param_sync.ensure_fused_weight_ready(weight)
    assert calls == ["native"]
    assert not hasattr(weight, param_sync._VERL_PARAM_READY_CALLBACK_ATTR)
    assert ddp.bucket.calls == []


def test_rewrap_replaces_old_owner_and_disabling_overlap_removes_marker():
    weight = torch.nn.Parameter(torch.zeros(3, 2))
    old = _DDP(weight)
    param_sync.register_ddp_param_ready_callbacks(old)
    new = _DDP(weight)
    new.bucket.target.fill_(2)
    param_sync.register_ddp_param_ready_callbacks(new)
    param_sync.ensure_fused_weight_ready(weight)
    assert old.bucket.calls == []
    assert new.bucket.calls == [False]
    torch.testing.assert_close(weight, new.bucket.target)

    disabled = _DDP(weight, overlap=False)
    param_sync.register_ddp_param_ready_callbacks(disabled)
    assert not hasattr(weight, param_sync._VERL_PARAM_READY_CALLBACK_ATTR)
    param_sync.ensure_fused_weight_ready(weight)
    assert disabled.bucket.calls == []


def test_callback_does_not_retain_ddp_or_bucket():
    weight = torch.nn.Parameter(torch.zeros(3, 2))
    ddp = _DDP(weight)
    param_sync.register_ddp_param_ready_callbacks(ddp)
    ddp_ref, bucket_ref = weakref.ref(ddp), weakref.ref(ddp.bucket)
    del ddp
    gc.collect()
    assert ddp_ref() is None and bucket_ref() is None
    param_sync.ensure_fused_weight_ready(weight)


def test_bucket_callback_is_shared_and_unbuffered_params_are_not_marked():
    first, second = (torch.nn.Parameter(torch.zeros(3, 2)) for _ in range(2))
    ddp = _DDP(first, second)
    frozen = torch.nn.Parameter(torch.zeros(3, 2), requires_grad=False)
    ddp.module.append(frozen)
    param_sync.register_ddp_param_ready_callbacks([ddp, torch.nn.Linear(2, 3)])
    attr = param_sync._VERL_PARAM_READY_CALLBACK_ATTR
    assert getattr(first, attr) is getattr(second, attr)
    assert not hasattr(frozen, attr)
    param_sync.ensure_fused_weight_ready(first)
    param_sync.ensure_fused_weight_ready(second)
    assert ddp.bucket.waits == 1


def test_unknown_ddp_overlap_contract_fails_at_registration():
    ddp = _DDP(torch.nn.Parameter(torch.zeros(3, 2)))
    del ddp.param_to_bucket_group
    with pytest.raises(RuntimeError, match="param_to_bucket_group"):
        param_sync.register_ddp_param_ready_callbacks(ddp)


def _load_consumer(path, name, namespace):
    # Execute the real consumer body in a CPU dependency harness. The ordinary
    # module import requires the CUDA/TE/Triton stack; its math is mocked here.
    tree = ast.parse((ROOT / path).read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, node], type_ignores=[]))
    exec(compile(module, path, "exec"), namespace)
    return namespace[name]


def _consumer(path_kind, weight, tied, ensure_ready):
    hidden = torch.tensor([[[0.5, -0.25]]], requires_grad=True)
    labels = torch.tensor([0])

    def linear_cross_entropy(hidden, weight, labels, *_args):
        logits = hidden.reshape(-1, 2) @ weight.t()
        log_probs = logits.log_softmax(-1)
        return log_probs[:, 0], -(log_probs.exp() * log_probs).sum(-1)

    namespace = {
        "torch": torch,
        "OrderedDict": OrderedDict,
        "CausalLMOutputForPPO": SimpleNamespace,
        "ensure_fused_weight_ready": ensure_ready,
        "linear_cross_entropy": linear_cross_entropy,
        "parallel_state": SimpleNamespace(get_tensor_model_parallel_group=lambda: None),
        "gather_from_sequence_parallel_region": lambda x: x,
        "has_config_logger_enabled": lambda config: False,
        "deprecate_inference_params": lambda ctx, params: ctx,
    }
    model = SimpleNamespace(
        config=SimpleNamespace(sequence_parallel=False, moe_n_hash_layers=0),
        training=True,
        post_process=True,
        mtp_process=False,
        share_embeddings_and_output_weights=tied,
        shared_embedding_or_output_weight=lambda: weight,
        embedding=SimpleNamespace(word_embeddings=SimpleNamespace(weight=weight)),
        output_layer=SimpleNamespace(weight=None if tied else weight),
        _preprocess=lambda **kwargs: (hidden, None, None, None, None),
        decoder=lambda **kwargs: hidden,
    )
    if path_kind == "native":
        forward = _load_consumer("verl/models/mcore/model_forward_fused.py", "fused_output_processor", namespace)
        output = forward(
            hidden_states=hidden,
            output_layer=model.output_layer,
            output_weight=weight if tied else None,
            labels=labels,
            context=SimpleNamespace(temperature=1.0),
            config=model.config,
        )
        log_probs = output.log_probs
    else:
        forward = _load_consumer("verl/models/mcore/model_forward_fused.py", "_fused_GPTModel_forward", namespace)
        log_probs = forward(model, input_ids=labels, position_ids=labels, attention_mask=None, labels=labels).log_probs
    return log_probs, torch.autograd.grad(log_probs.sum(), (hidden, weight))


@pytest.mark.parametrize("path_kind", ["native", "legacy"])
@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("ready_backend", ["compatibility", "native"])
def test_consumers_publish_fresh_weights_and_drain_handles_across_steps(path_kind, tied, ready_backend, monkeypatch):
    weight = torch.nn.Parameter(torch.zeros(3, 2))
    ddp = _DDP(weight)
    if ready_backend == "native":
        weight.native_ready = param_sync._DDPParamReadyCallback(ddp, ddp.bucket)
        monkeypatch.setattr(param_sync, "PARAM_READY_CALLBACK_ATTR", "native_ready")
        monkeypatch.setattr(param_sync, "_mcore_ensure_params_ready", lambda params: params[0].native_ready())
    param_sync.register_ddp_param_ready_callbacks(ddp)
    for step in range(3):
        ddp.bucket.target = torch.tensor([[1.0, 2.0], [3.0, 4.0], [-1.0, 0.0]]) * (step + 1)
        ddp.bucket.start_param_sync()
        actual, actual_grads = _consumer(path_kind, weight, tied, param_sync.ensure_fused_weight_ready)
        reference_weight = torch.nn.Parameter(ddp.bucket.target.clone())
        expected, expected_grads = _consumer(path_kind, reference_weight, tied, lambda weight: None)
        torch.testing.assert_close(actual, expected)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
            torch.testing.assert_close(actual_grad, expected_grad)
        assert ddp.bucket.param_gather_handle is None
        # MCore resets dispatch at the end of each optimizer step.
        ddp.bucket.param_gather_dispatched = False
    assert ddp.bucket.waits == 3


@pytest.mark.parametrize("path_kind", ["native", "legacy"])
def test_missing_publication_reads_stale_weights_and_fails_next_step(path_kind):
    weight = torch.nn.Parameter(torch.zeros(3, 2))
    ddp = _DDP(weight)
    ddp.bucket.target = torch.tensor([[1.0, 2.0], [3.0, 4.0], [-1.0, 0.0]])
    ddp.bucket.start_param_sync()
    actual, _ = _consumer(path_kind, weight, False, lambda weight: None)
    expected, _ = _consumer(path_kind, torch.nn.Parameter(ddp.bucket.target.clone()), False, lambda weight: None)
    assert not torch.allclose(actual, expected)
    ddp.bucket.param_gather_dispatched = False
    with pytest.raises(AssertionError, match="previous parameter gather"):
        ddp.bucket.start_param_sync()
