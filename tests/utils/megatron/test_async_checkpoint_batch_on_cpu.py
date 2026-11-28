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
"""Check staging, finalization and fallback semantics of combined async writes."""

import ast
import importlib.util
import io
import os
import subprocess
import sys
import types
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import pytest
import torch

SOURCE = Path(__file__).resolve().parents[3] / "verl/utils/megatron/async_checkpoint_batch.py"
spec = importlib.util.spec_from_file_location("async_checkpoint_batch_under_test", SOURCE)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
coalesce_async_requests = module.coalesce_async_requests


class Request(NamedTuple):
    async_fn: object
    async_fn_args: tuple
    finalize_fns: list
    async_fn_kwargs: dict
    preload_fn: object
    is_frozen: bool = False

    def add_finalize_fn(self, fn):
        self.finalize_fns.append(fn)


def make_request(name, tensor, events):
    buckets = [(name, name, ([("metadata", io.BytesIO(b"meta"))], [("tensor", tensor)]))]

    def preload():
        events.append(f"preload:{name}")
        return [(name, name, ([("metadata", io.BytesIO(b"meta"))], [("tensor", tensor.clone())]))]

    def write(rank, staged, queue):
        assert rank == 7 and queue == name
        events.append((f"write:{name}", staged[0][2][1][0][1].clone()))

    def finalize():
        events.append(f"finalize:{name}")

    return Request(write, (7, buckets, name), [finalize], {}, preload)


def _native_writer(rank, staged, queue):
    return None


def _native_preload(buckets):
    return buckets


def _native_request(tensor):
    buckets = [("file", "key", ([], [("tensor", tensor)]))]
    return Request(partial(_native_writer), (0, buckets, None), [], {}, partial(_native_preload, buckets))


def test_cpu_snapshot_clones_only_recognized_requests_before_training_resumes(monkeypatch):
    monkeypatch.setattr(module, "_native_mcore_functions", lambda: (_native_writer, _native_preload))
    values = [torch.tensor([1.0]), torch.tensor([2.0])]
    requests = [_native_request(value) for value in values]
    batched, size = coalesce_async_requests(requests, 1024, lambda eligible: eligible, snapshot_cpu_tensors=True)
    assert size == 8 and len(batched) == 1
    staged = batched[0].preload_fn()
    for value in values:
        value.fill_(9)
    assert [buckets[0][2][1][0][1].item() for buckets in staged] == [1.0, 2.0]


def test_cpu_snapshot_falls_back_for_an_unknown_preloader(monkeypatch):
    monkeypatch.setattr(module, "_native_mcore_functions", lambda: (_native_writer, _native_preload))
    requests = [_native_request(torch.ones(1)), make_request("unknown", torch.ones(1), [])]
    decisions = []
    result, size = coalesce_async_requests(
        requests, 1024, lambda eligible: decisions.append(eligible) or eligible, snapshot_cpu_tensors=True
    )
    assert result is requests and size is None and decisions == [False]


@pytest.mark.parametrize("limit,peer_accepts", [(7, True), (1024, False)])
def test_cpu_snapshot_falls_back_if_budget_or_any_rank_rejects(monkeypatch, limit, peer_accepts):
    monkeypatch.setattr(module, "_native_mcore_functions", lambda: (_native_writer, _native_preload))
    requests = [_native_request(torch.ones(1)), _native_request(torch.ones(1))]
    result, size = coalesce_async_requests(
        requests, limit, lambda eligible: eligible and peer_accepts, snapshot_cpu_tensors=True
    )
    assert result is requests and size is None


def test_batch_stages_every_state_before_any_write_and_preserves_callbacks():
    events = []
    model = torch.tensor([1.0, 2.0])
    optimizer = torch.tensor([3.0, 4.0])
    requests = [make_request("model", model, events), make_request("optimizer", optimizer, events)]
    batched, estimated = coalesce_async_requests(requests, 24, lambda eligible: eligible)
    assert estimated == 24 and len(batched) == 1
    request = batched[0]
    staged = request.preload_fn()
    assert events == ["preload:model", "preload:optimizer"]
    model.fill_(9)
    optimizer.fill_(8)
    request.async_fn(request.async_fn_args[0], staged, None)
    for finalize in request.finalize_fns:
        finalize()
    assert torch.equal(events[2][1], torch.tensor([1.0, 2.0]))
    assert torch.equal(events[3][1], torch.tensor([3.0, 4.0]))
    assert [events[2][0], events[3][0], *events[4:]] == [
        "write:model",
        "write:optimizer",
        "finalize:model",
        "finalize:optimizer",
    ]


@pytest.mark.parametrize("limit", [0, 23])
def test_disabled_or_over_budget_uses_original_requests(limit):
    requests = [make_request("model", torch.ones(2), []), make_request("optimizer", torch.ones(2), [])]
    calls = []
    result, estimated = coalesce_async_requests(requests, limit, lambda eligible: calls.append(eligible) or eligible)
    assert result is requests and estimated is None
    assert calls == ([] if limit == 0 else [False])


def test_other_rank_can_force_all_ranks_to_keep_original_requests():
    requests = [make_request("model", torch.ones(2), []), make_request("optimizer", torch.ones(2), [])]
    result, estimated = coalesce_async_requests(requests, 1024, lambda eligible: False)
    assert result is requests and estimated is None


@pytest.mark.parametrize("unsupported", ["no_preload", "unknown_bucket"])
def test_unsupported_request_never_uses_combined_path(unsupported):
    requests = [make_request("model", torch.ones(2), []), make_request("optimizer", torch.ones(2), [])]
    if unsupported == "no_preload":
        requests[0] = requests[0]._replace(preload_fn=None)
    else:
        requests[0] = requests[0]._replace(async_fn_args=(7, ["unexpected"], "model"))
    decisions = []
    result, estimated = coalesce_async_requests(requests, 1024, lambda eligible: decisions.append(eligible) or eligible)
    assert result is requests and estimated is None and decisions == [False]


def test_empty_rank_still_participates_and_finalizes():
    events = []
    empty = Request(None, (), [lambda: events.append("finalize:empty")], {}, None)
    active = make_request("active", torch.ones(2), events)
    batched, estimated = coalesce_async_requests([empty, active], 1024, lambda eligible: eligible)
    assert estimated == 12
    request = batched[0]
    request.async_fn(request.async_fn_args[0], request.preload_fn(), None)
    for finalize in request.finalize_fns:
        finalize()
    assert [events[0], events[1][0], *events[2:]] == [
        "preload:active",
        "write:active",
        "finalize:empty",
        "finalize:active",
    ]


def test_manager_dispatch_schedules_one_write_and_finalizes_after_all_original_requests(monkeypatch):
    path = SOURCE.parents[1] / "checkpoint/megatron_checkpoint_manager.py"
    tree = ast.parse(path.read_text())
    manager = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MegatronCheckpointManager"
    )
    method = next(
        node for node in manager.body if isinstance(node, ast.FunctionDef) and node.name == "_dispatch_finalize"
    )
    module_ast = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))

    queued = []
    fake_base = types.ModuleType("megatron.core.dist_checkpointing.strategies.base")
    fake_base.async_calls = SimpleNamespace(schedule_async_request=queued.append)
    for mod_name, mod_obj in (
        ("megatron", types.ModuleType("megatron")),
        ("megatron.core", types.ModuleType("megatron.core")),
        ("megatron.core.dist_checkpointing", types.ModuleType("megatron.core.dist_checkpointing")),
        (
            "megatron.core.dist_checkpointing.strategies",
            types.ModuleType("megatron.core.dist_checkpointing.strategies"),
        ),
        ("megatron.core.dist_checkpointing.strategies.base", fake_base),
    ):
        monkeypatch.setitem(sys.modules, mod_name, mod_obj)

    namespace = {
        "coalesce_async_requests": coalesce_async_requests,
        "_all_ranks_agree_checkpoint_batch": lambda eligible: eligible,
        "_noop_async_checkpoint_write": lambda *args: None,
        "log_with_rank": lambda *args, **kwargs: None,
        "logger": None,
    }
    exec(compile(module_ast, str(path), "exec"), namespace)

    class Config:
        async_save = True
        async_save_coalesce_max_staging_mib = 1

    manager_stub = SimpleNamespace(checkpoint_config=Config(), rank=0)
    events = []
    requests = [make_request("model", torch.ones(2), events), make_request("optimizer", torch.ones(2), events)]
    ok = namespace["_dispatch_finalize"](manager_stub, requests, lambda: events.append("finish:role"))
    assert ok is True and len(queued) == 1
    request = queued[0]
    request.async_fn(request.async_fn_args[0], request.preload_fn(), None)
    for callback in request.finalize_fns:
        callback()
    assert events[-3:] == ["finalize:model", "finalize:optimizer", "finish:role"]


@pytest.mark.parametrize("limit", [-1, True, 1.5])
def test_rejects_invalid_staging_limit(limit):
    with pytest.raises(ValueError, match="non-negative integer"):
        coalesce_async_requests([], limit, lambda eligible: eligible)


def test_combined_request_can_run_in_a_spawned_process():
    script = Path(__file__).with_name("checkpoint_batch_spawn_probe.py")
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        env=dict(os.environ, OMP_NUM_THREADS="1"),
        timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "SPAWN_BATCH_OK" in result.stdout


def test_cpu_snapshot_survives_a_concurrent_shared_tensor_update():
    script = Path(__file__).with_name("checkpoint_cpu_snapshot_spawn_probe.py")
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        env=dict(os.environ, OMP_NUM_THREADS="1"),
        timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "CPU_SNAPSHOT_OK" in result.stdout
