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

"""Logical two/four-rank contracts; these mocks are not NIC/RoCE validation."""

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts.communication_network_policy import build_network_policy_eligibility
from tests.special_sanity.test_communication_network_policy import _capability, _telemetry_records

spec = importlib.util.spec_from_file_location(
    "network_executor_under_test", Path(__file__).parents[2] / "verl/utils/communication_network.py"
)
network = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = network
spec.loader.exec_module(network)


@pytest.fixture(params=[2, 4])
def runtime(monkeypatch, request):
    world = request.param
    capability = _capability(world)
    eligibility = build_network_policy_eligibility(_telemetry_records(capability), capability)
    lanes = tuple(
        network.NetworkLane(rank, rail, traffic, f"Measured-{rail}", value)
        for rank in range(world)
        for rail, traffic, value in (("rail-a", "tc-low", 0), ("rail-b", "tc-high", 32))
    )
    operations = (
        network.NetworkOperation("comm_a", "broadcast", 16, "torch.float32"),
        network.NetworkOperation("comm_b", "all_to_all_single", 16, "torch.float32"),
    )
    calls = []
    monkeypatch.setattr(network.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(network.dist, "get_world_size", lambda: world)
    monkeypatch.setattr(network.dist, "get_backend", lambda group: "gloo")
    monkeypatch.setattr(network.dist, "get_process_group_ranks", lambda group: list(range(world)))
    monkeypatch.setattr(network.socket, "gethostname", lambda: "logical-host-0")

    def gather(output, value, group):
        if isinstance(value[0], tuple) and len(value[0]) == 2 and value[0][1] == "logical-host-0":
            output[:] = [((value[0][0], f"logical-host-{rank // (world // 2)}"), value[1]) for rank in range(world)]
        else:
            output[:] = [value] * world

    monkeypatch.setattr(network.dist, "all_gather_object", gather)
    monkeypatch.setattr(
        network.dist,
        "ProcessGroupNCCL",
        SimpleNamespace(Options=lambda: SimpleNamespace(config=SimpleNamespace(net_name=None, traffic_class=None))),
        raising=False,
    )

    def new_group(**kwargs):
        config = kwargs["pg_options"].config
        calls.append(("group", config.net_name, config.traffic_class))
        return config.net_name

    monkeypatch.setattr(network.dist, "new_group", new_group)
    monkeypatch.setattr(network.dist, "destroy_process_group", lambda group: calls.append(("destroy", group)))
    work = lambda: SimpleNamespace(wait=lambda: calls.append(("wait",)))
    monkeypatch.setattr(network.dist, "all_reduce", lambda *args, **kwargs: work())
    monkeypatch.setattr(network.dist, "broadcast", lambda *args, **kwargs: work())

    def all_to_all(output, tensor, **kwargs):
        output.copy_(tensor)
        return work()

    monkeypatch.setattr(network.dist, "all_to_all_single", all_to_all)
    zeros = torch.zeros
    monkeypatch.setattr(network.torch, "zeros", lambda *args, **kwargs: zeros(*args, **{**kwargs, "device": "cpu"}))
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda *args: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda *args: SimpleNamespace(wait_event=lambda event: calls.append(("fence",)))
    )
    monkeypatch.setattr(
        torch.cuda,
        "Event",
        lambda: SimpleNamespace(
            record=lambda *args: calls.append(("event",)), synchronize=lambda: calls.append(("physical",))
        ),
    )
    monkeypatch.setattr(network, "_check_environment", lambda: None)
    kwargs = dict(
        eligibility=eligibility,
        target_telemetry=eligibility.telemetry,
        lanes=lanes,
        operations=operations,
        evidence_sha256="a" * 64,
        approved_digest=network.network_plan_digest(eligibility, lanes, operations, "a" * 64),
        control_group="control",
        observe_binding=lambda group, operation: lanes[0 if operation == "comm_a" else 1],
    )
    return kwargs, calls


def test_runtime_executes_and_fences_exact_lanes(runtime):
    kwargs, calls = runtime
    executor = network.NetworkCollectiveExecutor(**kwargs)
    assert calls[:2] == [("group", "Measured-rail-a", 0), ("group", "Measured-rail-b", 32)]
    for step in (0, 2):
        executor.begin_step(step)
        tensor = torch.arange(4, dtype=torch.float32)
        first = executor.launch("comm_a", tensor)
        second = executor.launch("comm_b", tensor)
        torch.testing.assert_close(second.wait(), tensor)
        assert first.owned[0] is tensor
        executor.finish_step()
        assert not executor._works
    executor.close()
    executor.close()
    assert sum(call[0] == "destroy" for call in calls) == 2
    assert sum(call[0] == "physical" for call in calls) == 4


@pytest.mark.parametrize(
    "change,match",
    [
        ({"approved_digest": "b" * 64}, "approved digest"),
        ({"evidence_sha256": "unbound"}, "SHA-256"),
        ({"observe_binding": lambda *args: None}, "observed communicator"),
    ],
)
def test_invalid_approval_and_observation_fail_closed(runtime, change, match):
    kwargs, _ = runtime
    with pytest.raises(ValueError, match=match):
        network.NetworkCollectiveExecutor(**{**kwargs, **change})


def test_stock_plugin_cannot_claim_rail_pinning(runtime):
    kwargs, calls = runtime
    kwargs["lanes"] = (replace(kwargs["lanes"][0], net_name="IB"), *kwargs["lanes"][1:])
    with pytest.raises(ValueError, match="rail-specific"):
        network.NetworkCollectiveExecutor(**kwargs)
    assert not calls


@pytest.mark.parametrize("failure", ["skip", "double_begin", "not_tensor", "wrong_size", "early_finish"])
def test_step_sequence_errors_are_sticky(runtime, failure):
    kwargs, _ = runtime
    executor = network.NetworkCollectiveExecutor(**kwargs)
    executor.begin_step(0)
    with pytest.raises(ValueError, match="preflight"):
        if failure == "skip":
            executor.launch("comm_b", torch.ones(4))
        elif failure == "double_begin":
            executor.begin_step(1)
        elif failure == "not_tensor":
            executor.launch("comm_a", object())
        elif failure == "wrong_size":
            executor.launch("comm_a", torch.ones(8))
        else:
            executor.finish_step()
    assert executor._failed
    with pytest.raises(ValueError, match="preflight"):
        executor.begin_step(3)
    executor.close()


def test_wait_failure_is_not_retried():
    calls = []

    def fail():
        calls.append("wait")
        raise RuntimeError("transport failed")

    tensor = torch.ones(1)
    work = network.NetworkCollectiveWork(SimpleNamespace(wait=fail), tensor, (tensor,))
    for _ in range(2):
        with pytest.raises(RuntimeError, match="transport failed"):
            work.wait()
    assert calls == ["wait"]


@pytest.mark.parametrize("key", ["NCCL_NET", "NCCL_IB_TC", "NCCL_IB_SL", "NCCL_IB_HCA"])
def test_environment_overrides_are_rejected(monkeypatch, key):
    monkeypatch.setenv(key, "unexpected")
    with pytest.raises(ValueError, match="environment overrides"):
        network._check_environment()
