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
"""Production-method contracts for single-node multi-PD; no device or Ray startup."""

import asyncio
import copy
import os
import socket
from dataclasses import dataclass, replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from test_pd_contracts_on_cpu import REPLICA, SERVER, _load_functions, _metrics_pd_replica, _remote

ADAPTER = "verl/workers/rollout/vllm_rollout/vllm_rollout.py"
UTILS = "verl/workers/rollout/vllm_rollout/utils.py"


@pytest.mark.parametrize("reason", ["aborted", "abort"])
def test_aborted_prefill_returns_without_decode_and_releases_reservation(reason):
    dispatch = _load_functions(SERVER, ["_pd_dispatch"], "vLLMHttpServer")._pd_dispatch
    decode = AsyncMock()
    peer = SimpleNamespace(generate=SimpleNamespace(remote=decode))
    result = SimpleNamespace(stop_reason=reason, extra_fields={})
    replica = SimpleNamespace(
        _select_decode_peer_for_dispatch=AsyncMock(return_value=(peer, "d0")),
        _release_pd_decode_peer=AsyncMock(),
        _disaggregation_kv_transfer_config={"kv_connector": "MooncakeConnectorV1"},
        _build_pd_decode_kv_params=Mock(side_effect=AssertionError("aborted prefill has no KV")),
        generate=AsyncMock(return_value=result),
    )
    original_sampling = {"max_tokens": 100, "ignore_eos": False}
    assert asyncio.run(dispatch(replica, [1], original_sampling, "turn")) is result
    decode.assert_not_awaited()
    assert replica.generate.await_args.args[1] == {"max_tokens": 1, "ignore_eos": True}
    assert original_sampling == {"max_tokens": 100, "ignore_eos": False}
    reservation = replica._select_decode_peer_for_dispatch.await_args.kwargs["reservation_id"]
    replica._release_pd_decode_peer.assert_awaited_once_with("d0", "turn", reservation_id=reservation)


@pytest.mark.parametrize("device,backend", [("cuda", "nixl"), ("cuda", "mooncake"), ("npu", "mooncake")])
@pytest.mark.parametrize("shape", [(1, 3, 4, 4), (2, 2, 4, 4), (2, 3, 2, 4), (3, 1, 2, 2)])
def test_all_engines_have_disjoint_workers_and_every_prefill_gets_all_decodes(device, backend, shape):
    np, nd, ptp, dtp = shape
    replica, spawned = _metrics_pd_replica(2, device, backend, np, nd, ptp, dtp)
    asyncio.run(replica.launch_servers())
    assert len(spawned) == np + nd
    assert [o["role"] for o, _ in spawned] == ["prefill"] * np + ["decode"] * nd
    assert [w for options, _ in spawned for w in options["workers"]] == replica.workers
    offsets = [i * ptp for i in range(np)] + [np * ptp + i * dtp for i in range(nd)]
    assert [o["zmq_base_trainer_rank"] for o, _ in spawned] == offsets
    assert [o["tp"] for o, _ in spawned] == [ptp] * np + [dtp] * nd
    assert len({o["actor_name"] for o, _ in spawned}) == np + nd
    assert len({o["kv_transfer_config"]["engine_id"] for o, _ in spawned}) == np + nd
    for options, server in spawned[:np]:
        if device == "cuda" and backend == "mooncake":
            assert options["mooncake_bootstrap_port"] == options["side_channel_port"]
        server.set_pd_peer.remote.assert_awaited_once_with(
            decode_peers=[s for _, s in spawned[np:]],
            prefill_side_channel_port=options["side_channel_port"],
            prefill_engine_id=options["kv_transfer_config"]["engine_id"],
        )
    assert [h for _, h in replica.get_request_server_endpoints()] == [s for _, s in spawned[:np]]
    assert len(replica.get_metrics_server_endpoints()) == np + nd


def test_cross_node_workers_fail_before_spawning_any_engine():
    replica, spawned = _metrics_pd_replica(0, "npu", "mooncake", 2, 2)
    replica.workers[-1].__ray_call__ = _remote(lambda fn: ("other-node", "7", "::2"))
    with pytest.raises(ValueError, match="same node"):
        asyncio.run(replica.launch_servers())
    assert spawned == []
    replica._reserve_pd_ports.assert_not_called()


class _ConfigOwner:
    def __init__(self, config, model_config, device_mesh):
        self.config = config


@pytest.mark.parametrize("shape", [(1, 3, 4, 4), (2, 2, 4, 4), (2, 3, 2, 4), (3, 1, 2, 2)])
def test_trainer_actor_and_receiver_ipc_mapping_match_launch_for_every_rank(monkeypatch, shape):
    np, nd, ptp, dtp = shape
    footprint = np * ptp + nd * dtp
    lookup = Mock(side_effect=lambda name: name)
    adapter_cls = _load_functions(
        ADAPTER,
        ["ServerAdapter"],
        BaseRollout=_ConfigOwner,
        os=os,
        ray=SimpleNamespace(get_actor=lookup, get_runtime_context=lambda: SimpleNamespace(get_job_id=lambda: "job")),
        torch=SimpleNamespace(no_grad=lambda: lambda f: f),
        VLLM_SLEEP_LEVEL=2,
        is_support_ipc=lambda: True,
    ).ServerAdapter
    receiver = _load_functions(
        UTILS,
        ["_get_zmq_handle"],
        "vLLMColocateWorkerExtension",
        os=os,
        _resolve_vllm_weight_sync_local_rank=lambda rank, cfg: rank,
    )._get_zmq_handle
    config = SimpleNamespace(
        tensor_model_parallel_size=ptp,
        data_parallel_size=1,
        pipeline_model_parallel_size=1,
        layered_summon=False,
        disaggregation=SimpleNamespace(
            enabled=True, prefill_replicas=np, decode_replicas=nd, decode_tensor_model_parallel_size=dtp
        ),
    )
    config.get = lambda key, default=None: getattr(config, key, default)
    monkeypatch.setenv("RAY_LOCAL_WORLD_SIZE", "16")
    monkeypatch.setenv("VERL_RAY_JOB_ID", "job")
    for replica_rank in range(2):
        replica, spawned = _metrics_pd_replica(replica_rank, "npu", "mooncake", *shape)
        asyncio.run(replica.launch_servers())
        for options, _ in spawned:
            offset = options["zmq_base_trainer_rank"]
            for local in range(options["tp"]):
                monkeypatch.setenv("RANK", str(replica_rank * footprint + offset + local))
                adapter = adapter_cls(config, None, None)
                assert adapter.replica_rank == replica_rank
                assert adapter._pd_role == options["role"]
                assert adapter._pd_tp_local_rank == local
                assert adapter._ensure_server_handle() is (local == 0)
                if local == 0:
                    assert adapter.server_handle == options["actor_name"]
                monkeypatch.setenv("VERL_REPLICA_RANK", str(replica_rank))
                monkeypatch.setenv("VERL_ZMQ_BASE_TRAINER_RANK", str(offset))
                assert adapter.zmq_handle == receiver(SimpleNamespace(local_rank=local, model_runner=None))
    assert lookup.call_count == 2 * (np + nd)


@pytest.mark.parametrize("np,nd,ptp,dtp,expected", [(1, 3, 4, 4, 16), (2, 2, 4, 4, 16), (3, 1, 2, 2, 8)])
def test_replica_initialization_counts_all_prefill_workers(np, nd, ptp, dtp, expected):
    class Parent:
        def __init__(self, replica_rank, config, model_config, gpus_per_node, *args):
            self.replica_rank = replica_rank
            self.config = config
            self.gpus_per_node = gpus_per_node

    cls = _load_functions(REPLICA, ["vLLMPDReplica"], vLLMReplica=Parent).vLLMPDReplica
    config = SimpleNamespace(
        tensor_model_parallel_size=ptp,
        data_parallel_size=1,
        pipeline_model_parallel_size=1,
        disaggregation=SimpleNamespace(
            enabled=True,
            transfer_backend="mooncake",
            prefill_replicas=np,
            decode_replicas=nd,
            decode_tensor_model_parallel_size=dtp,
        ),
    )
    replica = cls(0, config, None, gpus_per_node=16)
    assert replica.world_size == expected
    assert replica.nnodes == 1
    with pytest.raises(NotImplementedError, match="single-node only"):
        cls(0, config, None, gpus_per_node=expected - 1)
    config.data_parallel_size = 2
    with pytest.raises(NotImplementedError, match="data_parallel_size=1"):
        cls(0, config, None, gpus_per_node=16)
    config.data_parallel_size = 1
    config.pipeline_model_parallel_size = 2
    with pytest.raises(NotImplementedError, match="pipeline_model_parallel_size=1"):
        cls(0, config, None, gpus_per_node=16)


@pytest.mark.parametrize("role", ["prefill", "decode"])
@pytest.mark.parametrize("value", [0, -0.1, 1.1])
def test_invalid_role_memory_fraction_is_rejected(role, value):
    validate = _load_functions(
        "verl/workers/config/disaggregation.py",
        ["__post_init__"],
        "DisaggregationConfig",
        _ALLOWED_BACKENDS=("mooncake",),
        _ALLOWED_MOONCAKE_PROTOCOLS=("tcp",),
    ).__post_init__
    config = SimpleNamespace(
        enabled=True,
        transfer_backend="mooncake",
        mooncake_protocol="tcp",
        prefill_replicas=2,
        decode_replicas=2,
        decode_tensor_model_parallel_size=4,
        prefill_gpu_memory_utilization=None,
        decode_gpu_memory_utilization=None,
        prefill_engine_kwargs={},
        decode_engine_kwargs={},
        bootstrap_port=None,
    )
    setattr(config, f"{role}_gpu_memory_utilization", value)
    with pytest.raises(ValueError, match=f"{role}_gpu_memory_utilization"):
        validate(config)


@pytest.mark.parametrize("role", ["prefill", "decode"])
@pytest.mark.parametrize(
    "key",
    [
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "data_parallel_size",
        "data_parallel_size_local",
        "data_parallel_start_rank",
        "kv_transfer_config",
        "worker_extension_cls",
    ],
)
@pytest.mark.parametrize("dashed", [False, True])
@pytest.mark.parametrize("value", [8, None])
def test_role_overrides_cannot_change_the_planned_topology(role, key, dashed, value):
    validate = _load_functions(
        "verl/workers/config/disaggregation.py",
        ["__post_init__"],
        "DisaggregationConfig",
        _ALLOWED_BACKENDS=("mooncake",),
        _ALLOWED_MOONCAKE_PROTOCOLS=("tcp",),
    ).__post_init__
    config = SimpleNamespace(
        enabled=True,
        transfer_backend="mooncake",
        mooncake_protocol="tcp",
        prefill_replicas=2,
        decode_replicas=2,
        decode_tensor_model_parallel_size=4,
        prefill_gpu_memory_utilization=None,
        decode_gpu_memory_utilization=None,
        prefill_engine_kwargs={},
        decode_engine_kwargs={},
        bootstrap_port=None,
    )
    key = key.replace("_", "-") if dashed else key
    setattr(config, f"{role}_engine_kwargs", {key: value})
    if value is None:
        validate(config)
    else:
        with pytest.raises(ValueError, match="cannot override PD topology"):
            validate(config)


@pytest.mark.parametrize(
    "device,backend", [("npu", "mooncake"), ("npu", "nixl"), ("cuda", "mooncake"), ("cuda", "nixl")]
)
def test_sleep_drains_every_prefill_and_orders_mooncake_consumers_first(device, backend):
    sleep = _load_functions(REPLICA, ["sleep"], "vLLMPDReplica").sleep
    events = []

    def server(name):
        return SimpleNamespace(
            wait_for_requests_to_drain=_remote(lambda: events.append(("drain", name))),
            sleep=_remote(lambda: events.append(("sleep", name))),
        )

    prefills = [server("p0"), server("p1")]
    decodes = [server("d0"), server("d1")]
    replica = SimpleNamespace(
        _prefill_servers=prefills,
        _decode_servers=decodes,
        servers=prefills + decodes,
        _is_ascend_platform=lambda: device == "npu",
        config=SimpleNamespace(disaggregation=SimpleNamespace(transfer_backend=backend)),
    )
    asyncio.run(sleep(replica))
    order = ["d0", "d1", "p0", "p1"] if device == "npu" or backend == "mooncake" else ["p0", "p1", "d0", "d1"]
    assert events == [("drain", "p0"), ("drain", "p1")] + [("sleep", name) for name in order]


def test_decode_sleep_failure_does_not_release_prefill():
    sleep = _load_functions(REPLICA, ["sleep"], "vLLMPDReplica").sleep
    p_sleep = AsyncMock()
    p = SimpleNamespace(
        wait_for_requests_to_drain=SimpleNamespace(remote=AsyncMock()), sleep=SimpleNamespace(remote=p_sleep)
    )
    d = SimpleNamespace(sleep=SimpleNamespace(remote=AsyncMock(side_effect=RuntimeError("unregister failed"))))
    replica = SimpleNamespace(
        _prefill_servers=[p], _decode_servers=[d], servers=[p, d], _is_ascend_platform=lambda: True
    )
    with pytest.raises(RuntimeError, match="unregister failed"):
        asyncio.run(sleep(replica))
    p_sleep.assert_not_awaited()


@pytest.mark.parametrize("contiguous", [False, True])
def test_each_engine_port_reservation_is_disjoint(contiguous):
    def allocate(address, with_alive_sock):
        sock = socket.socket()
        sock.bind((address, 0))
        return sock.getsockname()[1], sock

    methods = _load_functions(
        REPLICA,
        ["_reserve_pd_ports", "_reserve_port_range"],
        "vLLMPDReplica",
        socket=socket,
        get_free_port=allocate,
        is_valid_ipv6_address=lambda host: False,
    )
    replica = SimpleNamespace(
        _n_prefill=2,
        _n_decode=3,
        _prefill_tp=2,
        _decode_tp=4,
        world_size=16,
        _reserve_port_range=methods._reserve_port_range,
    )
    pp, bootstrap, dp, sockets, close_before_launch = methods._reserve_pd_ports(replica, "127.0.0.1", contiguous)
    try:
        assert len(pp) == 2 and len(dp) == 3
        actual_ports = [sock.getsockname()[1] for sock in sockets]
        assert len(actual_ports) == len(set(actual_ports))
        if contiguous:
            expected = [port + rank for ports, tp in ((pp, 2), (dp, 4)) for port in ports for rank in range(tp)]
            expected += list(range(bootstrap, bootstrap + 16))
            assert set(expected) == set(actual_ports)
        else:
            assert bootstrap == pp[0]
            assert set(pp + dp) == set(actual_ports)
        assert close_before_launch is contiguous
    finally:
        for sock in sockets:
            sock.close()


@dataclass
class _RoleConfig:
    disaggregation: object
    engine_kwargs: dict
    tensor_model_parallel_size: int = 4
    gpu_memory_utilization: float = 0.5
    ray_actor_max_concurrency: int = 128


def _spawn_role_configs(config):
    """Capture both roles' actor configs without starting Ray or vLLM."""
    spawn = _load_functions(
        REPLICA,
        ["_spawn_pd_server"],
        "vLLMPDReplica",
        copy=copy,
        _dc_replace=replace,
        os=os,
        ray=SimpleNamespace(
            get_runtime_context=lambda: SimpleNamespace(get_job_id=lambda: "job"),
            util=SimpleNamespace(scheduling_strategies=SimpleNamespace(NodeAffinitySchedulingStrategy=lambda **kw: kw)),
        ),
    )._spawn_pd_server
    actor = Mock()
    actor.options.return_value.remote.side_effect = lambda **kwargs: kwargs
    replica = SimpleNamespace(
        config=config,
        server_class=actor,
        model_config=None,
        rollout_mode=None,
        replica_rank=0,
        gpus_per_replica_node=16,
    )
    outputs = []
    for role, offset in (("prefill", 0), ("decode", 4)):
        outputs.append(spawn(replica, role, [], "node", "0,1", 2, {}, "127.0.0.1", 39000, 40000, role, offset))
    return [result["config"] for result in outputs], actor


def test_role_overrides_are_isolated_and_default_memory_is_inherited():
    config = _RoleConfig(
        disaggregation=SimpleNamespace(
            prefill_gpu_memory_utilization=0.7,
            decode_gpu_memory_utilization=None,
            prefill_engine_kwargs={"max_num_batched_tokens": 65536, "max_num_seqs": None},
            decode_engine_kwargs={"max_num_batched_tokens": 8192},
        ),
        engine_kwargs={"vllm": {"max_num_seqs": 32, "compilation_config": {"cudagraph_mode": "NONE"}}},
    )
    (p, d), actor = _spawn_role_configs(config)
    assert p.gpu_memory_utilization == 0.7 and d.gpu_memory_utilization == 0.5
    assert p.tensor_model_parallel_size == d.tensor_model_parallel_size == 2
    assert p.engine_kwargs["vllm"]["max_num_batched_tokens"] == 65536
    assert d.engine_kwargs["vllm"]["max_num_batched_tokens"] == 8192
    assert p.engine_kwargs["vllm"]["max_num_seqs"] == d.engine_kwargs["vllm"]["max_num_seqs"] == 32
    p.engine_kwargs["vllm"]["compilation_config"]["cudagraph_mode"] = "PIECEWISE"
    assert d.engine_kwargs["vllm"]["compilation_config"]["cudagraph_mode"] == "NONE"
    assert config.engine_kwargs == {"vllm": {"max_num_seqs": 32, "compilation_config": {"cudagraph_mode": "NONE"}}}
    assert [
        call.kwargs["runtime_env"]["env_vars"]["VERL_ZMQ_BASE_TRAINER_RANK"] for call in actor.options.call_args_list
    ] == ["0", "4"]


@pytest.mark.parametrize("engine_kwargs", [{}, {"vllm": None}, {"vllm": {}}])
def test_missing_or_null_vllm_kwargs_accept_role_overrides(engine_kwargs):
    original = copy.deepcopy(engine_kwargs)
    config = _RoleConfig(
        disaggregation=SimpleNamespace(
            prefill_gpu_memory_utilization=None,
            decode_gpu_memory_utilization=None,
            prefill_engine_kwargs={},
            decode_engine_kwargs={"max_num_batched_tokens": 8192},
        ),
        engine_kwargs=engine_kwargs,
    )
    (p, d), _ = _spawn_role_configs(config)
    assert p.engine_kwargs["vllm"] == {}
    assert d.engine_kwargs["vllm"] == {"max_num_batched_tokens": 8192}
    assert p.gpu_memory_utilization == d.gpu_memory_utilization == 0.5
    assert config.engine_kwargs == original


@pytest.mark.parametrize("role", ["prefill", "decode"])
@pytest.mark.parametrize("memory", [None, 0.6])
@pytest.mark.parametrize("key", ["gpu_memory_utilization", "gpu-memory-utilization"])
def test_explicit_role_memory_takes_priority_over_engine_kwargs(role, memory, key):
    config = _RoleConfig(
        disaggregation=SimpleNamespace(
            prefill_gpu_memory_utilization=None,
            decode_gpu_memory_utilization=None,
            prefill_engine_kwargs={},
            decode_engine_kwargs={},
        ),
        engine_kwargs={"vllm": {key: 0.8, "max_num_seqs": 32}},
    )
    setattr(config.disaggregation, f"{role}_gpu_memory_utilization", memory)
    # Include a conflicting role kwarg as well: the explicit field wins over both.
    setattr(config.disaggregation, f"{role}_engine_kwargs", {key: 0.9})
    (p, d), _ = _spawn_role_configs(config)
    selected, other = (p, d) if role == "prefill" else (d, p)
    if memory is None:
        assert selected.gpu_memory_utilization == 0.5
        assert selected.engine_kwargs["vllm"] == {key: 0.9, "max_num_seqs": 32}
    else:
        assert selected.gpu_memory_utilization == 0.6
        assert selected.engine_kwargs["vllm"] == {"gpu_memory_utilization": 0.6, "max_num_seqs": 32}
    assert other.gpu_memory_utilization == 0.5
    assert other.engine_kwargs["vllm"] == {key: 0.8, "max_num_seqs": 32}
    assert config.engine_kwargs == {"vllm": {key: 0.8, "max_num_seqs": 32}}
