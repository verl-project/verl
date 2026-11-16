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
"""Exercise production PD methods without importing vLLM or starting Ray.

AST extraction deliberately limits these to Python contract tests. Hardware,
worker initialization and connector integration require separate validation.
"""

from __future__ import annotations

import ast
import asyncio
import logging
import socket
import types
import uuid
from ipaddress import ip_address
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

ROOT = Path(__file__).resolve().parents[3]
SERVER = "verl/workers/rollout/vllm_rollout/vllm_async_server.py"
REPLICA = "verl/workers/rollout/vllm_rollout/vllm_pd_replica.py"
BASE_REPLICA = "verl/workers/rollout/replica.py"
MANAGER = "verl/workers/rollout/llm_server.py"


def _load_functions(path, names, owner=None, **namespace):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    body = tree.body if owner is None else next(n for n in tree.body if getattr(n, "name", None) == owner).body
    nodes = [n for n in body if getattr(n, "name", None) in names]
    assert len(nodes) == len(names)
    for node in nodes:
        node.decorator_list = []
    # Postpone annotations, as the optional runtime types are not imported here.
    module = ast.Module(body=[*ast.parse("from __future__ import annotations").body, *nodes], type_ignores=[])
    scope = {"asyncio": asyncio, "uuid": uuid, "logger": logging.getLogger(__name__), **namespace}
    exec(compile(ast.fix_missing_locations(module), str(ROOT / path), "exec"), scope)
    return types.SimpleNamespace(**{name: scope[name] for name in names})


def _remote(function):
    async def invoke(*args, **kwargs):
        return function(*args, **kwargs)

    return types.SimpleNamespace(remote=invoke)


def _metrics_pd_replica(rank, device, backend):
    methods = _load_functions(
        REPLICA,
        ["launch_servers", "get_metrics_server_endpoints", "_build_kv_transfer_config", "_format_server_address"],
        "vLLMPDReplica",
        get_device_name=lambda: device,
        is_valid_ipv6_address=lambda host: ip_address(host).version == 6,
    )
    spawned = []

    def spawn(**kwargs):
        port = 8000 + rank * 10 + len(spawned)
        server = types.SimpleNamespace(
            launch_server=types.SimpleNamespace(remote=AsyncMock()),
            set_pd_peer=types.SimpleNamespace(remote=AsyncMock()),
            get_server_address=_remote(lambda: ("::1", port)),
        )
        spawned.append((kwargs, server))
        return server

    replica = types.SimpleNamespace(
        replica_rank=rank,
        world_size=8,
        _prefill_tp=4,
        _decode_tp=2,
        _n_decode=2,
        name_suffix="",
        workers=[
            types.SimpleNamespace(__ray_call__=_remote(lambda fn, i=i: ("node", str(i), "::1"))) for i in range(8)
        ],
        config=types.SimpleNamespace(
            disaggregation=types.SimpleNamespace(transfer_backend=backend, mooncake_protocol="tcp")
        ),
        _decode_servers=[],
        _prefill_server_addresses=[],
        _decode_server_addresses=[],
        _is_ascend_platform=lambda: device == "npu",
        _collect_cuda_devices=lambda infos: ",".join(info[1] for info in infos),
        _reserve_pd_ports=Mock(return_value=(39000, 39010, [39020, 39030], [], False)),
        _build_kv_transfer_config=methods._build_kv_transfer_config,
        _format_server_address=methods._format_server_address,
        _spawn_pd_server=spawn,
    )
    replica.launch_servers = types.MethodType(methods.launch_servers, replica)
    replica.get_metrics_server_endpoints = types.MethodType(methods.get_metrics_server_endpoints, replica)
    return replica, spawned


@pytest.mark.parametrize(
    "device,backend,connector",
    [
        ("cuda", "nixl", "NixlConnector"),
        ("cuda", "mooncake", "MooncakeConnector"),
        ("npu", "mooncake", "MooncakeConnectorV1"),
    ],
)
def test_launch_discovers_every_pd_metrics_endpoint_without_changing_request_entry(device, backend, connector):
    replica, spawned = _metrics_pd_replica(3, device, backend)
    asyncio.run(replica.launch_servers())

    assert [address for address, _ in replica.get_metrics_server_endpoints()] == [
        "[::1]:8030",
        "[::1]:8031",
        "[::1]:8032",
    ]
    assert replica._server_address == "[::1]:8030"
    assert replica._server_handle is spawned[0][1]
    assert [options["kv_transfer_config"]["kv_connector"] for options, _ in spawned] == [connector] * 3
    spawned[0][1].set_pd_peer.remote.assert_awaited_once()
    assert spawned[0][1].set_pd_peer.remote.await_args.kwargs["decode_peers"] == [server for _, server in spawned[1:]]


@pytest.mark.parametrize(
    "name,device,backend",
    [
        ("vllm", "cuda", "nixl"),
        ("vllm", "cuda", "mooncake"),
        ("vllm", "npu", "mooncake"),
        ("vllm", "cuda", None),
        ("sglang", "cuda", None),
        ("trtllm", "cuda", None),
    ],
)
@pytest.mark.parametrize("prometheus,insight", [(False, False), (True, False), (False, True), (True, True)])
def test_manager_registers_metrics_separately_from_request_routing(name, device, backend, prometheus, insight):
    register = Mock()
    update_prometheus = Mock()
    initialize = _load_functions(
        MANAGER,
        ["_initialize_llm_servers"],
        "LLMServerManager",
        get_device_name=lambda: device,
        RLInsightLogger=types.SimpleNamespace(enabled=lambda: insight, register_rollout_metrics=register),
        update_prometheus_config=update_prometheus,
    )._initialize_llm_servers
    default_metrics = _load_functions(
        BASE_REPLICA, ["get_metrics_server_endpoints"], "RolloutReplica"
    ).get_metrics_server_endpoints

    def make_replica(replica_rank, **kwargs):
        if backend:
            replica, _ = _metrics_pd_replica(replica_rank, device, backend)

            async def init(*args):
                await replica.launch_servers()

        else:
            replica = types.SimpleNamespace(
                replica_rank=replica_rank,
                _server_address=f"[::1]:{8000 + replica_rank * 10}",
                _server_handle=object(),
            )
            replica.get_metrics_server_endpoints = types.MethodType(default_metrics, replica)
            init = AsyncMock()
        replica.init_hybrid = init
        replica.init_hybrid_colocated = init
        return replica

    config = types.SimpleNamespace(
        name=name,
        tensor_model_parallel_size=4,
        data_parallel_size=1,
        pipeline_model_parallel_size=1,
        n_gpus_per_node=8,
        disaggregation=types.SimpleNamespace(
            enabled=bool(backend), decode_tensor_model_parallel_size=2, prefill_replicas=1, decode_replicas=2
        ),
        prometheus=types.SimpleNamespace(enable=prometheus),
        disable_log_stats=False,
    )
    manager = types.SimpleNamespace(
        start_rank=3,
        rollout_config=config,
        model_config=None,
        worker_group=types.SimpleNamespace(world_size=16 if backend else 8),
        rollout_resource_pool=object(),
        rollout_replica_class=make_replica,
    )
    asyncio.run(initialize(manager))

    assert manager.server_addresses == ["[::1]:8030", "[::1]:8040"]
    assert manager.server_handles == [replica._server_handle for replica in manager.rollout_replicas]
    if backend:
        expected_addresses = [f"[::1]:{8000 + rank * 10 + index}" for rank in (3, 4) for index in range(3)]
        expected_labels = [
            {"replica": f"{rank}-{suffix}", "pd_role": role, "pd_index": index}
            for rank in (3, 4)
            for suffix, role, index in (("prefill", "prefill", 0), ("decode0", "decode", 0), ("decode1", "decode", 1))
        ]
    else:
        expected_addresses = ["[::1]:8030", "[::1]:8040"]
        expected_labels = [{"replica": 3}, {"replica": 4}]
    if insight:
        register.assert_called_once_with(expected_addresses, name, labels=expected_labels)
    else:
        register.assert_not_called()
    if prometheus:
        update_prometheus.assert_called_once_with(config.prometheus, expected_addresses, name)
    else:
        update_prometheus.assert_not_called()


@pytest.mark.parametrize(
    "fields",
    [
        [],
        ["prompt_ids"],
        ["prompt_ids", "reservation_id"],
        ["sampling_params", "priority", "session_id"],
        ["image_data", "video_data", "audio_data", "mm_processor_kwargs"],
    ],
)
def test_decode_router_only_receives_declared_fields(fields):
    functions = _load_functions(SERVER, ["_select_decode_peer_for_dispatch"], "vLLMHttpServer")
    calls = []
    peer = object()

    def acquire(request_id, **kwargs):
        calls.append((request_id, kwargs))
        assert set(kwargs) == set(fields)
        return "d", peer

    stub = types.SimpleNamespace(
        _pd_router=types.SimpleNamespace(acquire_server=_remote(acquire)),
        _pd_router_acquire_fields=fields,
        _pd_decode_peer_ids=["d"],
        _pd_decode_peers=[peer],
        _select_decode_peer=Mock(side_effect=AssertionError("unexpected fallback")),
    )
    generation = {
        "sampling_params": {"max_tokens": 20},
        "priority": 7,
        "session_id": "session",
        "image_data": ["image"],
        "video_data": ["video"],
        "audio_data": ["audio"],
        "mm_processor_kwargs": {"fps": 1},
    }
    result = asyncio.run(
        functions._select_decode_peer_for_dispatch(stub, "session", [1, 2], reservation_id="turn", **generation)
    )
    assert result == (peer, "d")
    assert calls == [
        (
            "session",
            {
                key: value
                for key, value in {**generation, "prompt_ids": [1, 2], "reservation_id": "turn"}.items()
                if key in fields
            },
        )
    ]
    stub._select_decode_peer.assert_not_called()


@pytest.mark.parametrize("outcome", ["success", "prefill_error", "decode_error", "cancelled"])
@pytest.mark.parametrize("release_fields", [[], ["request_id"], ["request_id", "reservation_id"]])
def test_dispatch_releases_reservation_on_every_exit(outcome, release_fields):
    helper = _load_functions(SERVER, ["_build_mooncake_decode_kv_params"])
    methods = _load_functions(
        SERVER,
        ["_pd_dispatch", "_build_pd_decode_kv_params", "_release_pd_decode_peer"],
        "vLLMHttpServer",
        _build_mooncake_decode_kv_params=helper._build_mooncake_decode_kv_params,
    )
    release = AsyncMock()
    decode = AsyncMock(return_value="generated")
    if outcome == "decode_error":
        decode.side_effect = RuntimeError("decode failed")
    peer = types.SimpleNamespace(generate=types.SimpleNamespace(remote=decode))
    prefill_params = {"remote_block_ids": [[1, 2]], "remote_engine_id": "original"}
    generate = AsyncMock(return_value=types.SimpleNamespace(extra_fields={"kv_transfer_params": prefill_params}))
    if outcome == "prefill_error":
        generate.side_effect = RuntimeError("prefill failed")
    elif outcome == "cancelled":
        generate.side_effect = asyncio.CancelledError()
    select = AsyncMock(return_value=(peer, "d"))
    stub = types.SimpleNamespace(
        _select_decode_peer_for_dispatch=select,
        _pd_router=types.SimpleNamespace(release_server=types.SimpleNamespace(remote=release)),
        _pd_router_release_fields=release_fields,
        _disaggregation_kv_transfer_config={"kv_connector": "MooncakeConnectorV1"},
        _pd_prefill_engine_id="fallback",
        _pd_prefill_side_channel_port=12345,
        generate=generate,
    )
    stub._build_pd_decode_kv_params = types.MethodType(methods._build_pd_decode_kv_params, stub)
    stub._release_pd_decode_peer = types.MethodType(methods._release_pd_decode_peer, stub)

    async def run():
        return await methods._pd_dispatch(stub, [1, 2], {"max_tokens": 20}, "turn", session_id="session")

    if outcome == "success":
        assert asyncio.run(run()) == "generated"
        params = decode.call_args.kwargs["kv_transfer_params"]
        assert params["remote_block_ids"] == [[1, 2]]
        assert params["remote_engine_id"] == "original"
        assert params["do_remote_prefill"] is True
        assert prefill_params == {"remote_block_ids": [[1, 2]], "remote_engine_id": "original"}
    else:
        exception = asyncio.CancelledError if outcome == "cancelled" else RuntimeError
        with pytest.raises(exception):
            asyncio.run(run())
    assert select.await_count == 1
    assert select.await_args.args == ("session", [1, 2])
    assert select.await_args.kwargs["sampling_params"] == {"max_tokens": 20}
    assert generate.await_args.args[1]["max_tokens"] == 1
    reservation_id = select.await_args.kwargs["reservation_id"]
    assert len(reservation_id) == 32
    identity_fields = {"request_id": "session", "reservation_id": reservation_id}
    release.assert_awaited_once_with(
        server_id="d", **{key: value for key, value in identity_fields.items() if key in release_fields}
    )


@pytest.mark.parametrize("repeat_cancel", [None, "acquire", "release"])
def test_pending_acquire_cancellation_releases_after_reply(repeat_cancel):
    methods = _load_functions(
        SERVER, ["_pd_dispatch", "_select_decode_peer_for_dispatch", "_release_pd_decode_peer"], "vLLMHttpServer"
    )

    async def run():
        recorded, reply, releasing, release_reply = [asyncio.Event() for _ in range(4)]
        reservations = {}
        remote_tasks = []
        peer = object()

        async def acquire(request_id, prompt_ids, reservation_id):
            assert prompt_ids == [1]
            reservations[reservation_id] = "d"
            recorded.set()
            await reply.wait()
            return "d", peer

        def remote_acquire(**kwargs):
            task = asyncio.create_task(acquire(**kwargs))
            remote_tasks.append(task)
            # Like a Ray ObjectRef: cancelling the waiter does not stop remote work.
            return asyncio.shield(task)

        async def release(server_id, request_id, reservation_id):
            releasing.set()
            await release_reply.wait()
            assert request_id == "session"
            assert reservations.pop(reservation_id) == server_id

        release_call = AsyncMock(side_effect=release)
        stub = types.SimpleNamespace(
            _pd_router=types.SimpleNamespace(
                acquire_server=types.SimpleNamespace(remote=remote_acquire),
                release_server=types.SimpleNamespace(remote=release_call),
            ),
            _pd_router_acquire_fields=["prompt_ids", "reservation_id"],
            _pd_router_release_fields=["request_id", "reservation_id"],
            _pd_decode_peer_ids=["d"],
            _select_decode_peer=Mock(side_effect=AssertionError("unexpected fallback")),
            generate=AsyncMock(side_effect=AssertionError("cancelled request must not run prefill")),
        )
        for name in ("_select_decode_peer_for_dispatch", "_release_pd_decode_peer"):
            setattr(stub, name, types.MethodType(getattr(methods, name), stub))
        dispatch = asyncio.create_task(
            methods._pd_dispatch(stub, [1], {"max_tokens": 20}, "turn", session_id="session")
        )
        await asyncio.wait_for(recorded.wait(), 1)
        dispatch.cancel("request cancelled")
        # Let the first cancellation enter cleanup before issuing another.
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        if repeat_cancel == "acquire":
            dispatch.cancel()
        assert not remote_tasks[0].cancelled()
        reply.set()
        await asyncio.wait_for(releasing.wait(), 1)
        if repeat_cancel == "release":
            dispatch.cancel()
        release_reply.set()
        with pytest.raises(asyncio.CancelledError, match="request cancelled"):
            await asyncio.wait_for(dispatch, 1)
        await asyncio.gather(*remote_tasks)
        assert reservations == {}
        assert release_call.await_count == 1
        stub.generate.assert_not_awaited()

    asyncio.run(run())


@pytest.mark.parametrize("blocks", [None, [], (), [[]], [[], []], ([], [])])
def test_mooncake_without_transferable_blocks_computes_locally(blocks):
    build = _load_functions(SERVER, ["_build_mooncake_decode_kv_params"])._build_mooncake_decode_kv_params
    original = {"remote_block_ids": blocks, "do_remote_decode": True, "do_remote_prefill": True}
    params = build(original, remote_engine_id="p", remote_bootstrap_addr="addr", transfer_id="t")
    assert params["do_remote_prefill"] is False
    assert params["do_remote_decode"] is False
    assert original["do_remote_decode"] is True


@pytest.mark.parametrize("blocks", [[0], [[0]], [[], [0]], ([], [1, 2])])
def test_mooncake_preserves_block_zero_and_mixed_groups(blocks):
    build = _load_functions(SERVER, ["_build_mooncake_decode_kv_params"])._build_mooncake_decode_kv_params
    params = build({"remote_block_ids": blocks}, remote_engine_id="p", remote_bootstrap_addr="addr", transfer_id="t")
    assert params["do_remote_prefill"] is True
    assert params["do_remote_decode"] is False
    assert params["remote_block_ids"] == blocks


def test_pd_resume_resumes_all_engines_before_any_admission():
    resume = _load_functions(REPLICA, ["resume_generation"], "vLLMPDReplica").resume_generation
    events = []

    def server(name):
        return types.SimpleNamespace(
            resume_engine_generation=_remote(lambda: events.append(("resume", name))),
            open_submission_gate=_remote(lambda: events.append(("open", name))),
        )

    asyncio.run(resume(types.SimpleNamespace(servers=[server("p"), server("d0"), server("d1")])))
    assert events == [(action, name) for action in ("resume", "open") for name in ("p", "d0", "d1")]


def test_pd_resume_failure_keeps_all_gates_closed():
    resume = _load_functions(REPLICA, ["resume_generation"], "vLLMPDReplica").resume_generation
    open_gate = AsyncMock()
    server = types.SimpleNamespace(
        resume_engine_generation=types.SimpleNamespace(remote=AsyncMock(side_effect=RuntimeError("resume failed"))),
        open_submission_gate=types.SimpleNamespace(remote=open_gate),
    )
    with pytest.raises(RuntimeError, match="resume failed"):
        asyncio.run(resume(types.SimpleNamespace(servers=[server])))
    open_gate.assert_not_awaited()


@pytest.mark.parametrize("role", ["prefill", "decode"])
def test_ascend_connector_config_carries_asymmetric_tp(role):
    build = _load_functions(REPLICA, ["_build_kv_transfer_config"], "vLLMPDReplica", get_device_name=lambda: "npu")
    cfg = build._build_kv_transfer_config(
        role, role, "mooncake", use_ascend_mooncake_v1=True, kv_port=39000, prefill_tp=4, decode_tp=2
    )
    assert cfg["kv_connector"] == "MooncakeConnectorV1"
    assert cfg["kv_role"] == {"prefill": "kv_producer", "decode": "kv_consumer"}[role]
    assert cfg["engine_id"] == role
    assert cfg["kv_buffer_device"] == "npu"
    assert cfg["kv_port"] == 39000
    assert cfg["kv_connector_extra_config"] == {
        "prefill": {"dp_size": 1, "tp_size": 4},
        "decode": {"dp_size": 1, "tp_size": 2},
    }
    with pytest.raises(ValueError, match="requires kv_port"):
        build._build_kv_transfer_config("decode", "d", "mooncake", use_ascend_mooncake_v1=True)


def test_port_ranges_are_contiguous_and_disjoint():
    reserve = _load_functions(
        REPLICA, ["_reserve_port_range"], "vLLMPDReplica", socket=socket, is_valid_ipv6_address=lambda _: False
    )._reserve_port_range
    sockets = []
    try:
        base1, first = reserve("127.0.0.1", 2)
        sockets.extend(first)
        base2, second = reserve("127.0.0.1", 3)
        sockets.extend(second)
        assert {base1, base1 + 1}.isdisjoint(range(base2, base2 + 3))
        assert [s.getsockname()[1] for s in first] == [base1, base1 + 1]
        assert [s.getsockname()[1] for s in second] == list(range(base2, base2 + 3))
    finally:
        for sock in sockets:
            sock.close()
    with pytest.raises(ValueError, match="positive"):
        reserve("127.0.0.1", 0)


@pytest.mark.parametrize("ascend", [False, True])
def test_sleep_uses_worker_lifecycle_without_peer_disconnect_rpc(ascend):
    sleep = _load_functions(SERVER, ["sleep"], "vLLMReplica").sleep
    events = []

    def server(name):
        # A strict interface: never invent collective_rpc or a disconnect API.
        return types.SimpleNamespace(
            wait_for_requests_to_drain=_remote(lambda: events.append(("drain", name))),
            sleep=_remote(lambda: events.append(("sleep", name))),
        )

    prefill, decode = server("p"), server("d")
    stub = types.SimpleNamespace(
        servers=[prefill, decode],
        _is_ascend_platform=lambda: ascend,
    )
    # Exercise any inherited/overridden hook still present in the source. This
    # regression must fail on the old PD hook, rather than omitting it in a stub.
    for path, owner in ((SERVER, "vLLMReplica"), (REPLICA, "vLLMPDReplica")):
        tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
        owner_node = next(node for node in tree.body if getattr(node, "name", None) == owner)
        if any(getattr(node, "name", None) == "_before_server_sleep" for node in owner_node.body):
            hook = _load_functions(path, ["_before_server_sleep"], owner)._before_server_sleep
            stub._before_server_sleep = types.MethodType(hook, stub)
    asyncio.run(sleep(stub))
    assert events == [
        ("drain", "p"),
        ("sleep", "p"),
        ("sleep", "d"),
    ]


def test_sleep_propagates_worker_failure():
    sleep = _load_functions(SERVER, ["sleep"], "vLLMReplica").sleep
    failure = AsyncMock(side_effect=RuntimeError("worker memory unregistration failed"))
    server = types.SimpleNamespace(
        wait_for_requests_to_drain=types.SimpleNamespace(remote=AsyncMock()),
        sleep=types.SimpleNamespace(remote=failure),
    )
    with pytest.raises(RuntimeError, match="worker memory unregistration failed"):
        asyncio.run(sleep(types.SimpleNamespace(servers=[server])))
    failure.assert_awaited_once()


def test_no_router_keeps_round_robin():
    select = _load_functions(SERVER, ["_select_decode_peer_for_dispatch"], "vLLMHttpServer")
    peer = object()
    stub = types.SimpleNamespace(_pd_router=None, _select_decode_peer=Mock(return_value=peer))
    assert asyncio.run(select._select_decode_peer_for_dispatch(stub, "session", [1])) == (peer, None)
    stub._select_decode_peer.assert_called_once()


@pytest.mark.parametrize("connector,remote", [("MooncakeConnector", True), ("MooncakeConnectorV1", False)])
def test_empty_prefill_respects_connector_specific_semantics(connector, remote):
    helper = _load_functions(SERVER, ["_build_mooncake_decode_kv_params"])
    build = _load_functions(
        SERVER,
        ["_build_pd_decode_kv_params"],
        "vLLMHttpServer",
        _build_mooncake_decode_kv_params=helper._build_mooncake_decode_kv_params,
    )._build_pd_decode_kv_params
    stub = types.SimpleNamespace(_pd_prefill_engine_id="p", _pd_prefill_side_channel_port=12345)
    params = build(stub, types.SimpleNamespace(extra_fields={}), connector, "transfer", "request")
    assert params["do_remote_prefill"] is remote
    assert params["do_remote_decode"] is False


def test_unified_resume_only_resumes_coordinator_then_opens_all_gates():
    resume = _load_functions(SERVER, ["resume_generation"], "vLLMReplica").resume_generation
    events = []
    servers = [
        types.SimpleNamespace(
            resume_engine_generation=_remote(lambda i=i: events.append(("resume", i))),
            open_submission_gate=_remote(lambda i=i: events.append(("open", i))),
        )
        for i in range(2)
    ]
    asyncio.run(resume(types.SimpleNamespace(servers=servers)))
    assert events == [("resume", 0), ("open", 0), ("open", 1)]


@pytest.mark.parametrize("contiguous", [False, True])
def test_partial_port_reservation_failure_closes_acquired_sockets(contiguous):
    sock = Mock()
    allocate = Mock(side_effect=[(39000, [sock] if contiguous else sock), OSError("port unavailable")])
    reserve = _load_functions(
        REPLICA,
        ["_reserve_pd_ports"],
        "vLLMPDReplica",
        get_free_port=allocate,
    )._reserve_pd_ports
    stub = types.SimpleNamespace(_prefill_tp=2, _decode_tp=2, world_size=4, _n_decode=1, _reserve_port_range=allocate)
    with pytest.raises(OSError, match="port unavailable"):
        reserve(stub, "127.0.0.1", contiguous)
    sock.close.assert_called_once()


def test_metrics_endpoints_cover_both_roles():
    metrics = _load_functions(REPLICA, ["get_metrics_server_endpoints"], "vLLMPDReplica").get_metrics_server_endpoints
    stub = types.SimpleNamespace(
        replica_rank=0,
        _prefill_servers=[object()],
        _decode_servers=[object(), object()],
        _prefill_server_addresses=["p:1"],
        _decode_server_addresses=["d0:1", "d1:1"],
    )
    assert metrics(stub) == [
        ("p:1", {"replica": "0-prefill", "pd_role": "prefill", "pd_index": 0}),
        ("d0:1", {"replica": "0-decode0", "pd_role": "decode", "pd_index": 0}),
        ("d1:1", {"replica": "0-decode1", "pd_role": "decode", "pd_index": 1}),
    ]


def test_original_router_selection_test_matches_current_production_methods():
    """Run the existing integration test body without importing the hardware stack."""
    methods = _load_functions(
        SERVER,
        ["_select_decode_peer", "_select_decode_peer_for_dispatch", "_release_pd_decode_peer"],
        "vLLMHttpServer",
    )
    test_path = "tests/workers/rollout/test_vllm_pd_disaggregation_on_cpu.py"
    stub_class = _load_functions(test_path, ["_DispatchStub"])._DispatchStub
    # Replace only the deferred vLLM import inside the stub's fallback method.
    stub_class._select_decode_peer = methods._select_decode_peer
    test = _load_functions(
        test_path,
        ["test_router_selection_acquires_and_releases_peer"],
        _DispatchStub=stub_class,
        _import_http_server=lambda: methods,
    ).test_router_selection_acquires_and_releases_peer
    test()


def test_concurrent_same_session_dispatches_have_distinct_matching_reservations():
    methods = _load_functions(
        SERVER,
        ["_pd_dispatch", "_select_decode_peer_for_dispatch", "_release_pd_decode_peer", "_build_pd_decode_kv_params"],
        "vLLMHttpServer",
    )
    acquire = AsyncMock()
    release = AsyncMock()
    decode = AsyncMock(return_value="generated")
    peer = types.SimpleNamespace(generate=types.SimpleNamespace(remote=decode))
    acquire.return_value = ("d", peer)
    stub = types.SimpleNamespace(
        _pd_router=types.SimpleNamespace(
            acquire_server=types.SimpleNamespace(remote=acquire),
            release_server=types.SimpleNamespace(remote=release),
        ),
        _pd_router_acquire_fields=["prompt_ids", "reservation_id"],
        _pd_router_release_fields=["request_id", "reservation_id"],
        _pd_decode_peer_ids=["d"],
        _disaggregation_kv_transfer_config={"kv_connector": "NixlConnector"},
        generate=AsyncMock(return_value=types.SimpleNamespace(extra_fields={"kv_transfer_params": {}})),
    )
    for name in ("_select_decode_peer_for_dispatch", "_release_pd_decode_peer", "_build_pd_decode_kv_params"):
        setattr(stub, name, types.MethodType(getattr(methods, name), stub))

    async def run():
        return await asyncio.gather(
            *[methods._pd_dispatch(stub, [1], {}, "same-request-id", session_id="session") for _ in range(2)]
        )

    assert asyncio.run(run()) == ["generated", "generated"]
    acquired = [call.kwargs for call in acquire.await_args_list]
    released = [call.kwargs for call in release.await_args_list]
    assert all(call["request_id"] == "session" for call in acquired + released)
    acquire_ids = {call["reservation_id"] for call in acquired}
    assert len(acquire_ids) == 2
    assert {call["reservation_id"] for call in released} == acquire_ids
