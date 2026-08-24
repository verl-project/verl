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
"""Accelerator-free unit tests for vLLM PD config and replica plumbing.

Covers Phase 1 of the verl-vllm-pd-disagg series:
  * ``DisaggregationConfig`` validation rules
  * ``RolloutConfig`` post_init coercion + name-vs-disagg.enabled gate
  * ``get_rollout_replica_class("vllm", disaggregation_enabled=True)`` resolves to ``vLLMPDReplica``
  * ``vLLMPDReplica`` config validation paths
  * GPU and Ascend connector selection/config shapes

Phase 2 (NIXL 1P:1D smoke) and Phase 3 (1P:ND scaling) add live Ray-actor and
vLLM-engine tests behind ``@pytest.mark.skipif(not CUDA_AVAILABLE)``.
"""

from __future__ import annotations

from collections import Counter
from unittest.mock import patch

import pytest

from verl.workers.config import DisaggregationConfig, RolloutConfig, RoutingPolicyConfig
from verl.workers.rollout.vllm_rollout.pd_routing import DecodePeerSelector

# ---------------------------------------------------------------------------
# DisaggregationConfig validation
# ---------------------------------------------------------------------------


def test_disaggregation_defaults_disabled_and_valid():
    cfg = DisaggregationConfig()
    assert cfg.enabled is False
    assert cfg.prefill_replicas == 1
    assert cfg.decode_replicas == 1
    assert cfg.transfer_backend == "nixl"
    assert cfg.bootstrap_port is None
    assert cfg.ib_device is None


def test_disaggregation_enabled_nixl_accepted():
    cfg = DisaggregationConfig(enabled=True, transfer_backend="nixl")
    assert cfg.enabled is True
    assert cfg.transfer_backend == "nixl"


def test_disaggregation_decode_policy_uses_vllm_router_defaults():
    cfg = DisaggregationConfig(enabled=True)
    assert cfg.decode_policy.type == "round_robin"
    assert cfg.decode_policy.load_check_interval_secs == 5
    assert cfg.decode_policy.virtual_nodes == 160
    assert cfg.decode_policy.cache_threshold == 0.3
    assert cfg.decode_policy.balance_abs_threshold == 64
    assert cfg.decode_policy.balance_rel_threshold == 1.5
    assert cfg.decode_policy.eviction_interval_secs == 120
    assert cfg.decode_policy.max_tree_size == 2**26

    for policy in ("random", "power_of_two", "consistent_hash", "rendezvous_hash", "cache_aware"):
        assert DisaggregationConfig(
            enabled=True,
            decode_policy=RoutingPolicyConfig(type=policy),
        ).decode_policy.type == policy


@pytest.mark.parametrize("backend", ["nixl", "mooncake", "ascend", "mori", "fake"])
def test_disaggregation_all_known_backends_pass_config_validation(backend):
    cfg = DisaggregationConfig(enabled=True, transfer_backend=backend)
    assert cfg.transfer_backend == backend


def test_disaggregation_unknown_backend_rejected():
    with pytest.raises(ValueError, match="transfer_backend"):
        DisaggregationConfig(enabled=True, transfer_backend="bogus")


def test_disaggregation_zero_replicas_rejected():
    with pytest.raises(ValueError, match="prefill_replicas"):
        DisaggregationConfig(enabled=True, prefill_replicas=0)
    with pytest.raises(ValueError, match="decode_replicas"):
        DisaggregationConfig(enabled=True, decode_replicas=0)


def test_disaggregation_bad_bootstrap_port_rejected():
    with pytest.raises(ValueError, match="bootstrap_port"):
        DisaggregationConfig(enabled=True, bootstrap_port=70000)


def test_disaggregation_disabled_skips_validation():
    """When enabled=False, even bad values are tolerated (matches YAML defaults)."""
    cfg = DisaggregationConfig(enabled=False, transfer_backend="bogus", bootstrap_port=70000)
    assert cfg.enabled is False


def test_effective_decode_tp_defaults_to_prefill_tp():
    cfg = DisaggregationConfig(enabled=True)
    assert cfg.effective_decode_tp(prefill_tp=4) == 4


def test_effective_decode_tp_respects_override():
    cfg = DisaggregationConfig(enabled=True, decode_tensor_model_parallel_size=2)
    assert cfg.effective_decode_tp(prefill_tp=4) == 2


# ---------------------------------------------------------------------------
# RolloutConfig wiring
# ---------------------------------------------------------------------------


def test_rollout_config_vllm_pd_enabled_ok():
    cfg = RolloutConfig(
        name="vllm",
        disaggregation=DisaggregationConfig(enabled=True),
    )
    assert cfg.disaggregation.enabled is True
    assert cfg.disaggregation.transfer_backend == "nixl"


@pytest.mark.parametrize("name", ["sglang", "vllm"])
def test_rollout_config_pd_enabled_passes_name_gate(name):
    """RolloutConfig accepts ``sglang`` / ``vllm`` when
    ``disaggregation.enabled=True`` — the canonical post-#6117 pattern."""
    cfg = RolloutConfig(
        name=name,
        disaggregation=DisaggregationConfig(enabled=True),
    )
    assert cfg.disaggregation.enabled is True


@pytest.mark.parametrize("name", ["trtllm"])
def test_rollout_config_non_pd_name_rejects_pd(name):
    with pytest.raises(ValueError, match="disaggregation.enabled=True"):
        RolloutConfig(
            name=name,
            disaggregation=DisaggregationConfig(enabled=True),
        )


def test_rollout_config_disabled_pd_works_on_any_backend():
    for name in ("sglang", "vllm", "trtllm"):
        cfg = RolloutConfig(name=name)
        assert cfg.disaggregation.enabled is False


def test_rollout_config_accepts_dict_disaggregation():
    """Hydra/OmegaConf hands the field as a plain dict; post_init must coerce."""
    cfg = RolloutConfig(
        name="vllm",
        disaggregation={"enabled": True, "decode_replicas": 3},
    )
    assert isinstance(cfg.disaggregation, DisaggregationConfig)
    assert cfg.disaggregation.decode_replicas == 3


def test_rollout_config_accepts_dictconfig_disaggregation():
    from omegaconf import OmegaConf

    dc = OmegaConf.create({"enabled": True, "transfer_backend": "nixl", "decode_replicas": 2})
    cfg = RolloutConfig(name="vllm", disaggregation=dc)
    assert isinstance(cfg.disaggregation, DisaggregationConfig)
    assert cfg.disaggregation.decode_replicas == 2


# ---------------------------------------------------------------------------
# Replica dispatcher
# ---------------------------------------------------------------------------


def test_dispatcher_vllm_with_flag_returns_pd_replica():
    """``get_rollout_replica_class('vllm', disaggregation_enabled=True)``
    resolves to ``vLLMPDReplica``; the same name without the flag resolves to
    the colocated ``vLLMReplica``. Mirrors the SGLang dispatch from PR #6117."""
    # Both branches import the vLLM engine modules eagerly, and
    # verl.third_party.vllm raises (not ImportError) when neither vllm nor
    # sglang is installed, as in the CPU-only CI env.
    pytest.importorskip("vllm")
    from verl.workers.rollout.replica import get_rollout_replica_class

    plain_cls = get_rollout_replica_class("vllm", disaggregation_enabled=False)
    pd_cls = get_rollout_replica_class("vllm", disaggregation_enabled=True)
    assert plain_cls.__name__ == "vLLMReplica"
    assert pd_cls.__name__ == "vLLMPDReplica"
    assert issubclass(pd_cls, plain_cls)


# ---------------------------------------------------------------------------
# vLLMPDReplica._build_kv_transfer_config
# ---------------------------------------------------------------------------


def _build_kv_cfg(
    *,
    role: str,
    engine_id: str = "test-eid",
    transfer_backend: str = "nixl",
    mooncake_protocol=None,
    use_ascend_mooncake_v1: bool = False,
    kv_port=None,
    prefill_tp=None,
    decode_tp=None,
):
    # Lazy import: only meaningful when vllm-rollout deps are importable.
    pytest.importorskip("vllm")
    from verl.workers.rollout.vllm_rollout.vllm_pd_replica import vLLMPDReplica

    return vLLMPDReplica._build_kv_transfer_config(
        role=role,
        engine_id=engine_id,
        transfer_backend=transfer_backend,
        mooncake_protocol=mooncake_protocol,
        use_ascend_mooncake_v1=use_ascend_mooncake_v1,
        kv_port=kv_port,
        prefill_tp=prefill_tp,
        decode_tp=decode_tp,
    )


@pytest.mark.parametrize(
    "npu_available,expected",
    [(False, False), (True, True)],
)
def test_ascend_detection_follows_verl_npu_availability(npu_available, expected):
    pytest.importorskip("vllm")
    from verl.workers.rollout.vllm_rollout import vllm_pd_replica as mod

    with patch.object(mod, "is_torch_npu_available", return_value=npu_available) as is_npu:
        assert mod.vLLMPDReplica._is_ascend_platform() is expected
    is_npu.assert_called_once_with(check_device=False)


def test_build_kv_transfer_config_prefill_role_maps_to_kv_producer():
    cfg = _build_kv_cfg(role="prefill", engine_id="e0")
    assert cfg["kv_connector"] == "NixlConnector"
    assert cfg["kv_role"] == "kv_producer"
    assert cfg["engine_id"] == "e0"
    assert cfg["kv_buffer_device"] == "cuda"
    assert "kv_connector_extra_config" not in cfg


def test_build_kv_transfer_config_decode_role_maps_to_kv_consumer():
    cfg = _build_kv_cfg(role="decode", engine_id="e1")
    assert cfg["kv_role"] == "kv_consumer"
    assert cfg["engine_id"] == "e1"


@pytest.mark.parametrize("role,expected_role", [("prefill", "kv_producer"), ("decode", "kv_consumer")])
def test_build_kv_transfer_config_mooncake_backend(role, expected_role):
    """Under transfer_backend='mooncake' the connector name must be
    'MooncakeConnector' on both prefill and decode legs. Earlier rev had a
    bug where decode silently fell back to NixlConnector because the
    decode call site forgot to pass transfer_backend through."""
    cfg = _build_kv_cfg(role=role, transfer_backend="mooncake")
    assert cfg["kv_connector"] == "MooncakeConnector"
    assert cfg["kv_role"] == expected_role


def test_build_kv_transfer_config_mooncake_protocol_pinned():
    """``mooncake_protocol`` lands in ``kv_connector_extra_config`` so vLLM's
    MooncakeConnector pins the requested transport instead of taking the
    upstream default ``"rdma"`` (which silently falls back to TCP on hosts
    without an RDMA NIC)."""
    cfg = _build_kv_cfg(role="prefill", transfer_backend="mooncake", mooncake_protocol="nvlink")
    assert cfg["kv_connector_extra_config"] == {"mooncake_protocol": "nvlink"}


def test_build_kv_transfer_config_mooncake_protocol_omitted_when_none():
    """When ``mooncake_protocol`` is not supplied, the field is absent so vLLM
    keeps its own default. Useful for the NIXL path where the field is moot."""
    cfg = _build_kv_cfg(role="prefill", transfer_backend="mooncake", mooncake_protocol=None)
    assert "kv_connector_extra_config" not in cfg


def test_build_kv_transfer_config_mooncake_protocol_ignored_for_nixl():
    """``mooncake_protocol`` is meaningful only for the Mooncake connector;
    the NIXL connector ignores it (UCX picks transport on its own)."""
    cfg = _build_kv_cfg(role="prefill", transfer_backend="nixl", mooncake_protocol="nvlink")
    assert "kv_connector_extra_config" not in cfg


@pytest.mark.parametrize("role,expected_role", [("prefill", "kv_producer"), ("decode", "kv_consumer")])
def test_build_kv_transfer_config_ascend_mooncake_v1(role, expected_role):
    cfg = _build_kv_cfg(
        role=role,
        transfer_backend="mooncake",
        use_ascend_mooncake_v1=True,
        kv_port=19001,
        prefill_tp=4,
        decode_tp=2,
    )
    assert cfg["kv_connector"] == "MooncakeConnectorV1"
    assert cfg["kv_role"] == expected_role
    assert cfg["kv_port"] == 19001
    assert cfg["kv_connector_extra_config"] == {
        "prefill": {"dp_size": 1, "tp_size": 4},
        "decode": {"dp_size": 1, "tp_size": 2},
    }


@pytest.mark.parametrize("protocol", ["nvlink", "local", "rdma", "tcp"])
def test_disagg_config_accepts_known_mooncake_protocols(protocol):
    DisaggregationConfig(enabled=True, transfer_backend="mooncake", mooncake_protocol=protocol)


def test_disagg_config_rejects_unknown_mooncake_protocol():
    with pytest.raises(ValueError, match="mooncake_protocol"):
        DisaggregationConfig(enabled=True, transfer_backend="mooncake", mooncake_protocol="bogus")


def test_disagg_config_default_mooncake_protocol_is_nvlink():
    """Default to ``nvlink`` so single-node PD on H100/H200 actually uses the
    NVLink fabric. vLLM's upstream Mooncake default is ``rdma``, which on a
    no-RDMA rack silently falls back to TCP loopback."""
    assert DisaggregationConfig().mooncake_protocol == "nvlink"


# ---------------------------------------------------------------------------
# vLLMPDReplica.__init__ validation paths
#
# We bypass the parent ``vLLMReplica.__init__`` (which calls ``ray.remote``)
# because that's a Phase-2 integration concern; here we only probe the new
# validation block that vLLMPDReplica adds on top.
# ---------------------------------------------------------------------------


def _make_pd_config(**overrides) -> RolloutConfig:
    disagg = DisaggregationConfig(
        enabled=overrides.pop("enabled", True),
        prefill_replicas=overrides.pop("prefill_replicas", 1),
        decode_replicas=overrides.pop("decode_replicas", 1),
        transfer_backend=overrides.pop("transfer_backend", "nixl"),
        decode_tensor_model_parallel_size=overrides.pop("decode_tensor_model_parallel_size", None),
        prefill_gpu_memory_utilization=overrides.pop("prefill_gpu_memory_utilization", None),
        decode_gpu_memory_utilization=overrides.pop("decode_gpu_memory_utilization", None),
        prefill_engine_kwargs=overrides.pop("prefill_engine_kwargs", {}),
        decode_engine_kwargs=overrides.pop("decode_engine_kwargs", {}),
        ib_device=overrides.pop("ib_device", None),
    )
    return RolloutConfig(
        name="vllm",
        tensor_model_parallel_size=overrides.pop("tensor_model_parallel_size", 1),
        data_parallel_size=overrides.pop("data_parallel_size", 1),
        disaggregation=disagg,
        **overrides,
    )


@pytest.fixture
def patched_replica_cls():
    """Patch parent ``vLLMReplica.__init__`` so we can exercise the PD validation
    in isolation (no Ray, no vLLMHttpServer remote-class construction)."""
    pytest.importorskip("vllm")
    from verl.workers.rollout.vllm_rollout import vllm_pd_replica as mod

    def _stub_super_init(self, replica_rank, config, model_config, gpus_per_node, *_a, **_kw):
        # Faithful subset of RolloutReplica + vLLMReplica state needed by
        # vLLMPDReplica.__init__ to run its validation block.
        from verl.utils.config import omega_conf_to_dataclass

        self.replica_rank = replica_rank
        self.config = omega_conf_to_dataclass(config)
        self.model_config = model_config
        self.gpus_per_node = gpus_per_node
        self.world_size = self.config.tensor_model_parallel_size * self.config.data_parallel_size
        self.gpus_per_replica_node = min(gpus_per_node, self.world_size)
        self.nnodes = max(1, self.world_size // self.gpus_per_replica_node)
        self.name_suffix = ""
        self.workers = []
        self.servers = []
        self.server_class = None  # ray.remote(vLLMHttpServer) is Phase 2 territory
        self._server_address = None
        self._server_handle = None

    with patch.object(mod.vLLMReplica, "__init__", _stub_super_init):
        yield mod.vLLMPDReplica


def test_pd_replica_init_happy_path_1p1d(patched_replica_cls):
    cfg = _make_pd_config()
    replica = patched_replica_cls(
        replica_rank=0,
        config=cfg,
        model_config=None,
        gpus_per_node=8,
    )
    assert replica._n_prefill == 1
    assert replica._n_decode == 1
    assert replica._prefill_tp == 1
    assert replica._decode_tp == 1
    assert replica.world_size == 2  # 1 prefill + 1 decode
    assert replica._prefill_servers == [] and replica._decode_servers == []


def test_pd_replica_init_happy_path_1p3d(patched_replica_cls):
    cfg = _make_pd_config(decode_replicas=3)
    replica = patched_replica_cls(
        replica_rank=0,
        config=cfg,
        model_config=None,
        gpus_per_node=8,
    )
    assert replica._n_decode == 3
    assert replica.world_size == 4


def test_pd_replica_init_accepts_tp_gt_1(patched_replica_cls):
    """TP>1 per PD engine works as long as the worker_group footprint adds
    up. Each worker independently computes its socket key as
    base + self.local_rank (see utils.py::_get_zmq_handle), so the single
    per-actor VERL_ZMQ_BASE_TRAINER_RANK env covers all TP-ranks of that
    actor."""
    cfg = _make_pd_config(
        tensor_model_parallel_size=2,
        decode_tensor_model_parallel_size=2,
        decode_replicas=2,
    )
    replica = patched_replica_cls(replica_rank=0, config=cfg, model_config=None, gpus_per_node=8)
    # prefill_tp=2 + decode_replicas=2 * decode_tp=2 = 6 GPUs
    assert replica.world_size == 6
    assert replica._prefill_tp == 2
    assert replica._decode_tp == 2


def test_pd_replica_init_accepts_mooncake(patched_replica_cls):
    """Mooncake is now an accepted transfer_backend (requires
    `mooncake-transfer-engine` pip pkg at runtime in the engine actor)."""
    cfg = _make_pd_config(transfer_backend="mooncake")
    replica = patched_replica_cls(replica_rank=0, config=cfg, model_config=None, gpus_per_node=8)
    assert replica.config.disaggregation.transfer_backend == "mooncake"


def test_pd_replica_init_rejects_unsupported_backend(patched_replica_cls):
    cfg = _make_pd_config(transfer_backend="mori")
    with pytest.raises(NotImplementedError, match="transfer_backend in"):
        patched_replica_cls(replica_rank=0, config=cfg, model_config=None, gpus_per_node=8)


def test_pd_replica_init_rejects_multi_prefill(patched_replica_cls):
    cfg = _make_pd_config(prefill_replicas=2)
    with pytest.raises(NotImplementedError, match="prefill_replicas=1"):
        patched_replica_cls(replica_rank=0, config=cfg, model_config=None, gpus_per_node=8)


def test_pd_replica_init_rejects_oversized_world(patched_replica_cls):
    cfg = _make_pd_config(decode_replicas=8)  # 1 + 8 = 9 GPUs needed
    with pytest.raises(NotImplementedError, match="single-node only"):
        patched_replica_cls(replica_rank=0, config=cfg, model_config=None, gpus_per_node=8)


def test_pd_role_engine_kwargs_deep_merge_without_cross_role_mutation(patched_replica_cls):
    cfg = _make_pd_config(
        prefill_gpu_memory_utilization=0.85,
        decode_gpu_memory_utilization=0.7,
        prefill_engine_kwargs={"max_num_batched_tokens": 65536, "max_num_seqs": None},
        decode_engine_kwargs={"max_num_batched_tokens": 2048, "max_num_seqs": 256},
        engine_kwargs={"vllm": {"max_num_seqs": 512}},
    )
    replica = patched_replica_cls(replica_rank=0, config=cfg, model_config=None, gpus_per_node=8)

    prefill = replica._build_pd_role_config("prefill", tp=1)
    decode = replica._build_pd_role_config("decode", tp=1)

    assert prefill.gpu_memory_utilization == 0.85
    assert decode.gpu_memory_utilization == 0.7
    assert prefill.engine_kwargs["vllm"] == {"max_num_batched_tokens": 65536, "max_num_seqs": 512}
    assert decode.engine_kwargs["vllm"] == {"max_num_batched_tokens": 2048, "max_num_seqs": 256}
    assert cfg.engine_kwargs["vllm"] == {"max_num_seqs": 512}


def test_pd_replica_exposes_prefill_request_and_all_metrics_endpoints(patched_replica_cls):
    cfg = _make_pd_config(decode_replicas=2)
    replica = patched_replica_cls(replica_rank=0, config=cfg, model_config=None, gpus_per_node=8)
    prefill_handles = [object()]
    replica._prefill_servers = prefill_handles
    replica._decode_servers = [object(), object()]
    replica._prefill_server_addresses = ["p0:8000"]
    replica._decode_server_addresses = ["d0:8000", "d1:8000"]

    assert replica.get_request_server_endpoints() == [
        ("p0:8000", prefill_handles[0]),
    ]
    assert [labels["pd_role"] for _, labels in replica.get_metrics_server_endpoints()] == [
        "prefill",
        "decode",
        "decode",
    ]


class _AwaitableRemoteMethod:
    def __init__(self):
        self.calls = []

    def remote(self, *args, **kwargs):
        self.calls.append((args, kwargs))

        async def _done():
            return None

        return _done()


class _SleepServerStub:
    def __init__(self):
        self.wait_for_requests_to_drain = _AwaitableRemoteMethod()
        self.collective_rpc = _AwaitableRemoteMethod()
        self.sleep = _AwaitableRemoteMethod()


@pytest.mark.asyncio
@pytest.mark.parametrize("npu_available,expected_disconnects", [(False, 0), (True, 2)])
async def test_pd_sleep_keeps_gpu_path_native_and_disconnects_ascend(
    patched_replica_cls, npu_available, expected_disconnects
):
    replica = object.__new__(patched_replica_cls)
    prefill = _SleepServerStub()
    decode = _SleepServerStub()
    replica._prefill_servers = [prefill]
    replica._prefill_server_addresses = ["p0:8000"]
    replica.servers = [prefill, decode]

    with patch(
        "verl.workers.rollout.vllm_rollout.vllm_pd_replica.is_torch_npu_available",
        return_value=npu_available,
    ):
        await replica.sleep()

    assert len(prefill.wait_for_requests_to_drain.calls) == 1
    assert len(prefill.sleep.calls) == 1
    assert len(decode.sleep.calls) == 1
    assert len(prefill.collective_rpc.calls) + len(decode.collective_rpc.calls) == expected_disconnects
    for server in (prefill, decode):
        for _, kwargs in server.collective_rpc.calls:
            assert kwargs == {"method": "disconnect_kv_transfer_peers"}


def test_pd_replica_init_rejects_dp_gt_1(patched_replica_cls):
    cfg = _make_pd_config(data_parallel_size=2)
    with pytest.raises(NotImplementedError, match="data_parallel_size=1"):
        patched_replica_cls(replica_rank=0, config=cfg, model_config=None, gpus_per_node=8)


def test_pd_replica_init_requires_disaggregation_enabled(patched_replica_cls):
    cfg = _make_pd_config(enabled=False)
    # RolloutConfig validation lets `name='vllm'` + `enabled=False` through
    # (that's the colocated path). But constructing vLLMPDReplica directly
    # without disagg.enabled is invalid — class-level assertion guards.
    with pytest.raises(AssertionError, match="disaggregation.enabled=True"):
        patched_replica_cls(replica_rank=0, config=cfg, model_config=None, gpus_per_node=8)


# ---------------------------------------------------------------------------
# vLLMHttpServer PD dispatch (Phase 2)
#
# We only need a minimal stub that carries the dispatch state — neither the
# real engine nor a Ray cluster. ``_select_decode_peer`` and ``_pd_dispatch``
# are unbound methods on vLLMHttpServer; the stub supplies the few attributes
# they read.
# ---------------------------------------------------------------------------


class _DispatchStub:
    """Minimal vLLMHttpServer-like instance for unbound-method tests."""

    def __init__(self, decode_peers, role="prefill", connector="NixlConnector", policy="round_robin"):
        self._disaggregation_role = role
        self._pd_decode_peers = list(decode_peers)
        peer_ids = [f"http://decode-{index}:8000" for index in range(len(self._pd_decode_peers))]
        self._pd_decode_selector = DecodePeerSelector(RoutingPolicyConfig(type=policy), peer_ids)
        # Used by _pd_dispatch to branch between NIXL (read kv_transfer_params
        # back from prefill) and Mooncake (construct it locally from prefill
        # engine_id + bootstrap addr).
        self._disaggregation_kv_transfer_config = {"kv_connector": connector}
        self._pd_prefill_engine_id = "eid-prefill"
        self._pd_prefill_side_channel_host = "127.0.0.1"
        self._pd_prefill_side_channel_port = 5559

    def _select_decode_peer(self, routing_key, prompt_ids):
        # Borrow the real implementation to keep the stub aligned with
        # production rotation semantics — _pd_dispatch's behavior must not
        # depend on the test's peer-selection policy.
        from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer

        return vLLMHttpServer._select_decode_peer(self, routing_key, prompt_ids)


def _import_http_server():
    pytest.importorskip("vllm")
    from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer

    return vLLMHttpServer


def test_select_decode_peer_round_robin_cycles():
    server_cls = _import_http_server()
    stub = _DispatchStub(decode_peers=["peer0", "peer1", "peer2"])
    picks = [server_cls._select_decode_peer(stub, "session", [1])[1] for _ in range(7)]
    assert picks == ["peer0", "peer1", "peer2", "peer0", "peer1", "peer2", "peer0"]
    assert stub._pd_decode_selector.pending_requests == [3, 2, 2]


def test_select_decode_peer_single_peer_returns_same():
    server_cls = _import_http_server()
    stub = _DispatchStub(decode_peers=["only_peer"])
    assert all(server_cls._select_decode_peer(stub, "session", [1])[1] == "only_peer" for _ in range(5))


def test_select_decode_peer_distribution_balanced_at_32_with_3_peers():
    """Lock in the empirical Phase-3a observation: 32 sequential dispatches
    across 3 decode peers land in an exact (floor, ceil) split of (10, 11),
    spread of 1. Catches accidental drift away from strict cycle (e.g., a
    future change that randomizes for cache-locality but loses balance)."""
    server_cls = _import_http_server()
    stub = _DispatchStub(decode_peers=["peer0", "peer1", "peer2"])
    counts = Counter(server_cls._select_decode_peer(stub, "session", [1])[1] for _ in range(32))
    assert sorted(counts.values()) == [10, 11, 11], (
        f"expected (10, 11, 11) hit counts under strict round-robin, got {dict(counts)}"
    )
    assert max(counts.values()) - min(counts.values()) <= 1


@pytest.mark.asyncio
async def test_pd_dispatch_routes_prefill_leg_then_decode_peer():
    """End-to-end shape of ``_pd_dispatch``: prefill leg sets max_tokens=1 +
    do_remote_decode, decode leg gets the prefill's kv_transfer_params."""
    from unittest.mock import MagicMock

    server_cls = _import_http_server()

    # Mock decode peer (Ray actor handle): peer.generate.remote(...) returns
    # a TokenOutput-shaped result.
    decode_peer = MagicMock()
    expected_decode_token_ids = [10, 20, 30]
    decode_peer.generate.remote = MagicMock(return_value=_make_awaitable_token_output(expected_decode_token_ids))

    # Stub server.generate: returns a TokenOutput with kv_transfer_params in
    # extra_fields, simulating the response NixlConnector populates.
    server_decode_kv = {
        "do_remote_prefill": True,
        "remote_engine_id": "eid-prefill",
        "remote_block_ids": [0, 1, 2],
        "remote_host": "10.0.0.1",
        "remote_port": 5559,
        "remote_request_id": "req-foo_P",
        "tp_size": 1,
    }
    captured_prefill_calls = []

    async def fake_generate(prompt_ids, sampling_params, request_id, **kw):
        captured_prefill_calls.append({"sampling_params": dict(sampling_params), "request_id": request_id, **kw})
        from verl.workers.rollout.replica import TokenOutput

        return TokenOutput(
            token_ids=[42],
            stop_reason="completed",
            extra_fields={"kv_transfer_params": server_decode_kv},
        )

    stub = _DispatchStub(decode_peers=[decode_peer])
    stub.generate = fake_generate

    result = await server_cls._pd_dispatch(
        stub,
        prompt_ids=[1, 2, 3],
        sampling_params={"max_tokens": 64, "temperature": 0.0},
        request_id="req-foo",
    )

    # Prefill leg was called once with max_tokens=1 + do_remote_decode params.
    assert len(captured_prefill_calls) == 1
    pcall = captured_prefill_calls[0]
    assert pcall["request_id"] == "req-foo_P"
    assert pcall["sampling_params"]["max_tokens"] == 1
    # `transfer_id` is added unconditionally (Mooncake requires it; NIXL
    # ignores it). Check the other two flags but allow the extra field.
    assert pcall["kv_transfer_params"]["do_remote_decode"] is True
    assert pcall["kv_transfer_params"]["do_remote_prefill"] is False
    assert "transfer_id" in pcall["kv_transfer_params"]

    # Decode peer was called with full sampling_params + the prefill's kv_transfer_params.
    decode_peer.generate.remote.assert_called_once()
    dkw = decode_peer.generate.remote.call_args
    # generate(prompt_ids, sampling_params, request_id, **kw); first three are positional.
    assert dkw.args[0] == [1, 2, 3]
    assert dkw.args[1]["max_tokens"] == 64  # NOT clamped to 1 on decode leg
    assert dkw.args[2] == "req-foo_D"
    assert dkw.kwargs["kv_transfer_params"] == server_decode_kv
    assert dkw.kwargs["priority"] == 0

    assert result.token_ids == expected_decode_token_ids
    assert stub._pd_decode_selector.pending_requests == [0]


@pytest.mark.asyncio
async def test_pd_dispatch_mooncake_constructs_decode_kv_params_locally():
    """Under Mooncake, prefill's request_finished returns (_, None) — no
    kv_transfer_params come back. _pd_dispatch must instead construct decode
    kv_transfer_params from the prefill state set by set_pd_peer (engine_id +
    bootstrap addr), preserving transfer_id across the two legs."""
    from unittest.mock import MagicMock

    server_cls = _import_http_server()

    decode_peer = MagicMock()
    decode_peer.generate.remote = MagicMock(return_value=_make_awaitable_token_output([7]))
    captured_prefill = []

    async def fake_generate(prompt_ids, sampling_params, request_id, **kw):
        captured_prefill.append(kw["kv_transfer_params"])
        from verl.workers.rollout.replica import TokenOutput

        # Mooncake's prefill response carries no kv_transfer_params.
        return TokenOutput(token_ids=[42], stop_reason="completed", extra_fields={})

    stub = _DispatchStub(decode_peers=[decode_peer], connector="MooncakeConnector")
    stub.generate = fake_generate

    await server_cls._pd_dispatch(
        stub,
        prompt_ids=[1, 2],
        sampling_params={"max_tokens": 16, "temperature": 0.0},
        request_id="req-mc",
    )

    assert len(captured_prefill) == 1
    pkv = captured_prefill[0]
    assert pkv["do_remote_decode"] is True
    transfer_id = pkv["transfer_id"]

    decode_peer.generate.remote.assert_called_once()
    dkw = decode_peer.generate.remote.call_args
    dkv = dkw.kwargs["kv_transfer_params"]
    assert dkv["do_remote_prefill"] is True
    assert dkv["do_remote_decode"] is False
    assert dkv["remote_engine_id"] == stub._pd_prefill_engine_id
    assert dkv["remote_bootstrap_addr"] == f"http://127.0.0.1:{stub._pd_prefill_side_channel_port}"
    # transfer_id must match across legs so prefill and decode rendezvous.
    assert dkv["transfer_id"] == transfer_id


@pytest.mark.asyncio
async def test_pd_dispatch_raises_when_prefill_returns_no_kv_params():
    """Sanity: if NixlConnector silently produced no kv_transfer_params, fail
    fast rather than handing an empty dict to the decode peer."""
    server_cls = _import_http_server()
    from verl.workers.rollout.replica import TokenOutput

    async def empty_prefill(prompt_ids, sampling_params, request_id, **kw):
        return TokenOutput(token_ids=[0], stop_reason="completed", extra_fields={})

    stub = _DispatchStub(decode_peers=["peer0"])
    stub.generate = empty_prefill

    with pytest.raises(RuntimeError, match="no kv_transfer_params"):
        await server_cls._pd_dispatch(
            stub,
            prompt_ids=[1],
            sampling_params={"max_tokens": 8},
            request_id="req-bare",
        )
    assert stub._pd_decode_selector.pending_requests == [0]


@pytest.mark.asyncio
async def test_pd_dispatch_releases_reservation_when_decode_is_cancelled():
    import asyncio
    from unittest.mock import MagicMock

    from verl.workers.rollout.replica import TokenOutput

    server_cls = _import_http_server()
    decode_peer = MagicMock()

    async def cancelled_decode():
        raise asyncio.CancelledError

    decode_peer.generate.remote = MagicMock(return_value=cancelled_decode())

    async def fake_generate(prompt_ids, sampling_params, request_id, **kw):
        return TokenOutput(token_ids=[0], stop_reason="completed", extra_fields={"kv_transfer_params": {}})

    stub = _DispatchStub(decode_peers=[decode_peer])
    stub.generate = fake_generate

    with pytest.raises(asyncio.CancelledError):
        await server_cls._pd_dispatch(
            stub,
            prompt_ids=[1],
            sampling_params={"max_tokens": 8},
            request_id="req-cancelled",
        )
    assert stub._pd_decode_selector.pending_requests == [0]


def _make_awaitable_token_output(token_ids):
    """Wrap a TokenOutput in an awaitable so the test can ``await`` the
    decode_peer.generate.remote(...) MagicMock return value."""
    from verl.workers.rollout.replica import TokenOutput

    out = TokenOutput(token_ids=token_ids, stop_reason="completed")

    async def _coro():
        return out

    return _coro()
