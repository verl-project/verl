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
"""GPU-free unit tests for PD disaggregation config + replica plumbing."""

from __future__ import annotations

import pytest

from verl.workers.config import DisaggregationConfig, RolloutConfig


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
    assert cfg.enabled is True and cfg.transfer_backend == "nixl"


def test_disaggregation_enabled_mooncake_accepted():
    cfg = DisaggregationConfig(enabled=True, transfer_backend="mooncake", ib_device="mlx5_roce0")
    assert cfg.transfer_backend == "mooncake"


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


def test_rollout_config_sglang_with_disagg_enabled_ok():
    cfg = RolloutConfig(
        name="sglang",
        disaggregation=DisaggregationConfig(enabled=True),
    )
    assert cfg.disaggregation.enabled is True


@pytest.mark.parametrize("name", ["vllm", "trtllm"])
def test_rollout_config_non_sglang_rejects_pd(name):
    with pytest.raises(ValueError, match="disaggregation.enabled=True"):
        RolloutConfig(
            name=name,
            disaggregation=DisaggregationConfig(enabled=True),
        )


def test_rollout_config_disabled_pd_works_on_any_backend():
    for name in ("sglang", "vllm", "trtllm"):
        cfg = RolloutConfig(name=name)
        assert cfg.disaggregation.enabled is False


def test_effective_decode_tp_defaults_to_prefill_tp():
    cfg = DisaggregationConfig(enabled=True)
    assert cfg.effective_decode_tp(prefill_tp=4) == 4


def test_effective_decode_tp_respects_override():
    cfg = DisaggregationConfig(enabled=True, decode_tensor_model_parallel_size=2)
    assert cfg.effective_decode_tp(prefill_tp=4) == 2


def test_registry_no_pd_aliases():
    """``sglang_pd``/``vllm_pd`` were dropped: PD is selected via the disaggregation flag."""
    from verl.workers.rollout.replica import RolloutReplicaRegistry

    assert "sglang" in RolloutReplicaRegistry._registry
    assert "vllm" in RolloutReplicaRegistry._registry
    assert "sglang_pd" not in RolloutReplicaRegistry._registry
    assert "vllm_pd" not in RolloutReplicaRegistry._registry


def _sglang_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("sglang") is not None


@pytest.mark.skipif(not _sglang_available(), reason="sglang not installed")
def test_dispatch_sglang_returns_pd_replica_when_flag_set():
    from verl.workers.rollout.replica import get_rollout_replica_class

    plain_cls = get_rollout_replica_class("sglang", disaggregation_enabled=False)
    pd_cls = get_rollout_replica_class("sglang", disaggregation_enabled=True)
    assert plain_cls.__name__ == "SGLangReplica"
    assert pd_cls.__name__ == "SGLangPDReplica"
    assert issubclass(pd_cls, plain_cls)


def test_dispatch_non_sglang_with_flag_raises():
    from verl.workers.rollout.replica import get_rollout_replica_class

    with pytest.raises(NotImplementedError, match="PD disaggregation"):
        get_rollout_replica_class("vllm", disaggregation_enabled=True)


def _assign_pd_role(rollout_rank: int, prefill_tp: int, decode_replicas: int, decode_tp: int):
    """Mirror of ServerAdapter.__init__'s role-assignment block."""
    if rollout_rank < prefill_tp:
        return "prefill", 0, rollout_rank
    off = rollout_rank - prefill_tp
    if off < decode_replicas * decode_tp:
        return "decode", off // decode_tp, off % decode_tp
    return None, None, None


@pytest.mark.parametrize(
    "prefill_tp,decode_replicas,decode_tp,rollout_rank,expected",
    [
        (1, 3, 1, 0, ("prefill", 0, 0)),
        (1, 3, 1, 1, ("decode", 0, 0)),
        (1, 3, 1, 2, ("decode", 1, 0)),
        (1, 3, 1, 3, ("decode", 2, 0)),
        (1, 7, 1, 0, ("prefill", 0, 0)),
        (1, 7, 1, 7, ("decode", 6, 0)),
        (2, 3, 2, 0, ("prefill", 0, 0)),
        (2, 3, 2, 1, ("prefill", 0, 1)),
        (2, 3, 2, 2, ("decode", 0, 0)),
        (2, 3, 2, 3, ("decode", 0, 1)),
        (2, 3, 2, 6, ("decode", 2, 0)),
        (2, 3, 2, 7, ("decode", 2, 1)),
    ],
)
def test_pd_role_assignment(prefill_tp, decode_replicas, decode_tp, rollout_rank, expected):
    assert _assign_pd_role(rollout_rank, prefill_tp, decode_replicas, decode_tp) == expected


@pytest.mark.parametrize("prefill_tp,decode_replicas,decode_tp", [(1, 3, 1), (1, 7, 1), (2, 3, 2), (1, 1, 4)])
def test_pd_role_covers_every_rank_exactly_once(prefill_tp, decode_replicas, decode_tp):
    world = prefill_tp + decode_replicas * decode_tp
    seen: set[tuple[str, int, int]] = set()
    for rr in range(world):
        role, srv, tp_rank = _assign_pd_role(rr, prefill_tp, decode_replicas, decode_tp)
        assert role is not None, f"rollout_rank={rr} got no role"
        seen.add((role, srv, tp_rank))
    assert len(seen) == world, "each rank must map to a distinct (role, server_index, tp_local_rank) triple"
