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
import inspect
import json
import tempfile
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from omegaconf import OmegaConf
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from verl.workers.config.cache_pool import KVCachePoolConfig, KVCachePoolMasterConfig
from verl.workers.rollout.vllm_rollout.kv_cache_pool import (
    MooncakeMasterActor,
    MooncakeMasterProcess,
    build_pd_kv_transfer_config,
    build_mooncake_json,
    build_mooncake_master_cmd,
    build_pd_store_connector_config,
    build_non_pd_kv_transfer_config,
    materialize_mooncake_config,
    mooncake_json_path,
    p2p_connector_name,
    parse_size_to_bytes,
    probe_tcp,
    resolve_decode_tp,
    resolve_store_tp,
    setup_kv_cache_pool,
    validate_platform_cache_pool,
)


def test_mooncake_json_path():
    job_id = "job-1"
    assert mooncake_json_path(job_id) == str(Path(tempfile.gettempdir()) / f"verl_mooncake_{job_id}.json")


def test_parse_size_to_bytes():
    assert parse_size_to_bytes(1073741824) == 1073741824
    assert parse_size_to_bytes("1GB") == 2**30
    assert parse_size_to_bytes("1024MB") == 2**30
    assert parse_size_to_bytes("4GB") == 4 * 2**30


def test_resolve_decode_tp():
    assert resolve_decode_tp(4, None) == 4
    assert resolve_decode_tp(4, 2) == 2


def test_resolve_store_tp_lcm_of_p_and_d():
    extra = resolve_store_tp(
        prefill_tp=4, decode_tp=2, prefill_tps=[4], user_store_tp_size=None, enable_store_tp_lcm=None
    )
    assert extra == {"store_tp_size": 4}


def test_resolve_store_tp_user_override():
    extra = resolve_store_tp(
        prefill_tp=4, decode_tp=2, prefill_tps=[4], user_store_tp_size=8, enable_store_tp_lcm=None
    )
    assert extra == {"store_tp_size": 8}


def test_resolve_store_tp_multi_p_auto_lcm():
    extra = resolve_store_tp(
        prefill_tp=4, decode_tp=2, prefill_tps=[4, 2], user_store_tp_size=None, enable_store_tp_lcm=None
    )
    assert extra == {"enable_store_tp_lcm": True, "prefill_tp_sizes": [4, 2]}


def test_resolve_store_tp_multi_p_explicit_false():
    extra = resolve_store_tp(
        prefill_tp=4, decode_tp=2, prefill_tps=[4, 2], user_store_tp_size=None, enable_store_tp_lcm=False
    )
    assert extra == {"store_tp_size": 4}


def test_resolve_store_tp_user_size_disables_auto_lcm():
    extra = resolve_store_tp(
        prefill_tp=4, decode_tp=2, prefill_tps=[4, 2], user_store_tp_size=8, enable_store_tp_lcm=None
    )
    assert extra == {"store_tp_size": 8}


def test_resolve_store_tp_explicit_lcm_writes_prefill_tp_sizes():
    extra = resolve_store_tp(
        prefill_tp=4, decode_tp=2, prefill_tps=[4], user_store_tp_size=None, enable_store_tp_lcm=True
    )
    assert extra == {"enable_store_tp_lcm": True, "prefill_tp_sizes": [4]}


def test_resolve_store_tp_explicit_lcm_uses_user_prefill_tp_sizes():
    extra = resolve_store_tp(
        prefill_tp=4,
        decode_tp=2,
        prefill_tps=[4],
        user_store_tp_size=None,
        enable_store_tp_lcm=True,
        user_prefill_tp_sizes=[4, 2],
    )
    assert extra == {"enable_store_tp_lcm": True, "prefill_tp_sizes": [4, 2]}


def test_resolve_store_tp_rejects_store_tp_and_lcm():
    with pytest.raises(ValueError, match="mutually exclusive"):
        resolve_store_tp(
            prefill_tp=4, decode_tp=2, prefill_tps=[4], user_store_tp_size=8, enable_store_tp_lcm=True
        )


def test_resolve_store_tp_explicit_lcm_requires_prefill_tp_sizes():
    with pytest.raises(ValueError, match="prefill_tp_sizes"):
        resolve_store_tp(
            prefill_tp=4, decode_tp=2, prefill_tps=[], user_store_tp_size=None, enable_store_tp_lcm=True
        )


def test_resolve_store_tp_lcm_validates_decode_against_prefill_lcm():
    with pytest.raises(ValueError, match="store_tp_size"):
        resolve_store_tp(
            prefill_tp=4, decode_tp=8, prefill_tps=[4, 2], user_store_tp_size=None, enable_store_tp_lcm=None
        )


def test_gpu_json_keys():
    cfg = KVCachePoolConfig()
    data = build_mooncake_json(store=cfg.store, master_address="10.0.0.1:50051", is_npu=False)
    assert set(data) == {
        "mode",
        "metadata_server",
        "master_server_address",
        "global_segment_size",
        "local_buffer_size",
        "protocol",
        "device_name",
        "enable_offload",
        "tenant_id",
    }
    assert data["protocol"] == "rdma"
    assert data["enable_offload"] is False
    assert data["global_segment_size"] == "4GB"
    assert "ssd_offload_path" not in data


def test_gpu_json_enable_offload_hardcoded_false():
    cfg = KVCachePoolConfig(store={"enable_offload": True})
    data = build_mooncake_json(store=cfg.store, master_address="10.0.0.1:50051", is_npu=False)
    assert data["enable_offload"] is False


def test_npu_json_ssd_on():
    cfg = KVCachePoolConfig(store={"ssd_offload_path": "/nvme/mooncake_offload"})
    data = build_mooncake_json(store=cfg.store, master_address="10.0.0.1:50051", is_npu=True)
    assert data["protocol"] == "ascend"
    assert data["enable_ssd_offload"] is True
    assert data["ssd_offload_path"] == "/nvme/mooncake_offload"
    assert data["preferred_segment"] is False
    assert data["prefer_alloc_in_same_node"] is True
    assert "mode" not in data
    assert "local_buffer_size" not in data
    assert "enable_offload" not in data


def test_npu_json_ssd_off_omits_path():
    cfg = KVCachePoolConfig()
    data = build_mooncake_json(store=cfg.store, master_address="10.0.0.1:50051", is_npu=True)
    assert data["enable_ssd_offload"] is False
    assert "ssd_offload_path" not in data
    assert data["preferred_segment"] is False
    assert data["prefer_alloc_in_same_node"] is True


def test_platform_gpu_rejects_ssd_path():
    cfg = KVCachePoolConfig(store={"ssd_offload_path": "/nvme/x"})
    with pytest.raises(ValueError, match="ssd_offload_path"):
        validate_platform_cache_pool(cache_pool=cfg, is_npu=False)


def test_platform_gpu_rejects_enable_offload():
    cfg = KVCachePoolConfig(store={"enable_offload": True})
    with pytest.raises(NotImplementedError, match="offload"):
        validate_platform_cache_pool(cache_pool=cfg, is_npu=False)


def test_platform_npu_allows_save_decode_cache():
    cfg = KVCachePoolConfig(enabled=True, connector={"save_decode_cache": True})
    validate_platform_cache_pool(cache_pool=cfg, is_npu=True)


def test_platform_npu_rejects_unaligned_segment():
    cfg = KVCachePoolConfig(store={"global_segment_size": "512MB"})
    with pytest.raises(ValueError, match="1GB"):
        validate_platform_cache_pool(cache_pool=cfg, is_npu=True)


def test_platform_gpu_rejects_npu_store_field():
    cfg = KVCachePoolConfig(store={"preferred_segment": True})
    with pytest.raises(ValueError, match="preferred_segment"):
        validate_platform_cache_pool(cache_pool=cfg, is_npu=False)


def test_gpu_multiconnector_prefill():
    pool = KVCachePoolConfig(enabled=True)
    cfg = build_pd_kv_transfer_config(
        role="prefill",
        engine_id="e0",
        kv_buffer_device="cuda",
        transfer_backend="mooncake",
        mooncake_protocol="nvlink",
        use_ascend_mooncake=False,
        kv_port=None,
        prefill_tp=4,
        decode_tp=2,
        cache_pool=pool,
        prefill_tps=[4],
    )
    assert cfg["kv_connector"] == "MultiConnector"
    assert cfg["kv_role"] == "kv_producer"
    assert cfg["engine_id"] == "e0"
    p2p, store = cfg["kv_connector_extra_config"]["connectors"]
    assert p2p["kv_connector"] == "MooncakeConnector"
    assert p2p["kv_role"] == "kv_producer"
    assert p2p["kv_connector_extra_config"] == {"mooncake_protocol": "nvlink"}
    assert "engine_id" not in p2p
    assert store["kv_connector"] == "MooncakeStoreConnector"
    assert store["kv_role"] == "kv_both"
    assert store["kv_connector_extra_config"]["lookup_rpc_port"] == "e0"
    assert store["kv_connector_extra_config"]["store_tp_size"] == 4
    assert "enable_store_tp_lcm" not in store["kv_connector_extra_config"]
    assert "save_decode_cache" not in store["kv_connector_extra_config"]


def test_gpu_multiconnector_lcm_omits_store_tp_size():
    pool = KVCachePoolConfig(enabled=True)
    cfg = build_pd_kv_transfer_config(
        role="prefill",
        engine_id="e0",
        kv_buffer_device="cuda",
        transfer_backend="mooncake",
        mooncake_protocol="nvlink",
        use_ascend_mooncake=False,
        kv_port=None,
        prefill_tp=4,
        decode_tp=2,
        cache_pool=pool,
        prefill_tps=[4, 2],
    )
    extra = cfg["kv_connector_extra_config"]["connectors"][1]["kv_connector_extra_config"]
    assert extra["enable_store_tp_lcm"] is True
    assert extra["prefill_tp_sizes"] == [4, 2]
    assert "store_tp_size" not in extra


def test_gpu_multiconnector_decode_save_decode_cache():
    pool = KVCachePoolConfig(enabled=True, connector={"save_decode_cache": True})
    cfg = build_pd_kv_transfer_config(
        role="decode",
        engine_id="e1",
        kv_buffer_device="cuda",
        transfer_backend="mooncake",
        mooncake_protocol="nvlink",
        use_ascend_mooncake=False,
        kv_port=None,
        prefill_tp=4,
        decode_tp=2,
        cache_pool=pool,
        prefill_tps=[4],
    )
    store = cfg["kv_connector_extra_config"]["connectors"][1]
    assert store["kv_role"] == "kv_consumer"
    assert store["kv_connector_extra_config"]["save_decode_cache"] is True


def test_npu_multiconnector():
    pool = KVCachePoolConfig(enabled=True)
    cfg = build_pd_kv_transfer_config(
        role="prefill",
        engine_id="e0",
        kv_buffer_device="npu",
        transfer_backend="mooncake",
        mooncake_protocol=None,
        use_ascend_mooncake=True,
        kv_port=20001,
        prefill_tp=4,
        decode_tp=2,
        cache_pool=pool,
        prefill_tps=[4],
    )
    p2p, store = cfg["kv_connector_extra_config"]["connectors"]
    assert p2p["kv_connector"] == "MooncakeConnectorV2"
    assert p2p["kv_port"] == 20001
    assert "kv_port" not in cfg
    assert store["kv_connector"] == "AscendStoreConnector"
    assert store["kv_role"] == "kv_producer"
    assert store["kv_connector_extra_config"] == {"lookup_rpc_port": "e0"}
    assert "store_tp_size" not in store["kv_connector_extra_config"]


def test_npu_multiconnector_decode_save_decode_cache():
    pool = KVCachePoolConfig(enabled=True, connector={"save_decode_cache": True})
    cfg = build_pd_kv_transfer_config(
        role="decode",
        engine_id="e1",
        kv_buffer_device="npu",
        transfer_backend="mooncake",
        mooncake_protocol=None,
        use_ascend_mooncake=True,
        kv_port=20001,
        prefill_tp=4,
        decode_tp=2,
        cache_pool=pool,
        prefill_tps=[4],
    )
    store = cfg["kv_connector_extra_config"]["connectors"][1]
    assert store["kv_connector"] == "AscendStoreConnector"
    assert store["kv_role"] == "kv_consumer"
    assert store["kv_connector_extra_config"]["save_decode_cache"] is True


def _store_lookup(engine_id):
    pool = KVCachePoolConfig(enabled=True)
    return build_pd_store_connector_config(
        role="prefill",
        is_npu=True,
        cache_pool=pool,
        engine_id=engine_id,
        prefill_tp=4,
        decode_tp=2,
        prefill_tps=[4],
    )["kv_connector_extra_config"]["lookup_rpc_port"]


@pytest.mark.parametrize("value", ["eid-0", "abc"])
def test_build_store_writes_engine_id_as_lookup(value):
    assert _store_lookup(value) == value


@pytest.mark.parametrize("value", [None, "", True, 0, 19001])
def test_build_store_rejects_non_engine_id_lookup(value):
    with pytest.raises(ValueError, match="engine_id"):
        _store_lookup(value)


def test_unified_gpu_store_connector():
    pool = KVCachePoolConfig(enabled=True, connector={"store_tp_size": 8, "cache_prefix": "tenant"})
    cfg = build_non_pd_kv_transfer_config(
        is_npu=False,
        cache_pool=pool,
        engine_id="e0",
        tp=4,
        kv_buffer_device="cuda",
    )
    assert cfg["kv_connector"] == "MooncakeStoreConnector"
    assert cfg["kv_role"] == "kv_both"
    assert cfg["engine_id"] == "e0"
    assert cfg["kv_buffer_device"] == "cuda"
    extra = cfg["kv_connector_extra_config"]
    assert extra["lookup_rpc_port"] == "e0"
    assert extra["store_tp_size"] == 8
    assert extra["cache_prefix"] == "tenant"
    assert "connectors" not in extra
    assert "use_layerwise" not in extra
    assert "backend" not in extra
    assert "save_decode_cache" not in extra


def test_unified_gpu_store_tp_defaults_to_replica_tp():
    pool = KVCachePoolConfig(enabled=True)
    cfg = build_non_pd_kv_transfer_config(
        is_npu=False,
        cache_pool=pool,
        engine_id="e0",
        tp=4,
        kv_buffer_device="cuda",
    )
    assert cfg["kv_connector_extra_config"]["store_tp_size"] == 4


def test_unified_gpu_rejects_store_tp_not_divisible():
    pool = KVCachePoolConfig(enabled=True, connector={"store_tp_size": 6})
    with pytest.raises(ValueError, match="store_tp_size"):
        build_non_pd_kv_transfer_config(
            is_npu=False,
            cache_pool=pool,
            engine_id="e0",
            tp=4,
            kv_buffer_device="cuda",
        )


def test_unified_npu_store_connector():
    pool = KVCachePoolConfig(enabled=True, kv_load_failure_policy="recompute")
    cfg = build_non_pd_kv_transfer_config(
        is_npu=True,
        cache_pool=pool,
        engine_id="e0",
        tp=4,
        kv_buffer_device="npu",
    )
    assert cfg["kv_connector"] == "AscendStoreConnector"
    assert cfg["kv_role"] == "kv_both"
    assert cfg["kv_buffer_device"] == "npu"
    assert cfg["kv_load_failure_policy"] == "recompute"
    assert cfg["kv_connector_extra_config"] == {"lookup_rpc_port": "e0"}


def test_unified_rejects_empty_engine_id():
    pool = KVCachePoolConfig(enabled=True)
    with pytest.raises(ValueError, match="engine_id"):
        build_non_pd_kv_transfer_config(
            is_npu=False,
            cache_pool=pool,
            engine_id="",
            tp=4,
            kv_buffer_device="cuda",
        )


def test_pool_disabled_keeps_single_p2p():
    cfg = build_pd_kv_transfer_config(
        role="prefill",
        engine_id="e0",
        kv_buffer_device="cuda",
        transfer_backend="mooncake",
        mooncake_protocol="nvlink",
        use_ascend_mooncake=False,
        kv_port=None,
        prefill_tp=4,
        decode_tp=4,
        cache_pool=KVCachePoolConfig(enabled=False),
    )
    assert cfg["kv_connector"] == "MooncakeConnector"
    assert cfg["engine_id"] == "e0"
    assert cfg["kv_buffer_device"] == "cuda"
    assert "connectors" not in cfg.get("kv_connector_extra_config", {})


def test_pool_disabled_nixl_single_connector():
    cfg = build_pd_kv_transfer_config(
        role="prefill",
        engine_id="e0",
        kv_buffer_device="cuda",
        transfer_backend="nixl",
        mooncake_protocol=None,
        use_ascend_mooncake=False,
        kv_port=None,
        prefill_tp=4,
        decode_tp=4,
        cache_pool=None,
    )
    assert cfg["kv_connector"] == "NixlConnector"
    assert cfg["engine_id"] == "e0"
    assert cfg["kv_buffer_device"] == "cuda"
    assert "connectors" not in cfg


def test_kv_load_failure_policy_written():
    pool = KVCachePoolConfig(enabled=True, kv_load_failure_policy="recompute")
    cfg = build_pd_kv_transfer_config(
        role="prefill",
        engine_id="e0",
        kv_buffer_device="cuda",
        transfer_backend="mooncake",
        mooncake_protocol="nvlink",
        use_ascend_mooncake=False,
        kv_port=None,
        prefill_tp=4,
        decode_tp=4,
        cache_pool=pool,
        prefill_tps=[4],
    )
    assert cfg["kv_load_failure_policy"] == "recompute"
    assert "kv_load_failure_policy" not in cfg["kv_connector_extra_config"]["connectors"][0]


def test_kv_load_failure_policy_omitted_by_default():
    pool = KVCachePoolConfig(enabled=True)
    cfg = build_pd_kv_transfer_config(
        role="prefill",
        engine_id="e0",
        kv_buffer_device="cuda",
        transfer_backend="mooncake",
        mooncake_protocol="nvlink",
        use_ascend_mooncake=False,
        kv_port=None,
        prefill_tp=4,
        decode_tp=4,
        cache_pool=pool,
        prefill_tps=[4],
    )
    assert "kv_load_failure_policy" not in cfg


def test_p2p_connector_name_multiconnector():
    name = p2p_connector_name(
        {
            "kv_connector": "MultiConnector",
            "kv_connector_extra_config": {"connectors": [{"kv_connector": "MooncakeConnector"}]},
        }
    )
    assert name == "MooncakeConnector"


def test_p2p_connector_name_missing_child():
    with pytest.raises(RuntimeError, match="connectors"):
        p2p_connector_name({"kv_connector": "MultiConnector", "kv_connector_extra_config": {}})


def test_extra_config_overrides():
    pool = KVCachePoolConfig(enabled=True, extra_config={"cache_prefix": "expA", "load_async": False})
    cfg = build_pd_kv_transfer_config(
        role="prefill",
        engine_id="e0",
        kv_buffer_device="cuda",
        transfer_backend="mooncake",
        mooncake_protocol="nvlink",
        use_ascend_mooncake=False,
        kv_port=None,
        prefill_tp=4,
        decode_tp=4,
        cache_pool=pool,
        prefill_tps=[4],
    )
    extra = cfg["kv_connector_extra_config"]["connectors"][1]["kv_connector_extra_config"]
    assert extra["cache_prefix"] == "expA"
    assert extra["load_async"] is False
    assert extra["lookup_rpc_port"] == "e0"


def test_extra_config_cannot_override_lookup_rpc_port():
    pool = KVCachePoolConfig(enabled=True, extra_config={"lookup_rpc_port": 19001})
    cfg = build_pd_kv_transfer_config(
        role="prefill",
        engine_id="e0",
        kv_buffer_device="cuda",
        transfer_backend="mooncake",
        mooncake_protocol="nvlink",
        use_ascend_mooncake=False,
        kv_port=None,
        prefill_tp=4,
        decode_tp=4,
        cache_pool=pool,
        prefill_tps=[4],
    )
    extra = cfg["kv_connector_extra_config"]["connectors"][1]["kv_connector_extra_config"]
    assert extra["lookup_rpc_port"] == "e0"


def test_store_tp_rejects_not_divisible():
    with pytest.raises(ValueError, match="store_tp_size"):
        resolve_store_tp(
            prefill_tp=4, decode_tp=2, prefill_tps=[4], user_store_tp_size=3, enable_store_tp_lcm=None
        )


def test_store_tp_rejects_decode_tp():
    with pytest.raises(ValueError, match="store_tp_size"):
        resolve_store_tp(
            prefill_tp=4, decode_tp=3, prefill_tps=[4], user_store_tp_size=4, enable_store_tp_lcm=None
        )


def _make_actor(**attrs):
    actor = MooncakeMasterActor()
    for name, value in attrs.items():
        setattr(actor, name, value)
    return actor, MooncakeMasterActor


def _fake_popen(*, poll=None, output=""):
    fake = MagicMock()
    fake.poll.return_value = poll
    fake.stdout = MagicMock()
    fake.stdout.read.side_effect = [output, ""] if output else [""]
    return fake


def test_master_cmd_minimal():
    cmd = build_mooncake_master_cmd(
        master=KVCachePoolMasterConfig(), port=50051, rpc_address="10.0.0.1", enable_offload=False
    )
    assert cmd[:5] == ["mooncake_master", "--rpc_port", "50051", "--rpc_address", "10.0.0.1"]
    assert "--enable_metric_reporting" not in cmd
    assert "--enable_offload=true" not in cmd
    assert "--eviction_high_watermark_ratio" not in cmd
    assert "--eviction_ratio" not in cmd


def test_master_cmd_ssd_and_tenants():
    master = KVCachePoolMasterConfig(
        enable_multi_tenants=True,
        tenant_quota_connector_type="file",
        tenant_quota_connector_uri="/etc/mooncake/tenant_quotas.yaml",
        client_ttl=120,
        eviction_high_watermark_ratio=0.8,
        eviction_ratio=0.2,
    )
    cmd = build_mooncake_master_cmd(master=master, port=50088, rpc_address="10.0.0.1", enable_offload=True)
    assert "--enable_offload=true" in cmd
    assert "--enable_multi_tenants=true" in cmd
    assert "--client_ttl" in cmd
    assert cmd[cmd.index("--eviction_high_watermark_ratio") + 1] == "0.8"
    assert cmd[cmd.index("--eviction_ratio") + 1] == "0.2"


def test_process_stop_idempotent():
    proc = MooncakeMasterProcess()
    fake = MagicMock()
    fake.poll.return_value = None
    proc._proc = fake
    proc.stop()
    fake.terminate.assert_called_once()
    proc.stop()  # second call must not raise


def test_probe_tcp_retries_until_success():
    conn = MagicMock()
    with patch(
        "verl.workers.rollout.vllm_rollout.kv_cache_pool.socket.create_connection",
        side_effect=[OSError("fail"), conn],
    ) as create_conn:
        probe_tcp("127.0.0.1", 50051, timeout_s=5.0, interval_s=0.0)
    assert create_conn.call_count == 2


def test_probe_tcp_timeout():
    with patch(
        "verl.workers.rollout.vllm_rollout.kv_cache_pool.socket.create_connection",
        side_effect=OSError("fail"),
    ):
        with pytest.raises(RuntimeError):
            probe_tcp("127.0.0.1", 50051, timeout_s=0.01, interval_s=0.01)


def test_missing_binary_uses_gpu_hint(monkeypatch):
    monkeypatch.setattr(
        "verl.workers.rollout.vllm_rollout.kv_cache_pool.is_torch_npu_available",
        lambda check_device=False: False,
    )
    actor, _ = _make_actor()
    with (
        patch("ray.util.get_node_ip_address", return_value="127.0.0.1"),
        patch("subprocess.Popen", side_effect=FileNotFoundError("mooncake_master")),
    ):
        with pytest.raises(RuntimeError, match="mooncake-transfer-engine") as ei:
            actor.start({"port": 1}, enable_offload=False)
    assert "mooncake-transfer-engine-npu" not in str(ei.value)


def test_probe_timeout_includes_cmd_and_output():
    actor, _ = _make_actor()
    fake = _fake_popen(output="master stderr/stdout")
    with (
        patch("ray.util.get_node_ip_address", return_value="127.0.0.1"),
        patch("subprocess.Popen", return_value=fake),
        patch(
            "verl.workers.rollout.vllm_rollout.kv_cache_pool.probe_tcp",
            side_effect=RuntimeError("timeout"),
        ),
    ):
        with pytest.raises(RuntimeError, match="mooncake_master") as ei:
            actor.start({"port": 50051}, enable_offload=False)
    message = str(ei.value)
    assert "master stderr/stdout" in message
    fake.terminate.assert_called()


def test_dead_child_fails_even_if_probe_would_succeed():
    actor, _ = _make_actor()
    fake = _fake_popen(poll=1, output="Address already in use")
    with (
        patch("ray.util.get_node_ip_address", return_value="127.0.0.1"),
        patch("subprocess.Popen", return_value=fake) as popen,
        patch("verl.workers.rollout.vllm_rollout.kv_cache_pool.probe_tcp") as probe,
        patch("verl.workers.rollout.vllm_rollout.kv_cache_pool.get_free_port") as get_port,
    ):
        with pytest.raises(RuntimeError, match="mooncake_master") as ei:
            actor.start({"port": 50051}, enable_offload=False)
    probe.assert_not_called()
    get_port.assert_not_called()
    popen.assert_called_once()
    message = str(ei.value)
    assert "Address already in use" in message


def test_process_start_does_not_probe():
    proc = MooncakeMasterProcess()
    fake = _fake_popen()
    with patch("subprocess.Popen", return_value=fake) as popen, patch(
        "verl.workers.rollout.vllm_rollout.kv_cache_pool.probe_tcp"
    ) as probe:
        proc.start(["mooncake_master", "--port", "1"])
    popen.assert_called_once()
    probe.assert_not_called()
    proc.stop()


def test_process_start_starts_stdout_forward_thread():
    proc = MooncakeMasterProcess()
    fake = _fake_popen()
    with patch("subprocess.Popen", return_value=fake):
        proc.start(["mooncake_master", "--port", "1"])
    assert proc._reader is not None
    assert proc._reader.daemon
    proc.stop()


def test_process_forwards_master_output_to_stdout(capsys):
    proc = MooncakeMasterProcess()
    line = (
        "I1019 11:00:00.000000 1 rpc_service.cpp:41] Master Metrics: "
        "Mem Storage: 14.07 GB / 16.00 GB (87.9%) | Keys: 10\n"
        "unrelated startup line\n"
    )
    fake = _fake_popen(output=line)
    with patch("subprocess.Popen", return_value=fake):
        proc.start(["mooncake_master", "--port", "1"], log_output=True)
        assert proc._reader is not None
        proc._reader.join(timeout=2)
    captured = capsys.readouterr()
    assert (
        "I1019 11:00:00.000000 1 rpc_service.cpp:41] Master Metrics: "
        "Mem Storage: 14.07 GB / 16.00 GB (87.9%) | Keys: 10"
    ) in captured.out
    assert "unrelated startup line" in captured.out
    proc.stop()


def test_process_forwards_output_split_across_reads(capsys):
    proc = MooncakeMasterProcess()
    fake = MagicMock()
    fake.poll.return_value = None
    fake.stdout = MagicMock()
    fake.stdout.read.side_effect = [
        "noise\nI1019 rpc_service.cpp:41] Master Metrics: Mem Storage: 1 GB / ",
        "16 GB (6.2%) | Keys: 1\n",
        "",
    ]
    with patch("subprocess.Popen", return_value=fake):
        proc.start(["mooncake_master", "--port", "1"], log_output=True)
        assert proc._reader is not None
        proc._reader.join(timeout=2)
    captured = capsys.readouterr()
    assert "noise" in captured.out
    assert "I1019 rpc_service.cpp:41] Master Metrics: Mem Storage: 1 GB / 16 GB (6.2%) | Keys: 1" in captured.out
    proc.stop()


def test_process_keeps_master_output_for_errors_by_default(capsys):
    proc = MooncakeMasterProcess()
    line = "startup failed: address already in use\n"
    fake = _fake_popen(output=line)
    with patch("subprocess.Popen", return_value=fake):
        proc.start(["mooncake_master", "--port", "1"])
        assert proc._reader is not None
        proc._reader.join(timeout=2)
    captured = capsys.readouterr()
    assert captured.out == ""
    proc.stop()
    assert "address already in use" in proc._output


def test_actor_start_omits_metric_reporting_flag():
    actor, _ = _make_actor()
    fake = _fake_popen()
    with (
        patch("ray.util.get_node_ip_address", return_value="127.0.0.1"),
        patch("subprocess.Popen", return_value=fake) as popen,
        patch("verl.workers.rollout.vllm_rollout.kv_cache_pool.probe_tcp"),
    ):
        actor.start({"port": 50051, "log_master_output": True}, enable_offload=False)
    cmd = popen.call_args.args[0]
    assert "--enable_metric_reporting" not in cmd
    assert actor._proc._log_output is True
    actor.stop()


def test_actor_start_treats_port_zero_as_ephemeral():
    actor, _ = _make_actor()
    fake = _fake_popen()
    reserve = MagicMock()
    order = []
    reserve.close.side_effect = lambda: order.append("close")

    def popen(*args, **kwargs):
        order.append("popen")
        return fake

    with (
        patch("ray.util.get_node_ip_address", return_value="127.0.0.1"),
        patch("subprocess.Popen", side_effect=popen) as popen_mock,
        patch("verl.workers.rollout.vllm_rollout.kv_cache_pool.probe_tcp"),
        patch(
            "verl.workers.rollout.vllm_rollout.kv_cache_pool.get_free_port",
            return_value=(54321, reserve),
        ) as get_port,
    ):
        addr = actor.start({"port": 0}, enable_offload=False)
    get_port.assert_called_once_with("127.0.0.1", with_alive_sock=True)
    assert order == ["close", "popen"]
    cmd = popen_mock.call_args.args[0]
    assert cmd[cmd.index("--rpc_address") + 1] == "127.0.0.1"
    assert addr == "127.0.0.1:54321"
    actor.stop()


def test_free_port_retries_when_address_in_use():
    actor, _ = _make_actor()
    first = _fake_popen(poll=1, output="bind: Address already in use\n")
    second = _fake_popen()
    reserves = [MagicMock(), MagicMock()]
    with (
        patch("ray.util.get_node_ip_address", return_value="10.0.0.1"),
        patch("subprocess.Popen", side_effect=[first, second]),
        patch("verl.workers.rollout.vllm_rollout.kv_cache_pool.probe_tcp"),
        patch(
            "verl.workers.rollout.vllm_rollout.kv_cache_pool.get_free_port",
            side_effect=[(1111, reserves[0]), (2222, reserves[1])],
        ) as get_port,
    ):
        addr = actor.start({"port": None}, enable_offload=False)
    assert get_port.call_count == 2
    assert addr == "10.0.0.1:2222"
    reserves[0].close.assert_called_once()
    reserves[1].close.assert_called_once()
    actor.stop()


def test_free_port_does_not_retry_other_exit():
    actor, _ = _make_actor()
    fake = _fake_popen(poll=1, output="invalid flag\n")
    with (
        patch("ray.util.get_node_ip_address", return_value="10.0.0.1"),
        patch("subprocess.Popen", return_value=fake),
        patch("verl.workers.rollout.vllm_rollout.kv_cache_pool.probe_tcp"),
        patch(
            "verl.workers.rollout.vllm_rollout.kv_cache_pool.get_free_port",
            return_value=(1111, MagicMock()),
        ) as get_port,
    ):
        with pytest.raises(RuntimeError, match="invalid flag"):
            actor.start({"port": 0}, enable_offload=False)
    get_port.assert_called_once()
    actor.stop()


def test_actor_start_formats_ipv6_address():
    actor, _ = _make_actor()
    fake = _fake_popen()
    with (
        patch("ray.util.get_node_ip_address", return_value="[2001:db8::1]"),
        patch("subprocess.Popen", return_value=fake) as popen,
        patch("verl.workers.rollout.vllm_rollout.kv_cache_pool.probe_tcp"),
    ):
        addr = actor.start({"port": 50051}, enable_offload=False)
    cmd = popen.call_args.args[0]
    assert cmd[cmd.index("--rpc_address") + 1] == "2001:db8::1"
    assert addr == "[2001:db8::1]:50051"
    assert actor._proc._reader is not None
    actor.stop()


def test_actor_del_calls_stop():
    actor, cls = _make_actor()
    actor._proc = MagicMock()
    cls.__del__(actor)
    actor._proc.stop.assert_called()


def test_actor_del_on_actor_class_is_noop():
    """ray.remote() wraps the class; driver GC of ActorClass has no _proc."""
    MooncakeMasterActor.__del__(MooncakeMasterActor)
    fake_cls = type("ActorClass(MooncakeMasterActor)", (), {})()
    MooncakeMasterActor.__del__(fake_cls)


_DRIVER_NODE_ID = "a" * 56
_MASTER_ADDRESS = "10.0.0.1:50051"


def _rollout_cfg(*, enabled=True, auto_start=True, enable_offload=False, ssd_offload_path=None):
    return OmegaConf.create(
        {
            "cache_pool": {
                "enabled": enabled,
                "master": {"auto_start": auto_start, "address": None},
                "store": {"enable_offload": enable_offload, "ssd_offload_path": ssd_offload_path},
            }
        }
    )


@contextmanager
def _patch_setup_ray(*, get_actor_side_effect="missing", npu=False):
    runtime = MagicMock()
    runtime.get_job_id.return_value = "job-1"
    runtime.get_node_id.return_value = _DRIVER_NODE_ID
    actor = MagicMock()
    actor.get_address.remote.return_value = "addr-ref"
    actor.start.remote.return_value = "start-ref"
    options_handle = MagicMock()
    options_handle.remote.return_value = actor
    actor_cls = MagicMock()
    actor_cls.options.return_value = options_handle
    get_actor_kwargs = {"return_value": actor}
    if get_actor_side_effect == "missing":
        get_actor_kwargs["side_effect"] = ValueError("not found")
    elif get_actor_side_effect is not None:
        get_actor_kwargs["side_effect"] = get_actor_side_effect
    with (
        patch("verl.workers.rollout.vllm_rollout.kv_cache_pool.is_torch_npu_available", return_value=npu),
        patch("ray.get_actor", **get_actor_kwargs) as get_actor,
        patch("ray.get", return_value=_MASTER_ADDRESS) as ray_get,
        patch("ray.get_runtime_context", return_value=runtime) as runtime_ctx,
        patch("ray.kill") as kill,
        patch("atexit.register") as atexit_reg,
        patch("verl.workers.rollout.vllm_rollout.kv_cache_pool.ray.remote", return_value=actor_cls),
    ):
        yield SimpleNamespace(
            get_actor=get_actor,
            ray_get=ray_get,
            runtime=runtime_ctx,
            kill=kill,
            atexit_reg=atexit_reg,
            options=actor_cls.options,
            actor=actor,
            options_handle=options_handle,
        )


def test_setup_kv_cache_pool_disabled_skips_ray():
    cfg = _rollout_cfg(enabled=False)
    with _patch_setup_ray() as p:
        setup_kv_cache_pool(cfg)
    p.get_actor.assert_not_called()
    p.runtime.assert_not_called()
    p.options.assert_not_called()
    p.kill.assert_not_called()


def test_setup_kv_cache_pool_gpu_rejects_enable_offload_before_actor():
    cfg = _rollout_cfg(auto_start=False, enable_offload=True)
    with _patch_setup_ray() as p:
        with pytest.raises(NotImplementedError, match="offload"):
            setup_kv_cache_pool(cfg)
    p.get_actor.assert_not_called()
    p.options.assert_not_called()
    p.runtime.assert_not_called()


def test_setup_kv_cache_pool_gpu_rejects_ssd_offload_path_before_actor():
    cfg = _rollout_cfg(auto_start=False, ssd_offload_path="/nvme/x")
    with _patch_setup_ray() as p:
        with pytest.raises(ValueError, match="ssd_offload_path"):
            setup_kv_cache_pool(cfg)
    p.get_actor.assert_not_called()
    p.options.assert_not_called()
    p.runtime.assert_not_called()


def test_setup_kv_cache_pool_creates_named_actor_on_miss():
    cfg = _rollout_cfg()
    with _patch_setup_ray() as p:
        setup_kv_cache_pool(cfg)
    p.options.assert_called_once()
    kwargs = p.options.call_args.kwargs
    assert kwargs["name"] == "verl_mooncake_master_job-1"
    strategy = kwargs["scheduling_strategy"]
    assert isinstance(strategy, NodeAffinitySchedulingStrategy)
    assert strategy.node_id == _DRIVER_NODE_ID
    assert strategy.soft is False
    p.options_handle.remote.assert_called_once()
    p.actor.start.remote.assert_called_once()
    master_cfg, enable_offload = p.actor.start.remote.call_args.args
    assert enable_offload is False
    assert master_cfg["auto_start"] is True
    p.ray_get.assert_called()
    assert cfg.cache_pool.master.address == _MASTER_ADDRESS
    p.atexit_reg.assert_called_once()
    p.kill.assert_not_called()


def test_setup_kv_cache_pool_reuses_existing_actor():
    cfg = _rollout_cfg()
    with _patch_setup_ray(get_actor_side_effect=None) as p:
        setup_kv_cache_pool(cfg)
    p.get_actor.assert_called_once_with("verl_mooncake_master_job-1")
    p.options.assert_not_called()
    p.actor.get_address.remote.assert_called_once()
    p.actor.start.remote.assert_not_called()
    assert cfg.cache_pool.master.address == _MASTER_ADDRESS
    p.atexit_reg.assert_called_once()


def test_setup_kv_cache_pool_auto_start_false_skips_actor():
    cfg = _rollout_cfg(auto_start=False)
    with _patch_setup_ray() as p:
        setup_kv_cache_pool(cfg)
    p.get_actor.assert_not_called()
    p.options.assert_not_called()
    p.runtime.assert_not_called()
    assert cfg.cache_pool.master.address is None


def test_setup_kv_cache_pool_kills_actor_if_start_fails():
    cfg = _rollout_cfg()
    with _patch_setup_ray() as p:
        p.ray_get.side_effect = RuntimeError("start failed")
        with pytest.raises(RuntimeError, match="start failed"):
            setup_kv_cache_pool(cfg)
    p.kill.assert_called_once_with(p.actor)
    p.atexit_reg.assert_not_called()
    assert cfg.cache_pool.master.address is None


def test_llm_server_manager_create_sets_up_pool_before_replicas():
    from verl.workers.rollout.llm_server import LLMServerManager

    src = inspect.getsource(LLMServerManager.create)
    assert src.index("setup_kv_cache_pool") < src.index("_initialize_llm_servers")


def test_validate_cache_pool_engine_kwargs_rejects_nonempty():
    pytest.importorskip("vllm")
    from verl.workers.rollout.vllm_rollout.vllm_pd_replica import vLLMPDReplica

    cfg = SimpleNamespace(engine_kwargs={"vllm": {"kv_transfer_config": {"kv_connector": "MooncakeStoreConnector"}}})
    with pytest.raises(ValueError, match="engine_kwargs.vllm.kv_transfer_config"):
        vLLMPDReplica._validate_cache_pool_engine_kwargs(cfg)


def test_validate_cache_pool_engine_kwargs_allows_missing_or_empty():
    pytest.importorskip("vllm")
    from verl.workers.rollout.vllm_rollout.vllm_pd_replica import vLLMPDReplica

    vLLMPDReplica._validate_cache_pool_engine_kwargs(SimpleNamespace(engine_kwargs=None))
    vLLMPDReplica._validate_cache_pool_engine_kwargs(SimpleNamespace(engine_kwargs={}))
    vLLMPDReplica._validate_cache_pool_engine_kwargs(SimpleNamespace(engine_kwargs={"vllm": None}))
    vLLMPDReplica._validate_cache_pool_engine_kwargs(SimpleNamespace(engine_kwargs={"vllm": {}}))
    vLLMPDReplica._validate_cache_pool_engine_kwargs(
        SimpleNamespace(engine_kwargs={"vllm": {"kv_transfer_config": {}}})
    )
    vLLMPDReplica._validate_cache_pool_engine_kwargs(
        SimpleNamespace(engine_kwargs={"vllm": {"kv_transfer_config": None}})
    )


def test_materialize_mooncake_config_writes_gpu_json(tmp_path):
    out = tmp_path / "mooncake.json"
    config = SimpleNamespace(
        cache_pool=KVCachePoolConfig(enabled=True, master={"address": "10.0.0.1:50051"}),
    )
    with patch("verl.workers.rollout.vllm_rollout.kv_cache_pool.is_torch_npu_available", return_value=False):
        materialize_mooncake_config(config, {"MOONCAKE_CONFIG_PATH": str(out)})
    data = json.loads(out.read_text())
    assert data["master_server_address"] == "10.0.0.1:50051"
    assert data["enable_offload"] is False


def test_materialize_mooncake_config_user_file_missing_master(tmp_path):
    path = tmp_path / "user.json"
    path.write_text("{}")
    config = SimpleNamespace(
        cache_pool=KVCachePoolConfig(
            enabled=True,
            master={"auto_start": False},
            store={"config_path": str(path)},
        ),
    )
    with pytest.raises(ValueError, match="master_server_address"):
        materialize_mooncake_config(config, {})


def test_materialize_does_not_write_destination_in_place(tmp_path):
    dest = tmp_path / "mooncake.json"
    dest.write_text('{"placeholder": true}')
    original = Path.write_text

    def guarded(self, data, *args, **kwargs):
        if self.resolve() == dest.resolve():
            raise AssertionError("must not truncate destination in place")
        return original(self, data, *args, **kwargs)

    config = SimpleNamespace(
        cache_pool=KVCachePoolConfig(enabled=True, master={"address": "10.0.0.1:50051"}),
    )
    with (
        patch("verl.workers.rollout.vllm_rollout.kv_cache_pool.is_torch_npu_available", return_value=False),
        patch.object(Path, "write_text", guarded),
    ):
        materialize_mooncake_config(config, {"MOONCAKE_CONFIG_PATH": str(dest)})
    data = json.loads(dest.read_text())
    assert data["master_server_address"] == "10.0.0.1:50051"


def test_materialize_mooncake_config_npu_creates_ssd_dir(tmp_path):
    ssd = tmp_path / "ssd_offload"
    out = tmp_path / "mooncake.json"
    config = SimpleNamespace(
        cache_pool=KVCachePoolConfig(
            enabled=True,
            master={"address": "10.0.0.1:50051"},
            store={"ssd_offload_path": str(ssd)},
        ),
    )
    assert not ssd.exists()
    with patch("verl.workers.rollout.vllm_rollout.kv_cache_pool.is_torch_npu_available", return_value=True):
        materialize_mooncake_config(config, {"MOONCAKE_CONFIG_PATH": str(out)})
    assert ssd.is_dir()


def _make_spawn_replica(*, cache_pool=None, python_hash_seed=0):
    pytest.importorskip("vllm")
    from verl.utils.config import omega_conf_to_dataclass
    from verl.workers.config import RolloutConfig
    from verl.workers.rollout.replica import RolloutMode
    from verl.workers.rollout.vllm_rollout.vllm_pd_replica import vLLMPDReplica

    pool = cache_pool or {
        "enabled": True,
        "python_hash_seed": python_hash_seed,
    }
    cfg = omega_conf_to_dataclass(
        RolloutConfig(
            name="vllm",
            tensor_model_parallel_size=4,
            disaggregation={"enabled": True, "transfer_backend": "mooncake", "decode_replicas": 1},
            cache_pool=pool,
        )
    )
    replica = vLLMPDReplica.__new__(vLLMPDReplica)
    replica.replica_rank = 0
    replica.config = cfg
    replica.model_config = None
    replica.gpus_per_node = 8
    replica.gpus_per_replica_node = 8
    replica.rollout_mode = RolloutMode.HYBRID
    replica._prefill_tp = 4
    replica._decode_tp = 2
    replica.server_class = MagicMock()
    options = MagicMock()
    replica.server_class.options.return_value = options
    options.remote.return_value = "server-handle"
    return replica


def _spawn(replica, **overrides):
    kwargs = {
        "role": "prefill",
        "workers": [],
        "node_id": "a" * 56,
        "cuda_visible_devices": "0",
        "tp": 4,
        "kv_transfer_config": {"kv_connector": "preexisting"},
        "side_channel_host": "127.0.0.1",
        "side_channel_port": 19000,
        "mooncake_bootstrap_port": 19000,
        "actor_name": "vllm_server_test",
    }
    kwargs.update(overrides)
    runtime = MagicMock()
    runtime.get_job_id.return_value = "job-1"
    with patch("ray.get_runtime_context", return_value=runtime):
        handle = replica._spawn_pd_server(**kwargs)
    env_vars = replica.server_class.options.call_args.kwargs["runtime_env"]["env_vars"]
    remote_kwargs = replica.server_class.options.return_value.remote.call_args.kwargs
    return handle, env_vars, remote_kwargs


def test_spawn_injects_pool_env_and_omits_mooncake_master():
    replica = _make_spawn_replica(python_hash_seed=7)
    preexisting = {"kv_connector": "preexisting"}
    handle, env_vars, remote_kwargs = _spawn(replica, kv_transfer_config=preexisting)
    assert handle == "server-handle"
    assert env_vars["PYTHONHASHSEED"] == "7"
    assert env_vars["MOONCAKE_CONFIG_PATH"] == mooncake_json_path("job-1")
    assert env_vars["VERL_RAY_JOB_ID"] == "job-1"
    assert "MOONCAKE_MASTER" not in env_vars
    assert remote_kwargs["disaggregation_kv_transfer_config"] is preexisting


def test_spawn_uses_user_config_path_for_env():
    replica = _make_spawn_replica(
        cache_pool={
            "enabled": True,
            "master": {"auto_start": False},
            "store": {"config_path": "/tmp/user_mooncake.json"},
        }
    )
    _, env_vars, _ = _spawn(replica)
    assert env_vars["MOONCAKE_CONFIG_PATH"] == "/tmp/user_mooncake.json"


def _make_unified_replica(*, cache_pool=None, engine_kwargs=None, is_reward=False, is_teacher=False, tp=4):
    pytest.importorskip("vllm")
    from verl.workers.config import RolloutConfig
    from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMReplica

    cfg = RolloutConfig(
        name="vllm",
        tensor_model_parallel_size=tp,
        cache_pool=cache_pool or {"enabled": True, "python_hash_seed": 7},
        engine_kwargs=engine_kwargs or {},
    )
    replica = vLLMReplica.__new__(vLLMReplica)
    replica.config = cfg
    replica.is_reward_model = is_reward
    replica.is_teacher_model = is_teacher
    return replica


def _prepare_non_pd_kv_transfer(replica, *, is_npu=False, device="cuda"):
    runtime = MagicMock()
    runtime.get_job_id.return_value = "job-1"
    with (
        patch(
            "verl.workers.rollout.vllm_rollout.vllm_async_server.is_torch_npu_available",
            return_value=is_npu,
        ),
        patch(
            "verl.workers.rollout.vllm_rollout.vllm_async_server.get_device_name",
            return_value=device,
        ),
        patch(
            "verl.workers.rollout.vllm_rollout.vllm_async_server.ray.get_runtime_context",
            return_value=runtime,
        ),
    ):
        return replica._prepare_non_pd_kv_transfer()


def test_prepare_non_pd_kv_transfer_gpu_injects_store_and_env():
    replica = _make_unified_replica()
    kv_cfg, env_vars = _prepare_non_pd_kv_transfer(replica)
    assert kv_cfg["kv_connector"] == "MooncakeStoreConnector"
    assert kv_cfg["kv_role"] == "kv_both"
    assert kv_cfg["kv_buffer_device"] == "cuda"
    assert kv_cfg["kv_connector_extra_config"]["store_tp_size"] == 4
    assert kv_cfg["kv_connector_extra_config"]["lookup_rpc_port"] == kv_cfg["engine_id"]
    assert env_vars["PYTHONHASHSEED"] == "7"
    assert env_vars["MOONCAKE_CONFIG_PATH"] == mooncake_json_path("job-1")
    assert env_vars["VERL_RAY_JOB_ID"] == "job-1"
    assert "MOONCAKE_MASTER" not in env_vars


def test_prepare_non_pd_kv_transfer_npu_uses_ascend_store():
    replica = _make_unified_replica()
    kv_cfg, _ = _prepare_non_pd_kv_transfer(replica, is_npu=True, device="npu")
    assert kv_cfg["kv_connector"] == "AscendStoreConnector"
    assert kv_cfg["kv_role"] == "kv_both"
    assert kv_cfg["kv_buffer_device"] == "npu"
    assert "store_tp_size" not in kv_cfg["kv_connector_extra_config"]
    assert "use_layerwise" not in kv_cfg["kv_connector_extra_config"]


@pytest.mark.parametrize("kwargs", [{"is_reward": True}, {"is_teacher": True}])
def test_prepare_non_pd_kv_transfer_rejects_reward_and_teacher(kwargs):
    replica = _make_unified_replica(**kwargs)
    with pytest.raises(ValueError, match="reward or teacher"):
        _prepare_non_pd_kv_transfer(replica)


def test_prepare_non_pd_kv_transfer_disabled_pool_skips():
    replica = _make_unified_replica(cache_pool={"enabled": False})
    assert _prepare_non_pd_kv_transfer(replica) == (None, {})


def test_prepare_non_pd_kv_transfer_rejects_engine_kwargs_kv_transfer_config():
    replica = _make_unified_replica(
        engine_kwargs={"vllm": {"kv_transfer_config": {"kv_connector": "MooncakeStoreConnector"}}}
    )
    with pytest.raises(ValueError, match="kv_transfer_config"):
        _prepare_non_pd_kv_transfer(replica)


def test_spawn_disabled_pool_skips_pool_env_and_keeps_kv_cfg():
    replica = _make_spawn_replica(cache_pool={"enabled": False})
    preexisting = {"kv_connector": "MooncakeConnector"}
    _, env_vars, remote_kwargs = _spawn(replica, kv_transfer_config=preexisting)
    assert "PYTHONHASHSEED" not in env_vars
    assert "MOONCAKE_CONFIG_PATH" not in env_vars
    assert remote_kwargs["disaggregation_kv_transfer_config"] is preexisting
