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
"""Pure helpers for Mooncake JSON, Store TP, connector assembly, and master actor."""

import atexit
import functools
import json
import math
import socket
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import ray
from omegaconf import OmegaConf
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from verl.utils.device import is_torch_npu_available
from verl.utils.net_utils import get_free_port, is_valid_ipv6_address
from verl.workers.config.cache_pool import (
    KVCachePoolConfig,
    KVCachePoolMasterConfig,
    KVCachePoolStoreConfig,
)

_FREE_PORT_ATTEMPTS = 3
_ADDR_IN_USE_MARKS = ("address already in use", "eaddrinuse")
_ROLE_TO_KV_ROLE = {
    "prefill": "kv_producer",
    "decode": "kv_consumer",
}
_SIZE_UNITS = (
    ("GB", 1024**3),
    ("MB", 1024**2),
    ("KB", 1024),
    ("B", 1),
)
_GPU_STORE_PROTOCOLS = (None, "rdma", "tcp")
_NPU_STORE_PROTOCOLS = (None, "ascend")
_GPU_CONNECTOR_DEFAULTS = {
    "lookup_async": False,
    "cache_prefix": "",
    "store_tp_size": None,
    "enable_store_tp_lcm": None,
    "prefill_tp_sizes": None,
}
_NPU_CONNECTOR_DEFAULTS = {
    "consumer_is_to_put": False,
    "consumer_is_to_load": False,
    "use_layerwise": False,
    "prefill_pp_size": None,
    "prefill_pp_layer_partition": None,
}


def parse_size_to_bytes(value: str | int) -> int:
    """Parse a byte size that may already be an int or a B/KB/MB/GB string."""
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    text = str(value).strip()
    upper = text.upper()
    for suffix, multiplier in _SIZE_UNITS:
        if upper.endswith(suffix):
            number = upper[: -len(suffix)].strip()
            return int(number) * multiplier
    return int(upper)


def resolve_decode_tp(prefill_tp: int, decode_tensor_model_parallel_size: int | None) -> int:
    """Return decode TP, falling back to prefill TP when unset."""
    if decode_tensor_model_parallel_size is None:
        return prefill_tp
    return decode_tensor_model_parallel_size


def resolve_store_tp(
    *,
    prefill_tp: int,
    decode_tp: int,
    prefill_tps: list[int],
    user_store_tp_size: int | None,
    enable_store_tp_lcm: bool | None,
    user_prefill_tp_sizes: list[int] | None = None,
) -> dict:
    """Return extra fields: either store_tp_size or LCM keys, not both."""
    if user_store_tp_size is not None and enable_store_tp_lcm is True:
        raise ValueError("store_tp_size and enable_store_tp_lcm=True are mutually exclusive")

    heterogeneous_prefill = len(set(prefill_tps)) > 1
    if user_store_tp_size is not None:
        use_lcm = False
    elif enable_store_tp_lcm is True:
        use_lcm = True
    elif enable_store_tp_lcm is False:
        use_lcm = False
    else:
        use_lcm = heterogeneous_prefill

    if use_lcm:
        prefill_tp_sizes = list(user_prefill_tp_sizes if user_prefill_tp_sizes is not None else prefill_tps)
        if not prefill_tp_sizes:
            raise ValueError("enable_store_tp_lcm=True requires a non-empty prefill_tp_sizes")
        store_tp = functools.reduce(math.lcm, prefill_tp_sizes)
        _validate_store_tp(store_tp, prefill_tp=prefill_tp, decode_tp=decode_tp)
        return {"enable_store_tp_lcm": True, "prefill_tp_sizes": prefill_tp_sizes}

    store_tp = user_store_tp_size if user_store_tp_size is not None else math.lcm(prefill_tp, decode_tp)
    _validate_store_tp(store_tp, prefill_tp=prefill_tp, decode_tp=decode_tp)
    return {"store_tp_size": store_tp}


def _validate_store_tp(store_tp: int, *, prefill_tp: int, decode_tp: int) -> None:
    for tp in (prefill_tp, decode_tp):
        if store_tp < tp or store_tp % tp != 0:
            raise ValueError(
                f"store_tp_size={store_tp} must be >= and divisible by prefill_tp={prefill_tp} "
                f"and decode_tp={decode_tp}"
            )


def mooncake_json_path(job_id: str) -> str:
    """Return the per-job Mooncake JSON path under the process temp dir."""
    return str(Path(tempfile.gettempdir()) / f"verl_mooncake_{job_id}.json")


def build_mooncake_json(*, store: KVCachePoolStoreConfig, master_address: str, is_npu: bool) -> dict:
    """Build the Mooncake store JSON payload for GPU or NPU."""
    if is_npu:
        enable_ssd_offload = bool(store.ssd_offload_path)
        data = {
            "metadata_server": store.metadata_server,
            "protocol": store.protocol or "ascend",
            "device_name": store.device_name,
            "master_server_address": master_address,
            "global_segment_size": store.global_segment_size,
            "preferred_segment": store.preferred_segment,
            "prefer_alloc_in_same_node": store.prefer_alloc_in_same_node,
            "enable_ssd_offload": enable_ssd_offload,
            "tenant_id": store.tenant_id,
        }
        if enable_ssd_offload:
            data["ssd_offload_path"] = store.ssd_offload_path
        return data
    return {
        "mode": store.mode,
        "metadata_server": store.metadata_server,
        "master_server_address": master_address,
        "global_segment_size": store.global_segment_size,
        "local_buffer_size": store.local_buffer_size,
        "protocol": store.protocol or "rdma",
        "device_name": store.device_name,
        "enable_offload": False,
        "tenant_id": store.tenant_id,
    }


def build_pd_p2p_connector_config(
    *,
    role: str,
    transfer_backend: str,
    mooncake_protocol: str | None,
    use_ascend_mooncake: bool,
    kv_port: int | None,
    prefill_tp: int | None,
    decode_tp: int | None,
) -> dict:
    """Build a P2P child connector dict without engine_id or kv_buffer_device."""
    kv_role = _ROLE_TO_KV_ROLE[role]
    if use_ascend_mooncake:
        return {
            "kv_connector": "MooncakeConnectorV2",
            "kv_role": kv_role,
            "kv_port": kv_port,
            "kv_connector_extra_config": {
                "prefill": {"dp_size": 1, "tp_size": prefill_tp},
                "decode": {"dp_size": 1, "tp_size": decode_tp},
            },
        }
    if transfer_backend == "nixl":
        return {
            "kv_connector": "NixlConnector",
            "kv_role": kv_role,
        }
    cfg: dict = {
        "kv_connector": "MooncakeConnector",
        "kv_role": kv_role,
    }
    if mooncake_protocol:
        cfg["kv_connector_extra_config"] = {"mooncake_protocol": mooncake_protocol}
    return cfg


def build_pd_store_connector_config(
    *,
    role: str,
    is_npu: bool,
    cache_pool: KVCachePoolConfig,
    engine_id: str,
    prefill_tp: int,
    decode_tp: int,
    prefill_tps: list[int],
) -> dict:
    """Build the Store child connector dict for GPU or NPU."""
    if not isinstance(engine_id, str) or not engine_id:
        raise ValueError(f"engine_id is required to set lookup_rpc_port, got {engine_id!r}")

    connector = cache_pool.connector
    extra: dict = {}
    if connector.load_async is not None:
        extra["load_async"] = connector.load_async

    if is_npu:
        if connector.consumer_is_to_put:
            extra["consumer_is_to_put"] = True
        if connector.consumer_is_to_load:
            extra["consumer_is_to_load"] = True
        if connector.use_layerwise:
            extra["use_layerwise"] = True
        if connector.prefill_pp_size is not None:
            extra["prefill_pp_size"] = connector.prefill_pp_size
        if connector.prefill_pp_layer_partition is not None:
            extra["prefill_pp_layer_partition"] = connector.prefill_pp_layer_partition
        if role == "decode" and connector.save_decode_cache:
            extra["save_decode_cache"] = True
        kv_connector = "AscendStoreConnector"
        kv_role = _ROLE_TO_KV_ROLE[role]
    else:
        extra.update(
            resolve_store_tp(
                prefill_tp=prefill_tp,
                decode_tp=decode_tp,
                prefill_tps=prefill_tps,
                user_store_tp_size=connector.store_tp_size,
                enable_store_tp_lcm=connector.enable_store_tp_lcm,
                user_prefill_tp_sizes=connector.prefill_tp_sizes,
            )
        )
        if connector.lookup_async:
            extra["lookup_async"] = True
        if connector.cache_prefix:
            extra["cache_prefix"] = connector.cache_prefix
        if role == "decode" and connector.save_decode_cache:
            extra["save_decode_cache"] = True
        kv_connector = "MooncakeStoreConnector"
        kv_role = "kv_both" if role == "prefill" else "kv_consumer"

    extra.update(cache_pool.extra_config)
    extra["lookup_rpc_port"] = engine_id
    return {
        "kv_connector": kv_connector,
        "kv_role": kv_role,
        "kv_connector_extra_config": extra,
    }


def build_pd_kv_transfer_config(
    *,
    role: str,
    engine_id: str,
    kv_buffer_device: str,
    transfer_backend: str,
    mooncake_protocol: str | None,
    use_ascend_mooncake: bool,
    kv_port: int | None,
    prefill_tp: int | None,
    decode_tp: int | None,
    cache_pool: KVCachePoolConfig | None = None,
    prefill_tps: list[int] | None = None,
) -> dict:
    """Assemble a single P2P connector or a MultiConnector with a Store child."""
    p2p = build_pd_p2p_connector_config(
        role=role,
        transfer_backend=transfer_backend,
        mooncake_protocol=mooncake_protocol,
        use_ascend_mooncake=use_ascend_mooncake,
        kv_port=kv_port,
        prefill_tp=prefill_tp,
        decode_tp=decode_tp,
    )
    if cache_pool is None or not cache_pool.enabled:
        return {**p2p, "engine_id": engine_id, "kv_buffer_device": kv_buffer_device}

    resolved_prefill_tp = prefill_tp
    resolved_decode_tp = resolve_decode_tp(prefill_tp, decode_tp) if prefill_tp is not None else decode_tp
    resolved_prefill_tps = list(prefill_tps) if prefill_tps is not None else [prefill_tp]
    store = build_pd_store_connector_config(
        role=role,
        is_npu=use_ascend_mooncake,
        cache_pool=cache_pool,
        engine_id=engine_id,
        prefill_tp=resolved_prefill_tp,
        decode_tp=resolved_decode_tp,
        prefill_tps=resolved_prefill_tps,
    )
    cfg = {
        "engine_id": engine_id,
        "kv_buffer_device": kv_buffer_device,
        "kv_connector": "MultiConnector",
        "kv_role": _ROLE_TO_KV_ROLE[role],
        "kv_connector_extra_config": {"connectors": [p2p, store]},
    }
    if cache_pool.kv_load_failure_policy is not None:
        cfg["kv_load_failure_policy"] = cache_pool.kv_load_failure_policy
    return cfg


def build_non_pd_kv_transfer_config(
    *,
    is_npu: bool,
    cache_pool: KVCachePoolConfig,
    engine_id: str,
    tp: int,
    kv_buffer_device: str,
) -> dict:
    """Build a flat Store connector for non-PD rollout. Both platforms use kv_both.

    GPU writes ``store_tp_size`` (default: this replica's TP). NPU omits Store TP
    and does not write ``use_layerwise``; that field is memcache prefill-only.
    """
    if not isinstance(engine_id, str) or not engine_id:
        raise ValueError(f"engine_id is required to set lookup_rpc_port, got {engine_id!r}")
    if tp < 1:
        raise ValueError(f"tp must be >= 1, got {tp}")

    connector = cache_pool.connector
    extra: dict = {}
    if connector.load_async is not None:
        extra["load_async"] = connector.load_async

    if is_npu:
        kv_connector = "AscendStoreConnector"
    else:
        store_tp = connector.store_tp_size if connector.store_tp_size is not None else tp
        if store_tp < tp or store_tp % tp != 0:
            raise ValueError(
                f"store_tp_size={store_tp} must be >= and divisible by tensor_model_parallel_size={tp}"
            )
        extra["store_tp_size"] = store_tp
        if connector.lookup_async:
            extra["lookup_async"] = True
        if connector.cache_prefix:
            extra["cache_prefix"] = connector.cache_prefix
        kv_connector = "MooncakeStoreConnector"

    extra.update(cache_pool.extra_config)
    extra["lookup_rpc_port"] = engine_id
    cfg = {
        "kv_connector": kv_connector,
        "kv_role": "kv_both",
        "engine_id": engine_id,
        "kv_buffer_device": kv_buffer_device,
        "kv_connector_extra_config": extra,
    }
    if cache_pool.kv_load_failure_policy is not None:
        cfg["kv_load_failure_policy"] = cache_pool.kv_load_failure_policy
    return cfg


def p2p_connector_name(kv_transfer_config: dict) -> str:
    """Return the P2P connector class name from a flat or MultiConnector config."""
    if kv_transfer_config.get("kv_connector") != "MultiConnector":
        return kv_transfer_config["kv_connector"]
    extra = kv_transfer_config.get("kv_connector_extra_config") or {}
    connectors = extra.get("connectors")
    if not connectors:
        raise RuntimeError("MultiConnector missing connectors[0]")
    return connectors[0]["kv_connector"]


def _atomic_write_text(path: Path, text: str) -> None:
    """Replace ``path`` atomically so concurrent readers never see a truncate."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{time.time_ns()}.{threading.get_ident()}.tmp")
    tmp.write_text(text)
    try:
        tmp.replace(path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def materialize_mooncake_config(config, environ) -> None:
    """Write or validate Mooncake JSON before the vLLM engine starts."""
    pool = getattr(config, "cache_pool", None)
    if pool is None or not pool.enabled:
        return
    if pool.store.config_path:
        path = Path(pool.store.config_path)
        if not path.is_file():
            raise FileNotFoundError(f"cache_pool.store.config_path not found: {path}")
        data = json.loads(path.read_text())
        if not data.get("master_server_address"):
            raise ValueError(f"master_server_address missing in {path}")
        return
    address = pool.master.address
    if not address:
        raise ValueError("cache_pool.master.address")
    is_npu = is_torch_npu_available(check_device=False)
    if is_npu and pool.store.ssd_offload_path:
        Path(pool.store.ssd_offload_path).mkdir(parents=True, exist_ok=True)
    payload = build_mooncake_json(store=pool.store, master_address=address, is_npu=is_npu)
    out = Path(environ.get("MOONCAKE_CONFIG_PATH") or mooncake_json_path(environ.get("VERL_RAY_JOB_ID", "0")))
    _atomic_write_text(out, json.dumps(payload, indent=2))


def validate_platform_cache_pool(*, cache_pool: KVCachePoolConfig, is_npu: bool) -> None:
    """Raise if store/connector fields are illegal for the current accelerator."""
    store = cache_pool.store
    connector = cache_pool.connector
    if is_npu:
        if store.protocol not in _NPU_STORE_PROTOCOLS:
            raise ValueError(f"cache_pool.store.protocol={store.protocol!r} is not supported on NPU")
        if store.device_name != "":
            raise ValueError("cache_pool.store.device_name must be empty on NPU")
        segment_bytes = parse_size_to_bytes(store.global_segment_size)
        if segment_bytes % (2**30) != 0:
            raise ValueError(
                f"cache_pool.store.global_segment_size={store.global_segment_size!r} must be 1GB-aligned on NPU"
            )
        _reject_non_default(connector, _GPU_CONNECTOR_DEFAULTS)
        return

    if cache_pool.backend != "mooncake":
        raise ValueError(f"cache_pool.backend={cache_pool.backend!r} is not supported on GPU")
    if store.protocol not in _GPU_STORE_PROTOCOLS:
        raise ValueError(f"cache_pool.store.protocol={store.protocol!r} is not supported on GPU")
    if store.ssd_offload_path:
        raise ValueError("cache_pool.store.ssd_offload_path is not supported on GPU")
    if store.enable_offload:
        raise NotImplementedError("cache_pool.store.enable_offload is not implemented on GPU")
    if store.preferred_segment is True:
        raise ValueError("cache_pool.store.preferred_segment is not supported on GPU")
    if store.prefer_alloc_in_same_node is False:
        raise ValueError("cache_pool.store.prefer_alloc_in_same_node must stay default on GPU")
    _reject_non_default(connector, _NPU_CONNECTOR_DEFAULTS)


def _reject_non_default(obj, defaults: dict) -> None:
    for name, default in defaults.items():
        value = getattr(obj, name)
        if value != default:
            raise ValueError(f"{name} is not supported on this platform")


def build_mooncake_master_cmd(
    *,
    master: KVCachePoolMasterConfig,
    port: int,
    rpc_address: str,
    enable_offload: bool,
) -> list[str]:
    """Build the mooncake_master argv from master config and runtime flags."""
    cmd = [
        "mooncake_master",
        "--rpc_port",
        str(port),
        "--rpc_address",
        rpc_address,
    ]
    if master.eviction_high_watermark_ratio is not None:
        cmd.extend(["--eviction_high_watermark_ratio", str(master.eviction_high_watermark_ratio)])
    if master.eviction_ratio is not None:
        cmd.extend(["--eviction_ratio", str(master.eviction_ratio)])
    if master.default_kv_lease_ttl is not None:
        cmd.extend(["--default_kv_lease_ttl", str(master.default_kv_lease_ttl)])
    if master.client_ttl is not None:
        cmd.extend(["--client_ttl", str(master.client_ttl)])
    if enable_offload:
        cmd.append("--enable_offload=true")
    if master.enable_multi_tenants:
        cmd.append("--enable_multi_tenants=true")
        if master.tenant_quota_connector_type:
            cmd.extend(["--tenant_quota_connector_type", master.tenant_quota_connector_type])
        if master.tenant_quota_connector_uri:
            cmd.extend(["--tenant_quota_connector_uri", master.tenant_quota_connector_uri])
    return cmd


def format_host_port(host: str, port: int) -> str:
    """Format host:port, wrapping IPv6 literals in brackets."""
    host = host.strip("[]")
    if is_valid_ipv6_address(host):
        return f"[{host}]:{port}"
    return f"{host}:{port}"


def probe_tcp(host: str, port: int, timeout_s: float = 30.0, interval_s: float = 0.5, is_alive=None) -> None:
    """Retry TCP connect until success or timeout. Does not include cmd in the error."""
    deadline = time.monotonic() + timeout_s
    last_err: OSError | None = None
    while True:
        if is_alive is not None and not is_alive():
            raise RuntimeError(f"process exited while probing {host}:{port}")
        try:
            with socket.create_connection((host, port), timeout=max(interval_s, 0.001)):
                if is_alive is not None and not is_alive():
                    raise RuntimeError(f"process exited while probing {host}:{port}")
                return
        except OSError as e:
            last_err = e
        if time.monotonic() >= deadline:
            break
        if interval_s > 0:
            time.sleep(interval_s)
    raise RuntimeError(f"TCP probe timed out for {host}:{port}") from last_err


class MooncakeMasterProcess:
    """Own a local mooncake_master subprocess. start() must not probe TCP."""

    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None
        self._output = ""
        self._chunks: list[str] = []
        self._reader: threading.Thread | None = None
        self._log_output = False

    def start(self, cmd: list[str], *, log_output: bool = False, reserve_sock=None) -> None:
        self._chunks = []
        self._output = ""
        self._log_output = log_output
        if reserve_sock is not None:
            try:
                reserve_sock.close()
            except OSError:
                pass
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        stdout = self._proc.stdout
        if stdout is None:
            self._reader = None
            return
        self._reader = threading.Thread(
            target=self._forward_stdout,
            args=(stdout,),
            name="mooncake-master-stdout",
            daemon=True,
        )
        self._reader.start()

    def _forward_stdout(self, stdout) -> None:
        pending = ""
        try:
            while True:
                chunk = stdout.read(4096)
                if not chunk:
                    break
                if isinstance(chunk, bytes):
                    chunk = chunk.decode(errors="replace")
                elif not isinstance(chunk, str):
                    break
                self._chunks.append(chunk)
                if not self._log_output:
                    continue
                pending += chunk
                while "\n" in pending:
                    line, pending = pending.split("\n", 1)
                    print(line, flush=True)
            if self._log_output and pending:
                print(pending, flush=True)
        except Exception:
            pass

    def stop(self) -> None:
        proc = self._proc
        if proc is not None:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            self._proc = None
        reader = self._reader
        if reader is not None:
            reader.join(timeout=1.0)
            self._reader = None
        self._output = "".join(self._chunks)


class MooncakeMasterActor:
    """Job-level mooncake_master owner. Probe lives here, not in MooncakeMasterProcess.start."""

    def __init__(self) -> None:
        self._proc = MooncakeMasterProcess()
        self._address: str | None = None

    def start(self, master_cfg: dict, enable_offload: bool) -> str:
        master = (
            master_cfg
            if isinstance(master_cfg, KVCachePoolMasterConfig)
            else KVCachePoolMasterConfig(**dict(master_cfg))
        )
        ip = ray.util.get_node_ip_address().strip("[]")
        pick_port = master.port in (None, 0)
        last_error: Exception | None = None
        for _ in range(_FREE_PORT_ATTEMPTS if pick_port else 1):
            reserve_sock = None
            port = master.port
            if pick_port:
                port, reserve_sock = get_free_port(ip, with_alive_sock=True)
            cmd = build_mooncake_master_cmd(
                master=master, port=port, rpc_address=ip, enable_offload=enable_offload
            )
            try:
                self._proc.start(cmd, log_output=master.log_master_output, reserve_sock=reserve_sock)
            except FileNotFoundError as e:
                self.stop()
                pkg = (
                    "mooncake-transfer-engine-npu"
                    if is_torch_npu_available(check_device=False)
                    else "mooncake-transfer-engine"
                )
                raise RuntimeError(f"mooncake_master not found; install {pkg}") from e

            address = format_host_port(ip, port)
            try:
                self._wait_until_listening(ip, port)
            except Exception as e:
                output = self._stop_and_capture_output()
                last_error = RuntimeError(
                    f"mooncake_master failed to listen on {address}; cmd={cmd!r}; output={output}"
                )
                if not pick_port or not any(mark in str(last_error).lower() for mark in _ADDR_IN_USE_MARKS):
                    raise last_error from e
                continue
            self._address = address
            return self._address
        raise last_error

    def _child_is_alive(self) -> bool:
        popen = getattr(self._proc, "_proc", None)
        return popen is not None and popen.poll() is None

    def _wait_until_listening(self, ip: str, port: int) -> None:
        if not self._child_is_alive():
            raise RuntimeError("mooncake_master exited before listening")
        probe_tcp(ip, port, is_alive=self._child_is_alive)

    def _stop_and_capture_output(self) -> str:
        self._proc.stop()
        return self._proc._output or ""

    def get_address(self) -> str:
        return self._address

    def stop(self) -> None:
        proc = getattr(self, "_proc", None)
        if proc is not None:
            proc.stop()

    def __del__(self) -> None:
        proc = getattr(self, "_proc", None)
        if proc is None:
            return
        try:
            proc.stop()
        except Exception:
            pass


def setup_kv_cache_pool(rollout_cfg) -> None:
    """Start or reuse the job-level mooncake_master and write ``master.address``.

    GPU offload is rejected before any Ray actor lookup or create. ``auto_start=False``
    returns after that check. Named-actor create binds to the driver node; a failed
    ``start`` kills the actor so the name is not stuck.
    """
    pool = rollout_cfg.get("cache_pool")
    if pool is None or not pool.get("enabled", False):
        return

    # Fail GPU offload before auto_start / get_actor.
    if not is_torch_npu_available(check_device=False):
        store = pool.get("store") or {}
        if store.get("ssd_offload_path"):
            raise ValueError("cache_pool.store.ssd_offload_path is not supported on GPU")
        if store.get("enable_offload"):
            raise NotImplementedError("cache_pool.store.enable_offload is not implemented on GPU")

    auto_start = pool.master.auto_start
    if not auto_start:
        return

    name = f"verl_mooncake_master_{ray.get_runtime_context().get_job_id()}"
    try:
        actor = ray.get_actor(name)
    except ValueError:
        enable_offload = bool((pool.get("store") or {}).get("ssd_offload_path"))
        driver_node = ray.get_runtime_context().get_node_id()
        actor = ray.remote(MooncakeMasterActor).options(
            name=name,
            scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=driver_node, soft=False),
        ).remote()
        try:
            master_cfg = OmegaConf.to_container(pool.master, resolve=True)
            address = ray.get(actor.start.remote(master_cfg, enable_offload))
        except Exception:
            ray.kill(actor)
            raise
    else:
        address = ray.get(actor.get_address.remote())
    OmegaConf.update(rollout_cfg, "cache_pool.master.address", address, force_add=True)
    atexit.register(lambda: ray.get(actor.stop.remote()))
