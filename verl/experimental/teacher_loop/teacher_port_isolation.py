# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""Port isolation for several colocated vLLM replicas on one node.

The shared teacher fleet runs ``num_replicas`` colocated replicas per node. Two
independent rendezvous allocate ports by probing and *closing* them, so replicas
created concurrently race on the same port:

1. ``RayWorkerGroup`` asks the node for a free ``MASTER_PORT`` for the gloo
   process group (``verl/single_controller/ray/base.py``). The shared-fleet
   replica overrides ``init_colocated`` and passes ``master_port_range`` from
   :func:`replica_gloo_port_range` (one disjoint window per replica).
2. vLLM's ``MultiprocExecutor`` builds the TP ``TCPStore`` with
   ``get_open_port()``, which is the same probe-bind-close pattern. Setting
   ``VLLM_PORT`` per replica is not enough, because the scan keeps incrementing
   on a busy port and walks into the next replica's window.
   :func:`install_vllm_port_guard` covers this: the reserved port is reused
   without probing, the scan is capped to the replica's window, and a per-node
   ``flock`` covers the gap between picking the port and the ``TCPStore``
   listening.

All windows sit below the usual ephemeral start (32768) so HCCL's ``bind(0)``
cannot take them, and the three replica kinds (student/teacher/reward) live in
disjoint bands.
"""

from __future__ import annotations

import fcntl
import logging
import os
import socket
import threading
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Below the common Linux ip_local_port_range start (32768).
VLLM_PORT_BASE = 12000
VLLM_PORT_STRIDE = 64
VLLM_PORT_KIND_OFFSET = {
    "student": 0,
    "teacher": 10000,
    "reward": 20000,
}
# Ports [base, base + REPLICA_DIST_PORT_BLOCK) are used by the vLLM engine: the
# reserved VLLM_PORT lives at the stride base. Ports [base + 4, base + stride)
# are reserved for the gloo MASTER_PORT scan.
REPLICA_DIST_PORT_BLOCK = 4


def replica_port_kind(*, is_teacher_model: bool = False, is_reward_model: bool = False) -> str:
    if is_reward_model:
        return "reward"
    if is_teacher_model:
        return "teacher"
    return "student"


def replica_stride_base(replica_rank: int, replica_kind: str = "student") -> int:
    """Return the first port number in this replica's reserved stride."""
    if replica_rank < 0:
        raise ValueError(f"replica_rank must be >= 0, got {replica_rank}")
    if replica_kind not in VLLM_PORT_KIND_OFFSET:
        raise ValueError(
            f"unknown replica_kind {replica_kind!r}, expected one of {sorted(VLLM_PORT_KIND_OFFSET)}"
        )
    base = VLLM_PORT_BASE + VLLM_PORT_KIND_OFFSET[replica_kind] + replica_rank * VLLM_PORT_STRIDE
    if base + VLLM_PORT_STRIDE - 1 > 65535:
        raise ValueError(
            f"computed port stride [{base}, {base + VLLM_PORT_STRIDE}) exceeds 65535 "
            f"for replica_kind={replica_kind} replica_rank={replica_rank}"
        )
    return base


def replica_vllm_port(replica_rank: int, replica_kind: str = "student") -> int:
    """Reserved ``VLLM_PORT`` for one colocated vLLM engine."""
    return replica_stride_base(replica_rank, replica_kind)


def replica_gloo_port_range(replica_rank: int, replica_kind: str = "student") -> list[int]:
    """Exclusive ``MASTER_PORT`` scan window for this replica's gloo process group.

    ``RayWorkerGroup`` probes a free port and closes it, so concurrently created
    worker groups on one node can be handed the same port and then block in
    ``TCPStore`` validation. The window keeps every replica to its own ports.
    """
    base = replica_stride_base(replica_rank, replica_kind)
    return [base + REPLICA_DIST_PORT_BLOCK, base + VLLM_PORT_STRIDE]


def replica_server_env(replica_rank: int, replica_kind: str = "student") -> dict[str, str]:
    """Environment that pins one replica's vLLM engine to its own port window.

    Applied to the rollout HTTP server actor, which spawns the engine, so the
    engine's ``TCPStore`` and data-parallel rendezvous stay inside the window.
    """
    stride_base = replica_stride_base(replica_rank, replica_kind)
    return {
        "VLLM_PORT": str(replica_vllm_port(replica_rank, replica_kind)),
        "VERL_VLLM_PORT_STRIDE": str(VLLM_PORT_STRIDE),
        "VERL_VLLM_PORT_SCAN_END": str(stride_base + VLLM_PORT_STRIDE),
        "VERL_REPLICA_RANK": str(replica_rank),
        "VERL_REPLICA_KIND": replica_kind,
    }


class ServerClassWithEnv:
    """Actor-class proxy that merges extra env vars into ``.options(runtime_env=...)``."""

    def __init__(self, actor_cls: Any, extra_env: dict[str, str]):
        self._actor_cls = actor_cls
        self._extra_env = extra_env

    def options(self, *args, **kwargs):
        runtime_env = dict(kwargs.get("runtime_env") or {})
        env_vars = {**dict(runtime_env.get("env_vars") or {}), **self._extra_env}
        kwargs["runtime_env"] = {**runtime_env, "env_vars": env_vars}
        return self._actor_cls.options(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._actor_cls, name)


# ---------------------------------------------------------------------------
# vLLM engine: keep every probe inside this replica's window
# ---------------------------------------------------------------------------
def _stride_base_from_env() -> Optional[int]:
    rank_raw = os.environ.get("VERL_REPLICA_RANK")
    kind = os.environ.get("VERL_REPLICA_KIND")
    if not rank_raw or not kind:
        return None
    try:
        rank = int(rank_raw)
    except ValueError:
        return None
    if rank < 0 or kind not in VLLM_PORT_KIND_OFFSET:
        return None
    return replica_stride_base(rank, kind)


def _infer_stride_base_from_port(vllm_port: int, stride: int = VLLM_PORT_STRIDE) -> int:
    return vllm_port - (vllm_port - VLLM_PORT_BASE) % stride


def replica_port_scan_end(vllm_port: Optional[int] = None) -> int:
    """First port *outside* this replica's exclusive window."""
    env_end = os.environ.get("VERL_VLLM_PORT_SCAN_END")
    if env_end:
        return int(env_end)
    stride = int(os.environ.get("VERL_VLLM_PORT_STRIDE", str(VLLM_PORT_STRIDE)))
    stride_base = _stride_base_from_env()
    if stride_base is not None:
        return stride_base + stride
    port = vllm_port if vllm_port is not None else int(os.environ.get("VLLM_PORT", "0") or 0)
    if port <= 0:
        return 0
    return _infer_stride_base_from_port(port, stride) + stride


def wait_until_port_listening(port: int, timeout_s: Optional[float] = None, interval_s: float = 0.05) -> bool:
    if timeout_s is None:
        timeout_s = float(os.environ.get("VERL_VLLM_TCPSTORE_LISTEN_TIMEOUT_S", "20"))
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.settimeout(interval_s)
            sock.connect(("127.0.0.1", port))
            return True
        except OSError:
            time.sleep(interval_s)
        finally:
            sock.close()
    return False


class NodeTcpStoreBindLock:
    """Exclusive per-node lock covering the get_open_port TOCTOU until TCPStore listens."""

    def __init__(self, lock_path: Optional[str] = None):
        job = os.environ.get("VERL_RAY_JOB_ID") or os.environ.get("RAY_JOB_ID") or "local"
        host = os.uname().nodename
        self.lock_path = lock_path or f"/tmp/verl-vllm-tcpstore-{job}-{host}.lock"
        self._handle: Any = None

    def acquire(self) -> None:
        self._handle = open(self.lock_path, "w")
        fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX)

    def release(self) -> None:
        handle = self._handle
        self._handle = None
        if handle is None:
            return
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        try:
            handle.close()
        except OSError:
            pass


_tls = threading.local()


def pick_replica_tcpstore_port() -> int:
    """Return this replica's reserved TCPStore port without probe-bind-close.

    The node lock is taken on the first call and released by a background thread
    once the port is listening, so later ``get_open_port()`` calls in the same
    executor keep scanning inside the replica window.
    """
    port = int(os.environ["VLLM_PORT"])
    if not getattr(_tls, "tcpstore_port_issued", False):
        lock = NodeTcpStoreBindLock()
        lock.acquire()
        _tls.tcpstore_port_issued = True
        threading.Thread(
            target=_release_tcpstore_lock_when_listening,
            args=(port, lock),
            name="verl-vllm-tcpstore-lock",
            daemon=True,
        ).start()
    return port


def _release_tcpstore_lock_when_listening(port: int, lock: NodeTcpStoreBindLock) -> None:
    try:
        wait_until_port_listening(port)
    finally:
        lock.release()


def capped_vllm_get_open_port(orig_get_open_port, orig_get_open_port_impl):
    """Wrap vLLM ``get_open_port`` so a scan cannot leave this replica's window."""

    def _wrapped() -> int:
        reserved = getattr(_tls, "use_reserved_tcpstore_port", False)
        issued = getattr(_tls, "tcpstore_port_issued", False)
        if reserved and not issued and os.environ.get("VLLM_PORT"):
            return pick_replica_tcpstore_port()
        scan_end = replica_port_scan_end()
        start = int(os.environ["VLLM_PORT"]) if os.environ.get("VLLM_PORT") else None
        if start is None or scan_end <= start:
            return orig_get_open_port()
        if issued:
            start += 1
        max_attempts = max(1, scan_end - start)
        return orig_get_open_port_impl(start_port=start, max_attempts=max_attempts)

    return _wrapped


def _wrap_executor_init(cls) -> None:
    orig = cls._init_executor
    if getattr(orig, "_verl_tcpstore_guarded", False):
        return

    def wrapped(self):
        _tls.use_reserved_tcpstore_port = True
        _tls.tcpstore_port_issued = False
        try:
            return orig(self)
        finally:
            _tls.use_reserved_tcpstore_port = False

    wrapped._verl_tcpstore_guarded = True
    cls._init_executor = wrapped


_orig_engine_core_run = None


def engine_core_run_with_port_guard(*args, **kwargs):
    """Spawn-safe EngineCore entrypoint.

    ``multiprocessing`` spawn pickles this function by module path, so the child
    re-imports it from verl (not a monkeypatched vLLM class) and installs the
    TCPStore guard before ``run_engine_core``.
    """
    install_vllm_port_guard()
    orig = _orig_engine_core_run
    if orig is None:
        from vllm.v1.engine.core import EngineCoreProc

        orig = EngineCoreProc.run_engine_core
    if orig is engine_core_run_with_port_guard:
        raise RuntimeError("EngineCore port-guard wrapper would recurse")
    return orig(*args, **kwargs)


def _patch_engine_core_entrypoint() -> None:
    global _orig_engine_core_run
    try:
        from vllm.v1.engine.core import EngineCoreProc
    except Exception as exc:  # vLLM not importable / different layout
        logger.debug("[teacher_port] EngineCoreProc not importable; skip entrypoint patch: %s", exc)
        return
    current = EngineCoreProc.run_engine_core
    if current is engine_core_run_with_port_guard:
        return
    _orig_engine_core_run = current
    EngineCoreProc.run_engine_core = staticmethod(engine_core_run_with_port_guard)


_installed = False


def install_vllm_port_guard() -> bool:
    """Patch vLLM port allocation for colocated engines. Safe to call repeatedly.

    Must run inside the rollout server actor (and in the EngineCore child, see
    :func:`engine_core_run_with_port_guard`). Returns ``True`` when the guard is
    active, ``False`` when vLLM is unavailable or the build has no
    ``network_utils`` to wrap.
    """
    global _installed
    if _installed:
        return True
    try:
        import vllm.utils.network_utils as net
    except Exception as exc:
        logger.debug("[teacher_port] vLLM network_utils not importable; skip port guard: %s", exc)
        return False

    if not getattr(net.get_open_port, "_verl_capped", False):
        orig = net.get_open_port
        orig_impl = net._get_open_port
        wrapped = capped_vllm_get_open_port(orig, orig_impl)
        wrapped._verl_capped = True
        net.get_open_port = wrapped

        def _capped_impl(start_port=None, max_attempts=None):
            scan_end = replica_port_scan_end()
            start = (
                start_port
                if start_port is not None
                else (int(os.environ["VLLM_PORT"]) if os.environ.get("VLLM_PORT") else None)
            )
            if start is not None and scan_end > 0:
                if start >= scan_end:
                    raise RuntimeError(
                        f"vLLM port scan exceeded the replica window: start={start} scan_end={scan_end}"
                    )
                cap = scan_end - start
                max_attempts = cap if max_attempts is None else min(max_attempts, cap)
            return orig_impl(start_port=start_port, max_attempts=max_attempts)

        net._get_open_port = _capped_impl

    for module_name, attr in (
        ("vllm.v1.executor.multiproc_executor", "MultiprocExecutor"),
        ("vllm.v1.executor.multiproc_executor", "get_open_port"),
        ("vllm_ascend.patch.platform.patch_multiproc_executor", "AscendMultiprocExecutor"),
        ("vllm_ascend.patch.platform.patch_multiproc_executor", "get_open_port"),
    ):
        try:
            module = __import__(module_name, fromlist=[attr])
        except Exception:
            continue
        if attr == "get_open_port":
            setattr(module, attr, net.get_open_port)
            continue
        cls = getattr(module, attr, None)
        if cls is not None:
            _wrap_executor_init(cls)

    _patch_engine_core_entrypoint()
    _installed = True
    os.environ["VERL_VLLM_PORT_GUARD_INSTALLED"] = "1"
    return True
