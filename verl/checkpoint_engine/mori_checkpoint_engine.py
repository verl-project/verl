# Copyright 2026 Advanced Micro Devices, Inc.
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
"""MoRI checkpoint engine: actor -> rollout weight sync with one-sided RDMA WRITE over MoRI-IO
(https://github.com/ROCm/mori).

Topology: P = min(num_chains, actor world size) stripes. Actor rank c sends stripe c (each tensor goes to the stripe
with the fewest bytes so far; all actor ranks iterate the same generator, so they agree without talking). On every
rollout node one GPU holds stripe c (local GPU c % n): it receives the stripe from outside and copies it to the other
GPUs of its node (over XGMI when available). The holders of a stripe form a chain across rollout nodes, so each node
receives one copy of the model, spread over all its NICs.

Every worker registers ONE device region once (first `prepare`) and keeps it, the QPs and the ZMQ sockets across
syncs: num_slots slots of bucket_size / num_chains bytes per stripe. Bucket k of a stripe uses slot k % num_slots.
Control messages (ZMQ):
  * prev -> me   {"t": "bucket", c, k, slot, uid, metas, used, is_last}   bucket k of stripe c is in my slot
  * me -> prev   {"t": "free", c, slot, from}                             my slot may be overwritten
  * me -> all    {"t": "abort", reason}                                   fail now instead of timing out
A next hop grants its slots (sends "free" for each) when it starts receiving, so no worker is written to before it
has entered the sync. No worker synchronizes the whole device: a write waits for an event recorded after its packing
copies, and a receiver frees a slot once an event recorded after the consumer's reads fired (consumers read on the
current stream, as verl's server adapters do) and the slot's forward writes completed. build_topology gives each
topology an id that tags every message, so workers that join, leave or are replaced never act on another sync's
messages.

Failures are fail-stop: a worker whose sync fails, is cancelled or is closed early refuses all further use, and a
worker refuses a new sync while its previous one runs; peers that leave are released, and the MoRI engine is replaced
once many have left. Why and how: see "mori detail" in verl/checkpoint_engine/README.md.

Notes (MI355X, Pollara, MoRI 1.2.4): MoRI cannot register PyTorch expandable-segment (VMM) memory, so the region is
allocated with cupy; transfer chunking must be on (off halves the bandwidth of one transfer); XGMI writes raise no
inbound notification at the target and a bounded `WaitFor` on them sleeps to its timeout (ROCm/mori#873), so
transfers are polled and receivers of XGMI hops rely on the bucket message, which is sent after the write completed.

Args (engine_kwargs["mori"]):
    num_chains (int): Max stripes. Default 8.
    num_slots (int): Buckets in flight per stripe and hop (>= 2); costs num_slots * bucket_size of device memory per
        worker. Default 2.
    intra_node (str): "xgmi" (default) copies within a node over XGMI; "rdma" keeps it on the NIC.
    node_id (str): Node identity used to group rollout workers. Default: the Ray node id.
    qp_per_transfer (int), num_nics_per_transfer (int), enable_chunking (bool): MoRI RDMA settings (4, 1, True).
    host (str): IP for MoRI and ZMQ. Default: the Ray node IP.
    timeout_s (float): Max wait for any single transfer or control message. Default 600.
    device (str): "cuda" (default) or "cpu" (tests only).
"""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import os
import socket
import time
import uuid
from types import SimpleNamespace
from typing import AsyncGenerator, AsyncIterator, Generator

import torch
import zmq
import zmq.asyncio

from verl.checkpoint_engine.base import (
    CheckpointEngine,
    CheckpointEngineRegistry,
    TensorMeta,
    merge_weight_chunks,
    split_weight_chunks,
)
from verl.workers.rollout.utils import ensure_async_iterator

# MoRI is imported lazily (first prepare()), so importing verl.checkpoint_engine never loads the MoRI/HIP stack;
# only its presence is checked here, which lets verl record a missing dependency as an import error.
if importlib.util.find_spec("mori") is None:
    raise ImportError("the mori checkpoint engine needs MoRI (pip install amd-mori)")

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_ALIGN = 256  # chunk offsets inside a bucket and bucket sizes; keeps every dtype view aligned
_MAX_RETIRED_PEERS = 16  # peers that may leave before the MoRI engine is replaced to release what MoRI keeps for them
# MoRI engines, regions and transfer statuses of engines that became unusable: never freed while the process lives
_KEEPALIVE: list = []


def _align(n: int) -> int:
    return (n + _ALIGN - 1) // _ALIGN * _ALIGN


def _alloc_region(nbytes: int, device: str) -> tuple[torch.Tensor, object]:
    """Allocate the region outside PyTorch's caching allocator (no expandable segments). Returns (tensor, owner)."""
    if device != "cuda":
        return torch.zeros(nbytes, dtype=torch.uint8, device=device), None
    try:
        from unittest.mock import patch

        with patch("importlib.metadata.distributions", return_value=[]):
            import cupy as cp
        owner = cp.zeros(nbytes, dtype=cp.uint8)
        t = torch.as_tensor(owner, dtype=torch.uint8, device="cuda")
        torch.cuda.synchronize()  # the zero-fill runs on cupy's stream; finish it before anything else uses the region
        return t, owner
    except Exception as e:  # no cupy, or no __cuda_array_interface__ interop: allocate with expandable segments off
        logger.warning(f"mori: cupy region allocation failed ({e!r}); using torch with expandable segments off")
        conf = os.environ.get("PYTORCH_ALLOC_CONF", "") + os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
        torch.cuda.memory._set_allocator_settings("expandable_segments:False")
        try:
            t = torch.zeros(nbytes, dtype=torch.uint8, device="cuda")
            torch.cuda.synchronize()
            return t, None
        finally:
            if "expandable_segments:True" in conf:
                torch.cuda.memory._set_allocator_settings("expandable_segments:True")


def _node_ip() -> str:
    try:
        import ray

        return ray.util.get_node_ip_address().strip("[]")
    except Exception:
        return socket.gethostbyname(socket.gethostname())


class _ReceiveStream:
    """What `receive_weights` returns: an async iterator over (name, tensor) whose `aclose()` also ends a sync that
    the consumer abandoned before its first tensor (an async generator closed before it started runs none of its
    code, so it could not mark the engine unusable or tell the peers)."""

    def __init__(self, engine: MoriCheckpointEngine, seq: tuple, global_steps: int | None):
        self._engine, self._seq, self._global_steps = engine, seq, global_steps
        self._gen = None
        self._closed = False

    def __aiter__(self):
        # `async for` iterates the inner generator itself: no extra await per tensor (there are ~19k for a 30B MoE)
        if self._gen is None and not self._closed:
            self._gen = self._engine._receive(self._seq, self._global_steps)
        return self._gen if self._gen is not None else self

    async def __anext__(self):
        if self._gen is None:
            if self._closed:
                raise StopAsyncIteration
            self._gen = self._engine._receive(self._seq, self._global_steps)
        return await self._gen.__anext__()

    async def aclose(self):
        if self._closed:
            return
        self._closed = True
        if self._gen is not None:
            await self._gen.aclose()  # a started generator runs its own failure path unless it completed
        eng = self._engine
        if eng._busy and eng._unusable is None and (eng._sync_id, eng._round) == self._seq:
            # never started, also when aiter() created the generator but no anext() ran it: the sync is abandoned
            err = RuntimeError("the consumer stopped before the first tensor")
            eng._mark_unusable(f"receive failed: {err}")
            await eng._notify_abort(self._seq, err)


@CheckpointEngineRegistry.register("mori")
class MoriCheckpointEngine(CheckpointEngine):
    """MoRI-IO checkpoint engine (NIC-striped RDMA WRITE, XGMI fan-out, one persistent registered region). See the
    module docstring.

    Args:
        bucket_size (int): Region size is num_slots * bucket_size per process; each stripe gets num_slots slots of
            bucket_size / num_chains bytes.
        is_master (bool): Unused; accepted for signature compatibility with the other engines.
    """

    def __init__(
        self,
        bucket_size: int,
        is_master: bool = False,
        num_chains: int = 8,
        num_slots: int = 2,
        qp_per_transfer: int = 4,
        num_nics_per_transfer: int = 1,
        enable_chunking: bool = True,
        host: str | None = None,
        timeout_s: float = 600.0,
        device: str = "cuda",
        intra_node: str = "xgmi",
        node_id: str | None = None,
    ):
        self.bucket_size = bucket_size
        self.is_master = is_master
        self.num_chains = max(1, int(num_chains))
        self.chain_bucket = max(_ALIGN, bucket_size // self.num_chains // _ALIGN * _ALIGN)
        assert int(num_slots) >= 2, f"num_slots must be >= 2, got {num_slots}"
        self.num_slots = int(num_slots)
        self.qp_per_transfer = qp_per_transfer
        self.num_nics_per_transfer = num_nics_per_transfer
        self.enable_chunking = enable_chunking
        self.host = host
        self.timeout_s = timeout_s
        self.device = device
        assert intra_node in ("xgmi", "rdma"), f"intra_node must be xgmi or rdma, got {intra_node!r}"
        self.intra_node = intra_node
        self.node_id = node_id
        self._chain_rank = None  # last position, kept for logs after finalize

        # created once, on first prepare(), and kept across syncs
        self._engine = None  # mori.io.IOEngine
        self._region: torch.Tensor | None = None
        self._owner = None
        self._mem = None  # mori.io.MemoryDesc of my region
        self._key: str | None = None
        self._zctx = None
        self._inbox = None
        self._inbox_addr: str | None = None
        self._push: dict[str, object] = {}  # inbox address -> PUSH socket
        self._peers: dict = {}  # key -> mori.io.EngineDesc of remote engines registered with MoRI
        # statuses of my MoRI writes not yet known to have succeeded (MoRI's completion callback points at them)
        self._inflight: set = set()
        self._unusable: str | None = None  # why this engine refuses all further use (fail-stop, see module docstring)
        self._busy = False  # a sync is running on this worker (or never finished)
        self._retired = 0  # peers dropped since the MoRI engine was built; MoRI keeps resources for each of them

        # per topology (set by init_process_group): every worker of a sync gets the same id from build_topology and
        # counts rounds from 0, so stale messages of earlier syncs or of workers that left are told apart
        self._sync_id: str | None = None
        self._round = 0

        # per sync (set by init_process_group, cleared by finalize)
        self.rank: int | None = None
        self._P = 0
        self._roles: dict[int, SimpleNamespace] = {}  # chain -> SimpleNamespace(prev, next)

    def _off(self, c: int, j: int) -> int:
        return (self.num_slots * c + j) * self.chain_bucket

    def _slot(self, c: int, j: int) -> torch.Tensor:
        o = self._off(c, j)
        return self._region[o : o + self.chain_bucket]

    # ------------------------------------------------------------------ setup
    def _setup(self):
        """Region and ZMQ inbox (once per worker), then the MoRI engine."""
        self._region, self._owner = _alloc_region(self.num_slots * self.num_chains * self.chain_bucket, self.device)
        ip = self.host or _node_ip()
        self._zctx = zmq.asyncio.Context()
        # never block on close: with the default (infinite) linger, garbage-collecting the context waits for every
        # undelivered message, e.g. an abort to a peer that is gone, and hangs the process at exit
        self._zctx.setsockopt(zmq.LINGER, 0)
        self._inbox = self._zctx.socket(zmq.PULL)
        if ":" in ip:
            self._inbox.setsockopt(zmq.IPV6, 1)
            port = self._inbox.bind_to_random_port("tcp://[::]")
            self._inbox_addr = f"tcp://[{ip}]:{port}"
        else:
            port = self._inbox.bind_to_random_port("tcp://0.0.0.0")
            self._inbox_addr = f"tcp://{ip}:{port}"
        self._open_mori(ip)

    def _open_mori(self, ip: str):
        """A new MoRI engine (fresh key) with the region registered in it."""
        from mori.io import BackendType, IOEngine, IOEngineConfig, RdmaBackendConfig

        key = f"verl-mori-{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        engine = IOEngine(key=key, config=IOEngineConfig(host=ip, port=0))
        engine.create_backend(
            BackendType.RDMA,
            RdmaBackendConfig(
                qp_per_transfer=self.qp_per_transfer,
                enable_notification=True,
                enable_transfer_chunking=self.enable_chunking,
                num_nics_per_transfer=self.num_nics_per_transfer,
            ),
        )
        self._xgmi = False
        if self.intra_node == "xgmi":
            # with both backends MoRI sends same-node GPU pairs over XGMI and everything else over RDMA
            try:
                from mori.io import XgmiBackendConfig

                engine.create_backend(BackendType.XGMI, XgmiBackendConfig())
                self._xgmi = True
            except Exception as e:
                logger.warning(f"mori: XGMI backend unavailable ({e!r}); intra-node copies go over RDMA")
        self._mem = engine.register_torch_tensor(self._region)
        self._engine, self._key, self._ip = engine, key, ip
        self._peers, self._retired = {}, 0
        logger.info(
            f"mori engine {key} ready: inbox {self._inbox_addr}, {self.num_chains} chains x {self.num_slots} x "
            f"{self.chain_bucket >> 20} MiB"
        )

    def _replace_mori(self):
        """Destroy the MoRI engine, which releases what MoRI keeps for every peer it ever talked to, and open a new
        one. Only between syncs of a usable worker: all my writes succeeded and none of my peers writes to me."""
        old, mem, peers = self._engine, self._mem, list(self._peers.values())
        for desc in peers:
            try:
                old.deregister_remote_engine(desc)
            except Exception as e:  # best effort: the peer may be gone already
                logger.warning(f"mori: deregister_remote_engine failed: {e!r}")
        old.deregister_memory(mem)
        self._engine = self._mem = old = None
        self._open_mori(self._ip)

    def prepare(self) -> dict:
        """Create the MoRI engine and register the region on first use; return this worker's metadata."""
        self._check_usable()
        if self._engine is None:
            self._setup()
        elif self._retired > max(_MAX_RETIRED_PEERS, 2 * len(self._peers)):
            self._replace_mori()
        return {
            "engine_desc": self._engine.get_engine_desc().pack(),
            "region": self._mem.pack(),
            "inbox": self._inbox_addr,
            "num_chains": self.num_chains,
            "chain_bucket": self.chain_bucket,
            "num_slots": self.num_slots,
            "node": self._node(),
            "xgmi": self._xgmi,
        }

    def _node(self) -> str:
        if self.node_id is not None:
            return self.node_id
        try:
            import ray

            return ray.get_runtime_context().get_node_id()
        except Exception:
            return socket.gethostname()

    @classmethod
    def build_topology(cls, actor_wg_world_size: int, rollout_world_size: int, metadata: list[dict]):
        """Stripe c (one per trainer rank, P = min(num_chains, trainer ranks)) is carried across nodes by one rollout
        GPU per node, its holder: local GPU c % n on every rollout node, so each node takes one copy of the model spread
        over its NICs. The holders form a chain trainer c -> node 1 -> node 2 -> ... (node order rotated by c), and
        each holder also copies the stripe to every other GPU of its node (over XGMI when available)."""
        A, R = metadata[:actor_wg_world_size], metadata[actor_wg_world_size:]
        N = rollout_world_size
        assert len(R) == N and N > 0
        assert len({(m["chain_bucket"], m["num_slots"]) for m in metadata}) == 1, (
            "all workers need the same bucket size, num_chains and num_slots"
        )
        P = min(min(m["num_chains"] for m in metadata), actor_wg_world_size)
        nodes: dict[str, list[int]] = {}
        for r, m in enumerate(R):
            nodes.setdefault(m["node"], []).append(r)
        node_list = list(nodes.values())

        def holders(c: int) -> list[int]:  # rollout index holding stripe c on each node, in chain order
            order = node_list[c % len(node_list) :] + node_list[: c % len(node_list)]
            return [members[c % len(members)] for members in order]

        actor_roles = [[] for _ in range(actor_wg_world_size)]
        rollout_roles = [[] for _ in range(N)]
        for c in range(P):
            hs = holders(c)
            actor_roles[c].append({"c": c, "prev": None, "nexts": [R[hs[0]]]})
            for i, h in enumerate(hs):
                local = next(members for members in node_list if h in members)
                nexts = ([R[hs[i + 1]]] if i + 1 < len(hs) else []) + [R[x] for x in local if x != h]
                rollout_roles[h].append({"c": c, "prev": A[c] if i == 0 else R[hs[i - 1]], "nexts": nexts})
                for x in local:
                    if x != h:
                        rollout_roles[x].append({"c": c, "prev": R[h], "nexts": []})
        sync_id = uuid.uuid4().hex  # one id for every worker of this topology, whatever its history
        actor_wg_kwargs = {
            "rank": [a if a < P else -1 for a in range(actor_wg_world_size)],
            "num_chains": [P] * actor_wg_world_size,
            "chain_roles": actor_roles,
            "sync_id": [sync_id] * actor_wg_world_size,
        }
        rollout_kwargs = {
            "rank": [P + r for r in range(N)],  # log label only; trainer stripe leaders are 0..P-1
            "num_chains": [P] * N,
            "chain_roles": rollout_roles,
            "sync_id": [sync_id] * N,
        }
        return actor_wg_kwargs, rollout_kwargs

    def _connect(self, meta: dict | None):
        from mori.io import EngineDesc, MemoryDesc

        if meta is None:
            return None
        desc = EngineDesc.unpack(meta["engine_desc"])
        if desc.key not in self._peers:
            self._engine.register_remote_engine(desc)
            self._peers[desc.key] = desc
        sock = self._push.get(meta["inbox"])
        if sock is None:
            sock = self._zctx.socket(zmq.PUSH)
            if meta["inbox"].startswith("tcp://["):
                sock.setsockopt(zmq.IPV6, 1)
            sock.connect(meta["inbox"])
            self._push[meta["inbox"]] = sock
        # MoRI routes a write over XGMI when both engines have the XGMI backend and sit on the same node; XGMI writes
        # raise no target-side notification (XgmiBackend::PopInboundTransferStatus always returns false), so the
        # receiver of such a hop relies on the sender announcing the bucket only after its write completed.
        xgmi_hop = self._xgmi and meta.get("xgmi", False) and meta.get("node") == self._node()
        return SimpleNamespace(
            key=desc.key,
            inbox=meta["inbox"],
            sock=sock,
            region=MemoryDesc.unpack(meta["region"]),
            notifies=not xgmi_hop,
        )

    def init_process_group(self, rank: int, num_chains: int, chain_roles: list[dict], sync_id: str):
        """Connect to my previous hop and next hops in every stripe I am in. Connections to peers that are still in
        the topology are kept across syncs; connections to peers that left are released."""
        self._check_usable()
        self.rank, self._chain_rank, self._P = rank, rank, num_chains
        self._sync_id, self._round = sync_id, 0
        self._roles = {}
        for role in chain_roles:
            self._roles[role["c"]] = SimpleNamespace(
                prev=self._connect(role["prev"]), nexts=[self._connect(m) for m in role["nexts"]]
            )
        self._retire_stale_peers()
        # No write happens here (MoRI builds QPs / IPC mappings lazily on the first write of a sync): a write outside
        # a sync could fail with nobody on the target side guarding its region.

    def _retire_stale_peers(self):
        """Drop the engine descriptor and the socket of every peer that is not in the current topology (a rollout
        worker that was removed or replaced) and count it: MoRI keeps its endpoints until `_replace_mori`. Safe here:
        a usable worker's previous sync ended with all its writes succeeded."""
        hops = [p for ro in self._roles.values() for p in (ro.prev, *ro.nexts) if p is not None]
        keys, inboxes = {p.key for p in hops}, {p.inbox for p in hops}
        for key in [k for k in self._peers if k not in keys]:
            try:
                self._engine.deregister_remote_engine(self._peers.pop(key))
            except Exception as e:  # best effort: the peer may be gone already
                logger.warning(f"mori: deregister_remote_engine({key}) failed: {e!r}")
            self._retired += 1
        for addr in [a for a in self._push if a not in inboxes]:
            self._push.pop(addr).close(linger=0)

    def finalize(self):
        """Keep the region, registrations, QPs and sockets for the next sync; only forget this sync's positions."""
        self.rank = None
        self._roles = {}

    def close(self):
        """Release sockets, the ZMQ context and the MoRI registrations. Not part of the per-sync API (`finalize` keeps
        everything for the next sync); call it when the engine is torn down for good, e.g. in tests."""
        for sock in self._push.values():
            sock.close(linger=0)
        self._push.clear()
        if self._inbox is not None:
            self._inbox.close(linger=0)
            self._inbox = None
        if self._zctx is not None:
            self._zctx.term()
            self._zctx = None
        if self._engine is not None and self._unusable is None:  # an unusable engine stays alive (_KEEPALIVE)
            for desc in self._peers.values():
                try:
                    self._engine.deregister_remote_engine(desc)
                except Exception as e:  # best effort at teardown
                    logger.warning(f"mori: deregister_remote_engine failed at close: {e!r}")
            if self._mem is not None:
                try:
                    self._engine.deregister_memory(self._mem)
                except Exception as e:  # best effort at teardown
                    logger.warning(f"mori: deregister_memory failed at close: {e!r}")
            self._engine = self._mem = self._region = self._owner = None
        self._peers.clear()

    # ------------------------------------------------------------------ in-flight writes
    def _check_usable(self):
        if self._unusable is not None:
            raise RuntimeError(f"mori: engine unusable after a failed sync ({self._unusable}); recreate the workers")
        if self._busy:
            raise RuntimeError("mori: the previous sync of this worker has not finished")

    def _mark_unusable(self, reason: str):
        """Fail-stop: from now on nothing of this engine is reused or freed (see the module docstring)."""
        if self._unusable is None:
            self._unusable = reason
            _KEEPALIVE.append((self._engine, self._mem, self._region, self._owner, self._inflight))
            logger.error(f"mori: engine {self._key} unusable: {reason}")

    def _track(self, st):
        self._inflight.add(st)
        return st

    def _check_drained(self):
        """End of a successful sync: every write I submitted was awaited until it succeeded."""
        if self._inflight:
            self._mark_unusable(f"{len(self._inflight)} MoRI writes not finished at the end of a sync")
            raise RuntimeError(f"mori: {self._unusable}")

    # ------------------------------------------------------------------ control plane
    def _start_pump(self, seq: tuple):
        """Single reader of the inbox: bucket messages -> per-chain queues, free messages -> per-slot credits."""
        self._buckets = {c: asyncio.Queue() for c in self._roles}
        self._abort: str | None = None
        self._credit = {
            (c, j, n.key): asyncio.Event()
            for c, ro in self._roles.items()
            for n in ro.nexts
            for j in range(self.num_slots)
        }
        # no credit yet: a next hop grants its slots once it is receiving (see _receive), so nothing is ever written
        # into a worker that has not entered this sync

        async def pump():
            while True:
                msg = await self._inbox.recv_pyobj()
                if msg.get("seq") != seq:
                    logger.warning(f"mori: dropping message from sync {msg.get('seq')} during sync {seq}: {msg['t']}")
                    continue
                if msg["t"] == "bucket":
                    self._buckets[msg["c"]].put_nowait(msg)
                elif msg["t"] == "free":
                    self._credit[(msg["c"], msg["slot"], msg["from"])].set()
                elif msg["t"] == "abort":  # wake every waiter; they re-check self._abort
                    self._abort = msg["reason"]
                    for e in self._credit.values():
                        e.set()
                    for q in self._buckets.values():
                        q.put_nowait(msg)

        self._pump = asyncio.ensure_future(pump())

    def _check_abort(self):
        if self._abort is not None:
            raise RuntimeError(f"mori: aborted by a peer in the chain: {self._abort}")

    async def _notify_abort(self, seq: tuple, err: BaseException):
        """Best effort: tell every neighbour this sync failed so it fails now instead of after timeout_s."""
        if isinstance(err, RuntimeError) and "aborted by a peer" in str(err):
            reason = self._abort  # pass the original reason along
        else:
            reason = f"{self._key} (rank {self._chain_rank}): {type(err).__name__}: {err}"
        peers = {p.sock for ro in self._roles.values() for p in (ro.prev, *ro.nexts) if p is not None}
        for sock in peers:
            try:
                await sock.send_pyobj({"t": "abort", "seq": seq, "reason": reason}, flags=zmq.NOBLOCK)
            except Exception:
                pass

    @staticmethod
    async def _reap(tasks):
        """Cancel unfinished tasks and collect every task's exception so none is reported as 'never retrieved'."""
        for t in tasks:
            if t is None:
                continue
            if not t.done():
                t.cancel()
            try:
                await t
            except BaseException:
                pass

    async def _stop_pump(self):
        self._pump.cancel()
        try:
            await self._pump
        except asyncio.CancelledError:
            pass

    async def _timed(self, aw, what: str):
        try:
            return await asyncio.wait_for(aw, self.timeout_s)
        except asyncio.TimeoutError:
            failure = self._engine.pop_peer_failure() if self._engine is not None else None
            raise TimeoutError(
                f"mori: timed out after {self.timeout_s}s waiting for {what}; peer failure: {failure}"
            ) from None

    async def _wait_status(self, st, what: str):
        async def poll():
            while st.InProgress():
                await asyncio.sleep(0.0002)

        await self._timed(poll(), what)
        if not st.Succeeded():
            raise RuntimeError(f"mori: {what} failed: {st.Message()}")

    async def _wait_inbound(self, prev_key: str, uid: int, what: str):
        async def poll():
            while True:
                st = self._engine.pop_inbound_transfer_status(prev_key, uid)
                if st is not None:
                    return st
                await asyncio.sleep(0.0002)

        st = await self._timed(poll(), what)
        if st.Failed():
            raise RuntimeError(f"mori: {what} reported failure: {st.Message()}")

    async def _write_next(self, seq, c, k, j, metas, used, is_last, steps, ready=None):
        """Write my slot (c, j) into the same slot of every next hop (in parallel) and announce it to each.

        `ready` is a device event recorded after the packing copies; the NIC must not read the slot before it."""
        if ready is not None:
            while not ready.query():
                await asyncio.sleep(0.0001)
        msg = {
            "t": "bucket",
            "seq": seq,
            "c": c,
            "k": k,
            "slot": j,
            "metas": metas,
            "used": used,
            "is_last": is_last,
            "global_steps": steps,
        }

        async def one(nxt):
            await self._timed(
                self._credit[(c, j, nxt.key)].wait(), f"{nxt.key} to free slot {j} of stripe {c} (bucket {k})"
            )
            self._check_abort()
            self._credit[(c, j, nxt.key)].clear()
            uid = None
            if used > 0:
                uid = self._engine.allocate_transfer_uid()
                st = self._track(self._engine.write(self._mem, self._off(c, j), nxt.region, self._off(c, j), used, uid))
                await self._wait_status(st, f"write of bucket {k} of stripe {c} ({used} B) to {nxt.key}")
                self._inflight.discard(st)
            await nxt.sock.send_pyobj({**msg, "uid": uid, "notified": nxt.notifies})

        tasks = [asyncio.ensure_future(one(n)) for n in self._roles[c].nexts]
        try:
            await asyncio.gather(*tasks)
        finally:
            await self._reap(tasks)

    async def _drain_credits(self, c: int, what: str):
        for n in self._roles[c].nexts:
            for j in range(self.num_slots):
                await self._timed(
                    self._credit[(c, j, n.key)].wait(), f"{what}: {n.key} to release slot {j} of stripe {c}"
                )
        self._check_abort()

    async def _release(self, seq, c, j, forward, consumed):
        """Free my slot (c, j) at my previous hop once `forward` (my writes of it to my next hops) completed and the
        `consumed` event (recorded after the consumer's reads of it) fired."""
        if forward is not None:
            await forward
        if consumed is not None:
            while not consumed.query():
                await asyncio.sleep(0.0001)
        self._check_abort()
        await self._roles[c].prev.sock.send_pyobj({"t": "free", "seq": seq, "c": c, "slot": j, "from": self._key})

    def _mark_ready(self):
        """Event after the copies queued so far on the current stream (the NIC may read the slot once it fires)."""
        if self.device != "cuda":
            return None
        ev = torch.cuda.Event()
        ev.record()
        return ev

    # ------------------------------------------------------------------ data plane
    @torch.no_grad()
    async def send_weights(
        self,
        weights: Generator[tuple[str, torch.Tensor], None, None],
        global_steps: int | None = None,
    ):
        """Every actor rank must iterate all weights (the trainer's collectives run inside the generator); the leader
        of stripe c packs and pushes the tensors assigned to stripe c."""
        assert self.rank is not None, "init_process_group was not called"
        self._check_usable()
        self._busy = True
        self._round += 1
        try:
            if not self._roles:  # not leading a stripe: only drive the generator
                async for _ in ensure_async_iterator(weights):
                    pass
            else:
                await self._push_stripe(self._stripe_pieces(weights), global_steps)
        except BaseException as e:  # incl. cancellation: nothing of this engine may be reused (fail-stop)
            self._mark_unusable(f"send failed: {type(e).__name__}: {e}")
            raise
        self._busy = False

    async def _stripe_pieces(self, weights) -> AsyncGenerator[tuple[TensorMeta, torch.Tensor], None]:
        """Chunks of the tensors of my stripe. Each tensor goes to the least-loaded stripe so far; every actor rank sees
        the same tensors in the same order, so all compute the same assignment."""
        (c,) = self._roles.keys()
        loads, mine = [0] * self._P, False
        async for tensor_meta, chunk in split_weight_chunks(weights, self.chain_bucket):
            if tensor_meta.chunk_offset == 0:  # a new tensor
                to = min(range(self._P), key=loads.__getitem__)
                loads[to] += tensor_meta.shape.numel() * tensor_meta.dtype.itemsize
                mine = to == c
            if mine:
                yield tensor_meta, chunk

    async def _push_stripe(self, pieces, global_steps):
        """Pack the pieces of my stripe into my slots and push each full bucket down the stripe."""
        (c,) = self._roles.keys()
        seq = (self._sync_id, self._round)
        self._start_pump(seq)
        cb = self.chain_bucket
        start, nbytes, k, j, offset = time.time(), 0, 0, 0, 0
        metas: list[TensorMeta] = []
        writing: list[asyncio.Future | None] = [None] * self.num_slots  # write task reading from my slot (c, j)
        try:
            async for tensor_meta, chunk in pieces:
                at = _align(offset)
                if at + tensor_meta.chunk_size > cb:
                    # the write task waits for an event after the packing copies, so packing the next bucket overlaps
                    # with the copy and the transfer of this one
                    writing[j] = asyncio.ensure_future(
                        self._write_next(seq, c, k, j, metas, offset, False, global_steps, self._mark_ready())
                    )
                    nbytes += offset
                    k, j, offset, at, metas = k + 1, (k + 1) % self.num_slots, 0, 0, []
                    if writing[j] is not None:  # my slot (c, j) is still being read by an earlier write
                        await writing[j]
                tensor_meta.offset = at
                metas.append(tensor_meta)
                self._slot(c, j)[at : at + tensor_meta.chunk_size].copy_(chunk, non_blocking=True)
                offset = at + tensor_meta.chunk_size
            writing[j] = asyncio.ensure_future(
                self._write_next(seq, c, k, j, metas, offset, True, global_steps, self._mark_ready())
            )
            nbytes += offset
            for w in writing:
                if w is not None:
                    await w
            await self._drain_credits(c, "send_weights")
        except BaseException as e:
            self._mark_unusable(f"send failed: {type(e).__name__}: {e}")  # first, before anything can be cancelled
            await self._notify_abort(seq, e)
            raise
        finally:
            await self._reap(writing)
            await self._stop_pump()
        self._check_drained()
        dt = time.time() - start
        logger.info(
            f"mori stripe {c}: sent {k + 1} buckets, {nbytes / 2**30:.2f} GiB in {dt:.2f}s "
            f"({nbytes / max(dt, 1e-9) / 2**30:.2f} GiB/s)"
        )

    def receive_weights(self, global_steps: int | None = None) -> AsyncIterator[tuple[str, torch.Tensor]]:
        """Receive every stripe, forward each bucket down its stripe, and yield (name, tensor) one at a time.

        The sync starts at this call, not at the first iteration: the worker is busy from here on, and closing the
        returned stream before its first tensor (a server adapter that fails while setting up) still ends the sync
        as failed and tells the peers."""
        assert self.rank is not None and self._roles, "init_process_group was not called"
        self._check_usable()
        self._busy = True
        self._round += 1
        return _ReceiveStream(self, (self._sync_id, self._round), global_steps)

    @torch.no_grad()
    async def _receive(self, seq, global_steps) -> AsyncGenerator[tuple[str, torch.Tensor], None]:
        self._start_pump(seq)
        start = time.time()
        try:  # I am guarded now (busy, failure path below): let my previous hops write into my slots
            for c, ro in self._roles.items():
                for j in range(self.num_slots):
                    await ro.prev.sock.send_pyobj({"t": "free", "seq": seq, "c": c, "slot": j, "from": self._key})
        except BaseException as e:
            self._mark_unusable(f"receive failed: {type(e).__name__}: {e}")
            await self._notify_abort(seq, e)
            await self._stop_pump()
            raise
        acc = {"buckets": 0, "bytes": 0}
        handoff: asyncio.Queue = asyncio.Queue()

        async def run_chain(c):
            try:
                async for name, weight in merge_weight_chunks(
                    self._receive_chain(c, seq, global_steps, acc), self.chain_bucket
                ):
                    taken = asyncio.Event()
                    await handoff.put(("item", name, weight, taken))
                    await taken.wait()  # the consumer asked for the next tensor: it is done with this one
                await handoff.put(("done", c, None, None))
            except asyncio.CancelledError:
                raise
            except BaseException as e:
                await handoff.put(("error", c, e, None))

        tasks = [asyncio.ensure_future(run_chain(c)) for c in self._roles]
        remaining = len(tasks)
        try:
            while remaining:
                kind, a, b, taken = await handoff.get()
                if kind == "item":
                    yield a, b
                    taken.set()
                elif kind == "done":
                    remaining -= 1
                else:
                    raise b
        except BaseException as e:  # incl. GeneratorExit (the consumer stopped mid-sync) and cancellation
            # first, before anything can be cancelled: a consumer may still read a slot on the GPU, MoRI writes may
            # still run, so nothing of this engine may be reused (fail-stop)
            self._mark_unusable(f"receive failed: {type(e).__name__}: {e}")
            await self._notify_abort(seq, e)
            raise
        finally:
            await self._reap(tasks)
            await self._stop_pump()
        self._check_drained()
        self._busy = False
        logger.info(
            f"mori rank {self.rank} received {acc['buckets']} buckets from {len(self._roles)} stripes, "
            f"{acc['bytes'] / 2**30:.2f} GiB in {time.time() - start:.2f}s"
        )

    async def _receive_chain(self, c, seq, global_steps, acc) -> AsyncGenerator[tuple[TensorMeta, torch.Tensor], None]:
        ro = self._roles[c]
        k, early = 0, {}
        # per slot: the task that frees it once its forward writes have left and the consumer is done with it; the
        # loop moves on to the next bucket meanwhile, so up to num_slots buckets per stripe are in flight here
        release: list[asyncio.Future | None] = [None] * self.num_slots
        forward = None
        try:
            while True:
                while k not in early:
                    msg = await self._timed(self._buckets[c].get(), f"bucket {k} of stripe {c} from {ro.prev.key}")
                    self._check_abort()
                    early[msg["k"]] = msg
                msg = early.pop(k)
                steps = msg["global_steps"]
                if global_steps is not None and steps is not None and steps != global_steps:
                    raise RuntimeError(f"mori: got weights of step {steps} while expecting step {global_steps}")
                j = msg["slot"]
                assert j == k % self.num_slots, f"bucket {k} of stripe {c} arrived in slot {j}"
                for r in release:  # surface a failed forward or free early
                    if r is not None and r.done():
                        r.result()
                if release[j] is not None:  # already done: prev wrote bucket k into slot j only after it got the free
                    await release[j]
                    release[j] = None
                if msg["uid"] is not None and msg.get("notified", True):  # RDMA hop; XGMI hops raise no notification
                    await self._wait_inbound(ro.prev.key, msg["uid"], f"arrival of bucket {k} of stripe {c}")
                forward = None
                if ro.nexts:  # next node's holder and/or the other GPUs of my node
                    forward = asyncio.ensure_future(
                        self._write_next(seq, c, k, j, msg["metas"], msg["used"], msg["is_last"], steps)
                    )
                buf = self._slot(c, j)
                for tensor_meta in msg["metas"]:
                    yield tensor_meta, buf[tensor_meta.offset : tensor_meta.offset + tensor_meta.chunk_size]
                # the consumer asked for the next tensor, so its reads of this bucket's views are queued on the
                # current stream: free the slot once they ran and the forward writes have left
                release[j] = asyncio.ensure_future(self._release(seq, c, j, forward, self._mark_ready()))
                forward = None
                acc["buckets"] += 1
                acc["bytes"] += msg["used"]
                if msg["is_last"]:
                    break
                k += 1
            for j, r in enumerate(release):
                if r is not None:
                    await r
                    release[j] = None
            if ro.nexts:
                await self._drain_credits(c, "receive_weights")
        finally:
            await self._reap([forward, *release])
