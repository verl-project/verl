# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
"""``sharded_nixl``: ownership-aware trainer -> rollout weight sync over NIXL WRITE.

Every trainer rank exports only the shards it holds (``engine.get_per_tensor_param_shard`` or,
for Megatron, ``get_per_tensor_param_hf_shard``) and WRITEs them into each generator's
registered bucket buffers; there is no trainer-side all-gather. Each generator reports its
``RolloutLayout`` and receives only the parameters it loads. The bucket layout follows from the
rank-identical export order, so it is built once and cached on the generators by digest.
Planning lives in ``sharded_nixl_layout``.

Per-sync protocol (S = sync id, NB staging buffers per worker)::

    generator g -> every trainer:   hello{S, rollout layout, cached layout/plan digests}   (ZMQ)
    trainer 0   -> every generator: sync{S, layout digest, num_buckets [, full layout]}     (ZMQ)
    trainer r   -> generator g:     plan{S, plan key, tables[k] = [(dst, len)...]}  if g lacks it (ZMQ)
    for bucket k:
        trainer r: wait ACK(S, k-NB) from every g; pack own slices into send_buf[k % NB];
                   NIXL WRITE slices -> g.recv_buf[k % NB] with notif SNX{S,k,r,hashes}
        generator: wait SNX from all trainers; verify hashes; yield tensors of bucket k;
                   send_notif ACK{S,k} to every trainer

When every generator advertises ``rank_local_named_tensors`` with slice tables of the same local
shapes, the sync uses a second layout built from those local (box) shapes and each generator
receives only its TP slice of each tensor.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import struct
import time
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Optional
from unittest.mock import patch

import torch
import zmq

from verl.checkpoint_engine.base import CheckpointEngine, CheckpointEngineRegistry, TensorMeta, merge_weight_chunks
from verl.checkpoint_engine.nixl_checkpoint_engine import NixlAgent, NixlAgentMetadata
from verl.checkpoint_engine.sharded_nixl_layout import (
    DEFAULT_ALIGN,
    BoxWriteSlice,
    BucketLayout,
    LayoutBuilder,
    ParamEntry,
    WriteSlice,
    align_up,
    contribution_range,
    expand_fused_moe_entry,
    index_runs,
    local_entry,
    local_shapes_key,
    merge_box_plans,
    normalize_slices,
    owned_name_filter,
    plan_indexed_writes,
    plan_rank_local_indexed_writes,
    plan_rank_local_writes,
    plan_writes,
    slices_digest,
)
from verl.workers.rollout.base import RolloutLayout

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_SNX_HDR = struct.Struct("<QIII")  # sync, bucket, trainer rank, n_hashes_inline (0xFFFFFFFF: sent over ZMQ)
_ACK_HDR = struct.Struct("<QI")  # sync, bucket
_HASH = struct.Struct("<Q")

NAMED = "named_tensors"
RANK_LOCAL = "rank_local_named_tensors"


@dataclass
class ShardedNixlWorkerMeta:
    """What a worker advertises to its peers from ``prepare``."""

    agent: NixlAgentMetadata
    buffers: list[tuple[int, int, int]]  # (addr, nbytes, device id) per staging buffer
    mem_type: str  # "VRAM" | "DRAM"
    bucket_size: int
    num_buffers: int


@dataclass
class _Peer:
    """Per-peer state kept across syncs: a trainer's view of a generator, or a generator's view
    of a trainer."""

    meta: ShardedNixlWorkerMeta
    name: str
    rank: int
    layout: Optional[RolloutLayout] = None
    owns: Any = None
    plan_key: str = ""
    # bucket -> [(dst, len)] (named_tensors) or [(dst, len, box_key)] (rank_local_named_tensors:
    # TP ranks share (dst, len) in the common rank-local layout but get different bytes)
    tables: dict[int, list[tuple]] = field(default_factory=dict)
    tables_key: str = ""
    known_digests: set = field(default_factory=set)
    known_plan_keys: dict[str, str] = field(default_factory=dict)


def _hash_u8_many(t: torch.Tensor, ranges: list[tuple[int, int]]) -> list[int]:
    """One hash per ``(offset, length)`` byte range of ``t``: the XOR of the range's little-endian
    8-byte words, the tail zero-padded, whatever the range's alignment in ``t``.

    ``hash_tensor`` XORs elements, so over bytes it is only 8 bits wide; each byte lane of the
    words is reduced separately instead.
    """
    word_rows, owners = [], []
    zero_word = t.new_zeros(8)
    for i, (offset, length) in enumerate(ranges):
        body = length - length % 8
        pieces = [t[offset : offset + body]] if body else []
        if body < length:
            pieces.append(torch.cat([t[offset + body : offset + length], zero_word[: 8 - (length - body)]]))
        for piece in pieces:
            word_rows.append(torch.hash_tensor(piece.view(-1, 8), dim=0))
            owners.append(i)
    hashes = [0] * len(ranges)
    if word_rows:
        for i, lanes in zip(owners, torch.stack(word_rows).tolist(), strict=True):
            hashes[i] ^= sum(int(lane) << (8 * j) for j, lane in enumerate(lanes))
    return hashes


@CheckpointEngineRegistry.register("sharded_nixl")
class ShardedNixlCheckpointEngine(CheckpointEngine):
    """See module docstring.

    Args:
        bucket_size: staging bucket size in bytes.
        is_master: rank 0 of the trainer group.
        device: ``"cuda"`` (GPU staging, GPUDirect RDMA) or ``"cpu"`` (pinned host staging).
        rollout_dtype: wire dtype for floating point tensors.
        num_buffers: staging ring depth; must match on both sides.
        align: byte alignment of tensors inside a bucket.
        ownership: ``"adapter"`` (filter by the rollout worker's ``get_weight_ownership``) or
            ``"all"`` (send every parameter to every generator).
        notif_max_bytes: inline the per-slice hash table in the NIXL notification up to this
            size; larger tables go over ZMQ.
        ack_timeout_s / notif_timeout_s: fail instead of hanging.
    """

    wire_format = "named_tensors"
    send_weights_takes_engine = True

    def __init__(
        self,
        bucket_size: int,
        is_master: bool = False,
        device: str = "cuda",
        rollout_dtype: torch.dtype = torch.bfloat16,
        num_buffers: int = 2,
        align: int = DEFAULT_ALIGN,
        ownership: str = "adapter",
        notif_max_bytes: int = 32 << 10,
        ack_timeout_s: float = 600.0,
        notif_timeout_s: float = 600.0,
        _debug_corrupt_slice: bool = False,
        _debug_sync_offset: int = 0,
    ) -> None:
        assert num_buffers >= 2, "sharded_nixl needs at least 2 staging buffers"
        assert ownership in {"all", "adapter"}, ownership
        self.bucket_size = int(bucket_size)
        self.is_master = is_master
        self.device = device
        self.mem_type = "VRAM" if device == "cuda" else "DRAM"
        # engine_kwargs from yaml carry dtypes as strings
        self.rollout_dtype = (
            getattr(torch, rollout_dtype.removeprefix("torch.")) if isinstance(rollout_dtype, str) else rollout_dtype
        )
        self.num_buffers = int(num_buffers)
        self.align = int(align)
        self.ownership = ownership
        self.notif_max_bytes = int(notif_max_bytes)
        self.ack_timeout_s = float(ack_timeout_s)
        self.notif_timeout_s = float(notif_timeout_s)
        self._debug_corrupt_slice = bool(_debug_corrupt_slice)  # test hook: flips one byte after hashing, once
        self._debug_sync_offset = int(_debug_sync_offset)  # test hook: sync-id mismatch between peers
        self.needs_rollout_layout = ownership == "adapter"

        # kept across syncs
        self.agent: Optional[NixlAgent] = None
        self.bufs: list[torch.Tensor] = []
        self._buf_keepalive: list[Any] = []  # cupy arrays behind self.bufs
        self.reg_descs: list[Any] = []
        self.buf_descs: list[tuple[int, int, int]] = []
        self.peers: dict[str, _Peer] = {}
        self.peer_order: list[str] = []
        self.role: Optional[str] = None
        self.rank: Optional[int] = None
        self.world_size: Optional[int] = None
        self.num_trainers: Optional[int] = None
        self._layout: Optional[BucketLayout] = None
        self._plan_cache: dict[str, tuple[Optional[tuple[int, int]], list[WriteSlice]]] = {}  # src -> (range, writes)
        self._entries_by_src: dict[str, list[ParamEntry]] = {}
        self._uses_hf_shard_export = False
        self._hf_runs: dict[str, list[list[tuple[int, int, int]]]] = {}  # src_name -> index_runs per entry
        self._sync_counter = 0
        self._rollout_layout: Optional[RolloutLayout] = None
        self._gen_plans: dict[tuple[str, str], dict[int, list[tuple[int, int]]]] = {}  # (trainer, plan_key) -> tables
        # rank_local_named_tensors, trainer side: local-shapes key -> layout; plan_key -> this rank's
        # {src_name: [BoxWriteSlice]}; the deduped union for the current generator set
        self._rl_layouts: dict[str, BucketLayout] = {}
        self._rl_plans: dict[str, dict[str, list[BoxWriteSlice]]] = {}
        self._rl_merged: Optional[tuple[tuple, dict[str, list[BoxWriteSlice]], int]] = None
        self.active_wire_format = NAMED
        # per sync
        self._sync_task: Optional[asyncio.Task] = None  # the running send_weights / receive
        self._inflight: dict[int, list[Any]] = {}  # bucket -> NIXL transfer handles
        self._notifs: dict[tuple, bytes] = {}  # (peer, kind, sync, bucket) -> parked notification

    # ------------------------------------------------------------------ lifecycle
    def prepare(self) -> ShardedNixlWorkerMeta:
        """Creates the agent and registers the staging buffers on the first call only."""
        if self.agent is None:
            self.agent = NixlAgent()
        else:
            self._drop_stale_traffic()
        if not self.bufs:
            self._alloc_buffers(self.bucket_size)
        return ShardedNixlWorkerMeta(
            agent=self.agent.get_agent_metadata(),
            buffers=list(self.buf_descs),
            mem_type=self.mem_type,
            bucket_size=self.bucket_size,
            num_buffers=self.num_buffers,
        )

    def _drop_stale_traffic(self):
        """Drop what a failed sync left behind: its send/receive if still running, transfer handles,
        ZMQ messages, NIXL notifications.

        The manager returns as soon as one side fails, so the other side may still be waiting; left
        running, it would race the retry for its notifications. Otherwise the next sync reads the old
        messages first, and a retry of the same sync id can match an old ACK and overwrite a slot the
        generator still reads. The manager runs every worker's ``prepare`` before any sync traffic,
        so nothing current is dropped.
        """
        # prepare runs on the actor's event loop, so the old task cannot poll again before it unwinds
        if self._sync_task is not None and not self._sync_task.done():
            self._sync_task.cancel()
        self.finalize()
        sock = zmq.Socket.shadow(self.agent.socket.underlying)
        while True:
            try:
                sock.recv(zmq.NOBLOCK)
            except zmq.Again:
                break
        self.agent.messages.clear()
        self.agent.get_new_notifs()

    def _alloc_buffers(self, nbytes: int):
        for _ in range(self.num_buffers):
            if self.device == "cuda":
                with patch("importlib.metadata.distributions", return_value=[]):
                    import cupy as cp
                # cupy allocation sidesteps the registration error torch's
                # expandable_segments allocator causes (same trick as the nixl engine).
                cp_buf = cp.zeros(nbytes, dtype=cp.uint8)
                buf = torch.as_tensor(cp_buf, dtype=torch.uint8)
                self._buf_keepalive.append(cp_buf)
            else:
                buf = torch.zeros(nbytes, dtype=torch.uint8, device="cpu", pin_memory=True)
            self.bufs.append(buf)
            self.reg_descs.append(self.agent.register_memory(buf))
            dev = buf.get_device() if buf.is_cuda else 0
            self.buf_descs.append((buf.data_ptr(), nbytes, dev))

    def _free_buffers(self):
        if self.agent is not None:
            for rd in self.reg_descs:
                try:
                    self.agent.deregister_memory(rd)
                except Exception:  # noqa: BLE001
                    pass
        self.reg_descs, self.bufs, self.buf_descs, self._buf_keepalive = [], [], [], []

    def _ensure_send_capacity(self, nbytes: int):
        """Trainer only: grow the staging buffers to ``nbytes``.

        In rank-local mode a trainer packs one copy per distinct box, so with T trainers and a
        TP=t replica a bucket needs up to ~ceil(t / T) x bucket_size of staging. Generators never
        address trainer buffers, so the advertised ``bucket_size`` is unchanged.
        """
        if self.bufs and self.bufs[0].numel() >= nbytes:
            return
        assert not self._inflight, "sharded_nixl: cannot resize staging buffers with transfers in flight"
        new = align_up(nbytes, 1 << 20)
        old = self.bufs[0].numel() if self.bufs else 0
        logger.warning(f"sharded_nixl: growing trainer staging buffers {old} -> {new} bytes (rank-local)")
        self._free_buffers()
        self._alloc_buffers(new)

    @classmethod
    def build_topology(cls, actor_wg_world_size: int, rollout_world_size: int, metadata: list[ShardedNixlWorkerMeta]):
        """Full bipartite topology: every trainer gets every generator's metadata and vice versa."""
        T, G = actor_wg_world_size, rollout_world_size
        trainer_meta, rollout_meta = list(metadata[:T]), list(metadata[T : T + G])
        actor_wg_kwargs = {
            "role": ["trainer"] * T,
            "rank": list(range(T)),
            "world_size": [T] * T,
            "peers": [rollout_meta] * T,
            "num_trainers": [T] * T,
        }
        rollout_kwargs = {
            "role": ["rollout"] * G,
            "rank": list(range(G)),
            "world_size": [G] * G,
            "peers": [trainer_meta] * G,
            "num_trainers": [T] * G,
        }
        return actor_wg_kwargs, rollout_kwargs

    def init_process_group(
        self, role: str, rank: int, world_size: int, peers: list[ShardedNixlWorkerMeta], num_trainers: int
    ):
        """Adds new peers and drops vanished ones; existing connections are kept."""
        self.role, self.rank, self.world_size, self.num_trainers = role, rank, world_size, num_trainers
        new_order = []
        seen = set()
        for i, meta in enumerate(peers):
            name = meta.agent.agent_name
            seen.add(name)
            new_order.append(name)
            if name in self.peers:
                self.peers[name].meta = meta
                self.peers[name].rank = i
            else:
                self.agent.add_remote_agent(meta.agent)
                self.peers[name] = _Peer(meta=meta, name=name, rank=i)
            if meta.num_buffers != self.num_buffers or meta.bucket_size != self.bucket_size:
                raise ValueError(
                    f"sharded_nixl: peer {name} has num_buffers={meta.num_buffers}, bucket_size={meta.bucket_size}; "
                    f"expected {self.num_buffers}/{self.bucket_size} (engine_kwargs must match on both sides)"
                )
        for name in list(self.peers):
            if name not in seen:
                self.agent.remove_remote_agent(name)
                del self.peers[name]
        self.peer_order = new_order
        logger.info(f"sharded_nixl init_process_group role={role} rank={rank}/{world_size} peers={len(self.peers)}")

    def finalize(self):
        for hs in self._inflight.values():
            for h in hs:
                try:
                    self.agent.release_xfer_handle(h)
                except Exception:  # noqa: BLE001
                    pass
        self._inflight = {}
        self._notifs = {}

    # ------------------------------------------------------------------ ownership
    def set_rollout_layout(self, layout) -> None:
        if isinstance(layout, dict):
            layout = RolloutLayout.from_dict(layout)
        self._rollout_layout = layout

    @staticmethod
    def _plan_key(layout_digest: str, rl: Optional[RolloutLayout], wire_format: str = NAMED) -> str:
        """Cache key of one generator's write plan: the layout plus that generator's ownership."""
        h = hashlib.sha256(layout_digest.encode())
        h.update(wire_format.encode())
        if rl is not None:
            d = rl.to_dict()
            slices = d.pop("slices", None)
            d["slice_digest"] = rl.slice_digest or slices_digest(slices)
            h.update(json.dumps(d, sort_keys=True, default=list).encode())
        return h.hexdigest()

    def _choose_wire_format(self, gens: list[_Peer]) -> tuple[str, Optional[str]]:
        """``(RANK_LOCAL, local-shapes key)`` iff every generator advertises it with the same local
        shapes, else ``(NAMED, None)``. Deterministic in the hellos, so all trainers agree."""
        if not gens:
            return NAMED, None
        keys = set()
        for g in gens:
            rl = g.layout
            if rl is None or RANK_LOCAL not in (rl.wire_formats or []):
                return NAMED, None
            keys.add(local_shapes_key(rl.slices))
        if len(keys) != 1:
            logger.warning("sharded_nixl: generators disagree on rank-local shapes; falling back to named_tensors")
            return NAMED, None
        return RANK_LOCAL, keys.pop()

    def _rank_local_layout(self, key: str, rl: RolloutLayout) -> BucketLayout:
        """The layout of local (box) shapes, in export order; shared by generators with ``key``."""
        lay = self._rl_layouts.get(key)
        if lay is None:
            boxes = normalize_slices(rl.slices)
            builder = LayoutBuilder(self.bucket_size, self.align)
            for entries in self._entries_by_src.values():  # insertion order == export order
                for e in entries:
                    spec = boxes.get(e.name)
                    if spec is not None and tuple(spec[0]) != tuple(e.shape):
                        raise ValueError(
                            f"sharded_nixl: slice {e.name!r} full_shape {spec[0]} != trainer shape {e.shape}"
                        )
                    builder.add(local_entry(e, spec[1] if spec is not None else None))
            lay = self._rl_layouts[key] = builder.build()
        return lay

    def _rank_local_plan(self, layout: BucketLayout, g: _Peer) -> dict[str, list[BoxWriteSlice]]:
        plan = self._rl_plans.get(g.plan_key)
        if plan is None:
            boxes = normalize_slices(g.layout.slices)
            if self._uses_hf_shard_export:
                plan = plan_rank_local_indexed_writes(self._entries_by_src, self._hf_runs, layout, boxes, g.owns)
            else:
                ranges = {src: rng for src, (rng, _w) in self._plan_cache.items()}
                plan = plan_rank_local_writes(self._entries_by_src, ranges, layout, boxes, g.owns)
            self._rl_plans[g.plan_key] = plan
        return plan

    @staticmethod
    def _box_tables(layout: BucketLayout, plan: dict[str, list[BoxWriteSlice]]) -> dict[int, list[tuple]]:
        tables: dict[int, list[tuple]] = {k: [] for k in range(layout.num_buckets)}
        for ws in plan.values():
            for w in ws:
                tables[w.bucket_id].append((w.dst_offset, w.length, w.box_key))
        for k in tables:
            tables[k].sort(key=lambda t: (t[0], t[1]))
        return tables

    # ------------------------------------------------------------------ side channel helpers
    async def _read(self, peer: str, typ: str, timeout: float) -> dict:
        """The next ZMQ message from ``peer``, which must be of type ``typ`` (messages per peer are
        FIFO, so anything else is a protocol error)."""
        try:
            msg = await asyncio.wait_for(self.agent.read_message(peer), timeout=timeout)
        except asyncio.TimeoutError as e:
            raise TimeoutError(f"sharded_nixl: timed out waiting for {typ!r} from {peer}") from e
        if msg.get("type") != typ:
            raise RuntimeError(f"sharded_nixl: expected {typ!r} from {peer}, got {msg.get('type')!r}")
        return msg

    async def _wait_notif(self, peer: str, kind: bytes, sync: int, k: int, timeout: float) -> bytes:
        """Poll NIXL notifications until ``(peer, kind, sync, k)`` arrives, parking the others."""
        key = (peer, kind, sync, k)
        t0 = time.time()
        while key not in self._notifs:
            for remote, msgs in self.agent.get_new_notifs().items():
                for m in msgs:
                    if m[:3] == b"SNX":
                        sync_k = _SNX_HDR.unpack_from(m, 3)[:2]
                    elif m[:3] == b"ACK":
                        sync_k = _ACK_HDR.unpack_from(m, 3)
                    else:
                        logger.warning(f"sharded_nixl: ignoring unknown notification {m[:16]!r} from {remote}")
                        continue
                    self._notifs[(remote, m[:3], *sync_k)] = m
            if key in self._notifs:
                break
            if time.time() - t0 > timeout:
                raise TimeoutError(f"sharded_nixl: timed out waiting for {kind!r} sync={sync} bucket={k} from {peer}")
            await asyncio.sleep(0)
        return self._notifs.pop(key)

    async def _wait_xfers_done(self, handles: list[Any], timeout: float):
        t0 = time.time()
        pending = list(handles)
        while pending:
            still = []
            for h in pending:
                st = self.agent.check_xfer_state(h)
                if st == "ERR":
                    raise RuntimeError("sharded_nixl: NIXL transfer failed (state ERR)")
                if st != "DONE":
                    still.append(h)
            pending = still
            if pending:
                if time.time() - t0 > timeout:
                    raise TimeoutError("sharded_nixl: timed out waiting for NIXL WRITE completion")
                await asyncio.sleep(0)
        for h in handles:
            self.agent.release_xfer_handle(h)

    # ------------------------------------------------------------------ trainer side
    def _iter_export(self, engine):
        gen, _ = engine.get_per_tensor_param_shard()
        return gen

    def _iter_hf_export(self, engine, with_indices: bool = True):
        gen, _ = engine.get_per_tensor_param_hf_shard(with_indices=with_indices)
        return gen

    def _model_type(self, engine) -> Optional[str]:
        try:
            return engine.model_config.hf_config.model_type
        except AttributeError:
            return None

    def _wire_dtype(self, t: torch.Tensor) -> torch.dtype:
        return self.rollout_dtype if t.is_floating_point() else t.dtype

    def _hf_entries(self, src_name: str, slots, dtype_name: str) -> list[ParamEntry]:
        """The entries of one HF-coordinate shard, in the wire dtype (the send loop casts to it)."""
        dtype = getattr(torch, dtype_name.replace("torch.", ""))
        dtype = self.rollout_dtype if dtype.is_floating_point else dtype
        return [ParamEntry(name, tuple(shape), dtype, src_name) for name, shape in slots]

    def _build_layout(self, engine) -> BucketLayout:
        """First sync: one pass over the export builds the layout and this rank's write plan."""
        if hasattr(engine, "get_per_tensor_param_hf_shard"):
            return self._build_hf_layout(engine)

        model_type = self._model_type(engine)
        builder = LayoutBuilder(self.bucket_size, self.align)
        entries_by_src: dict[str, list[ParamEntry]] = {}
        plan: dict[str, tuple[Optional[tuple[int, int]], list[WriteSlice]]] = {}
        slots_by_name: dict[str, list] = {}
        for src_name, local, spec in self._iter_export(engine):
            entries = expand_fused_moe_entry(src_name, tuple(spec.full_shape), self._wire_dtype(local), model_type)
            entries_by_src[src_name] = entries
            for e in entries:
                slots_by_name[e.name] = builder.add(e)
            rng = contribution_range(spec, local.numel(), entries[0].itemsize)
            plan[src_name] = (rng, plan_writes(slots_by_name, entries, rng) if rng is not None else [])
            del local
        layout = builder.build()
        self._entries_by_src = entries_by_src
        self._plan_cache = plan
        return layout

    def _build_hf_layout(self, engine) -> BucketLayout:
        """Build the layout from backend-converted HF-coordinate shards."""
        builder = LayoutBuilder(self.bucket_size, self.align)
        entries_by_src: dict[str, list[ParamEntry]] = {}
        for src_name, slots, dtype_name, counts, indices, values, _group in self._iter_hf_export(engine):
            entries = self._hf_entries(src_name, slots, dtype_name)
            entries_by_src[src_name] = entries
            for entry in entries:
                builder.add(entry)
            del counts, indices, values
        layout = builder.build()
        self._entries_by_src = entries_by_src
        self._plan_cache = {}
        for src_name, slots, dtype_name, counts, indices, values, _group in self._iter_hf_export(engine):
            runs = self._hf_runs[src_name] = index_runs(counts, indices)
            self._plan_cache[src_name] = (None, plan_indexed_writes(layout, entries_by_src[src_name], runs))
            del slots, dtype_name, counts, indices, values
        self._uses_hf_shard_export = True
        return layout

    def _tables_for(self, layout: BucketLayout, owns) -> dict[int, list[tuple[int, int]]]:
        """Per-bucket sorted ``(dst, len)`` of this rank's writes to a generator with ``owns``."""
        tables: dict[int, list[tuple[int, int]]] = {k: [] for k in range(layout.num_buckets)}
        for _src, (_rng, slices) in self._plan_cache.items():
            for s in slices:
                if owns(s.name):
                    tables[s.bucket_id].append((s.dst_offset, s.length))
        for k in tables:
            tables[k].sort()
        return tables

    @torch.no_grad()
    async def send_weights(self, engine, global_steps: int | None = None) -> dict[str, float] | None:
        """Send this rank's shards; metrics are returned on rank 0 only."""
        assert self.role == "trainer", f"send_weights called on role={self.role}"
        self._sync_task = asyncio.current_task()
        t_start = time.time()
        S = (global_steps if global_steps is not None else self._sync_counter) + self._debug_sync_offset
        # counted per attempt on both sides, so a sync that fails on one side only does not split them
        self._sync_counter += 1
        gens = [self.peers[n] for n in self.peer_order]
        nb = self.num_buffers

        # 0. hellos: ownership + caches of every generator
        hello_bytes = 0
        for g in gens:
            msg = await self._read(g.name, "hello", self.notif_timeout_s)
            if msg["sync"] != S:
                raise RuntimeError(f"sharded_nixl: generator {g.name} is at sync {msg['sync']}, trainer at {S}")
            g.layout = RolloutLayout.from_dict(msg.get("layout"))
            g.owns = owned_name_filter(g.layout)
            g.known_digests = set(msg.get("known_digests", []))
            g.known_plan_keys = dict(msg.get("known_plan_keys", {}))
            hello_bytes += len(json.dumps(msg, default=str))

        # 1. layout: built on the first sync, then validated against the export below
        first_sync = self._layout is None
        if first_sync:
            self._layout = self._build_layout(engine)
        full_layout = layout = self._layout
        wire_format, rl_key = self._choose_wire_format(gens)
        if self.is_master and wire_format != self.active_wire_format:
            logger.info(f"sharded_nixl: wire format {self.active_wire_format} -> {wire_format}")
        self.active_wire_format = wire_format
        if wire_format == RANK_LOCAL:
            layout = self._rank_local_layout(rl_key, gens[0].layout)

        # 2. sync header (rank 0) and this rank's write tables, for generators that lack them
        meta_bytes = 0
        my_name = self.agent.agent_name
        for g in gens:
            g.plan_key = self._plan_key(layout.digest, g.layout, wire_format)
            if self.rank == 0:
                msg = {
                    "type": "sync",
                    "sync": S,
                    "digest": layout.digest,
                    "num_buckets": layout.num_buckets,
                    "num_trainers": self.num_trainers,
                    "wire_format": wire_format,
                }
                if layout.digest not in g.known_digests:
                    msg["layout"] = layout.to_wire()
                self.agent.send_message(g.name, msg)
                meta_bytes += _approx_size(msg)

        # 3. export pass: pack and WRITE bucket by bucket. Export order == layout order on every
        #    rank, so a bucket can be flushed as soon as a slice of a later bucket shows up.
        stats = {"payload": 0, "meta": 0}
        my_dev = self.buf_descs[0][2]
        pending: list[tuple[WriteSlice | BoxWriteSlice, torch.Tensor]] = []
        for g in gens:
            if g.tables_key != g.plan_key:
                if wire_format == RANK_LOCAL:
                    g.tables = self._box_tables(layout, self._rank_local_plan(layout, g))
                else:
                    g.tables = self._tables_for(layout, g.owns)
                g.tables_key = g.plan_key
            if g.known_plan_keys.get(my_name) != g.plan_key:
                # the generator only needs (dst, len) per slice (its own box is implied)
                wire_tables = {k: [(t[0], t[1]) for t in v] for k, v in g.tables.items()}
                msg = {"type": "plan", "sync": S, "plan_key": g.plan_key, "rank": self.rank, "tables": wire_tables}
                self.agent.send_message(g.name, msg)
                stats["meta"] += _approx_size(msg)
        box_plan: dict[str, list[BoxWriteSlice]] = {}
        if wire_format == RANK_LOCAL:
            # One packed copy per distinct (bucket, dst, len, box): DP replicas and replicated k/v
            # heads share pieces. Size the staging buffers for the fullest bucket.
            mkey = tuple(g.plan_key for g in gens)
            if self._rl_merged is None or self._rl_merged[0] != mkey:
                box_plan = merge_box_plans(self._rank_local_plan(layout, g) for g in gens)
                per_bucket: dict[int, int] = {}
                for ws in box_plan.values():
                    for w in ws:
                        # + the worst-case 8-byte padding: _flush packs in export order, not this one
                        per_bucket[w.bucket_id] = per_bucket.get(w.bucket_id, 0) + w.length + 7
                self._rl_merged = (mkey, box_plan, max(per_bucket.values(), default=0))
            _mk, box_plan, need = self._rl_merged
            self._ensure_send_capacity(need)

        async def _flush(k: int, items: list[tuple[WriteSlice, torch.Tensor]]):
            """Pack this rank's part of bucket ``k`` and WRITE it to every generator."""
            slot = k % nb
            if k >= nb:
                # the slot's previous bucket must be written and ACKed by every generator
                await self._wait_xfers_done(self._inflight.pop(k - nb, []), self.notif_timeout_s)
                for g in gens:
                    await self._wait_notif(g.name, b"ACK", S, k - nb, self.ack_timeout_s)
            send_buf = self.bufs[slot]
            off = 0
            packed: dict[
                tuple, int
            ] = {}  # (dst, len) [named] or (dst, len, box_key) [rank-local] -> offset in send_buf
            for s_, u8 in items:
                if isinstance(s_, BoxWriteSlice):
                    key = (s_.dst_offset, s_.length, s_.box_key)
                    if key in packed:
                        continue
                    off = (off + 7) // 8 * 8
                    assert off + s_.length <= send_buf.numel(), (
                        "sharded_nixl: packed contributions exceed the staging buffer"
                    )
                    pos = off
                    for seg in s_.segments:
                        dst = send_buf[pos : pos + seg.nbytes]
                        if seg.nrows == 1:
                            dst.copy_(u8[seg.src_offset : seg.src_offset + seg.row_bytes], non_blocking=True)
                        else:
                            src = u8[seg.src_offset : seg.src_offset + (seg.nrows - 1) * seg.stride + seg.row_bytes]
                            src = src.as_strided((seg.nrows, seg.row_bytes), (seg.stride, 1))
                            dst.view(seg.nrows, seg.row_bytes).copy_(src, non_blocking=True)
                        pos += seg.nbytes
                    packed[key] = off
                    off += s_.length
                    continue
                off = (off + 7) // 8 * 8
                assert off + s_.length <= send_buf.numel(), "sharded_nixl: packed contributions exceed the bucket"
                send_buf[off : off + s_.length].copy_(u8[s_.src_offset : s_.src_offset + s_.length], non_blocking=True)
                packed[(s_.dst_offset, s_.length)] = off
                off += s_.length
            # the shards are on GPU even with cpu staging, so the copies above are async either way
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            hashes: dict[tuple[int, int], int] = {}
            if packed:
                keys = list(packed)
                hashes = dict(zip(keys, _hash_u8_many(send_buf, [(packed[kk], kk[1]) for kk in keys]), strict=True))
            if self._debug_corrupt_slice and self.rank == 0 and k == 0 and packed:
                self._debug_corrupt_slice = False  # one sync only, so a retry can succeed
                off0 = next(iter(packed.values()))
                send_buf[off0] ^= 0xFF
                if self.device == "cuda":
                    torch.cuda.synchronize()
            handles = []
            for g in gens:
                table = g.tables.get(k, [])
                if not table:
                    # the generator still waits for an SNX from every trainer
                    self.agent.send_notif(g.name, b"SNX" + _SNX_HDR.pack(S, k, self.rank, 0))
                    continue
                remote_buf_addr, _rb_len, remote_dev = g.meta.buffers[slot]
                local_descs = self.agent.get_xfer_descs(
                    [(self.buf_descs[slot][0] + packed[t], t[1], my_dev) for t in table], mem_type=self.mem_type
                )
                remote_descs = self.agent.get_xfer_descs(
                    [(remote_buf_addr + t[0], t[1], remote_dev) for t in table], mem_type=g.meta.mem_type
                )
                notif = self._encode_snx(S, k, table, hashes, g)
                h = self.agent.initialize_xfer("WRITE", local_descs, remote_descs, g.name, notif)
                if self.agent.transfer(h) == "ERR":
                    raise RuntimeError(f"sharded_nixl: NIXL WRITE to {g.name} failed at bucket {k}")
                handles.append(h)
                stats["payload"] += sum(t[1] for t in table)
            self._inflight[k] = handles

        model_type = self._model_type(engine)
        n_params = 0
        next_k = 0
        # the HF write plan is cached by _build_hf_layout, so the send loop needs values only
        export = (
            self._iter_hf_export(engine, with_indices=False)
            if self._uses_hf_shard_export
            else self._iter_export(engine)
        )
        for record in export:
            n_params += 1
            if self._uses_hf_shard_export:
                src_name, slots, dtype_name, counts, indices, local, _group = record
                entries = self._hf_entries(src_name, slots, dtype_name)
                cached = self._entries_by_src.get(src_name)
                if cached is None or [e.key() for e in cached] != [e.key() for e in entries]:
                    raise RuntimeError(f"sharded_nixl: HF shard export changed between syncs at {src_name!r}")
                n_idx = local.numel() if indices is None else indices.numel()
                if sum(counts.tolist()) != local.numel() or n_idx != local.numel():
                    raise RuntimeError(f"sharded_nixl: invalid HF shard export at {src_name!r}")
            else:
                src_name, local, spec = record
            if not self._uses_hf_shard_export and not first_sync:
                exp = expand_fused_moe_entry(src_name, tuple(spec.full_shape), self._wire_dtype(local), model_type)
                cached = self._entries_by_src.get(src_name)
                if cached is None or [e.key() for e in cached] != [e.key() for e in exp]:
                    raise RuntimeError(f"sharded_nixl: shard export changed between syncs at {src_name!r}")
            slices = box_plan.get(src_name, []) if wire_format == RANK_LOCAL else self._plan_cache[src_name][1]
            if not slices:
                continue
            local = local.detach()
            if local.is_floating_point() and local.dtype != self.rollout_dtype:
                local = local.to(self.rollout_dtype)
            u8 = local.contiguous().view(-1).view(torch.uint8)
            for s_ in slices:
                while s_.bucket_id > next_k:
                    await _flush(next_k, pending)
                    pending = []
                    next_k += 1
                pending.append((s_, u8))
        # the remaining buckets, including empty ones (their SNX is still due)
        for k in range(next_k, layout.num_buckets):
            await _flush(k, pending)
            pending = []
        payload_bytes = stats["payload"]
        meta_bytes += stats["meta"]

        # 4. drain: the last NB buckets were never waited on by a later flush
        for k in sorted(self._inflight):
            await self._wait_xfers_done(self._inflight[k], self.notif_timeout_s)
        self._inflight = {}
        for k in range(max(0, layout.num_buckets - nb), layout.num_buckets):
            for g in gens:
                await self._wait_notif(g.name, b"ACK", S, k, self.ack_timeout_s)
        total = time.time() - t_start

        local = {
            "total_time_s": total,
            "payload_mbytes": payload_bytes / (1 << 20),
            "metadata_bytes": float(meta_bytes + hello_bytes),
        }
        metrics = self._reduce_metrics(local)
        metrics["checkpoint_engine/num_buckets"] = float(layout.num_buckets)
        metrics["checkpoint_engine/num_params"] = float(n_params)
        metrics["checkpoint_engine/model_mbytes"] = full_layout.total_bytes / (1 << 20)
        metrics["checkpoint_engine/rank_local"] = float(wire_format == RANK_LOCAL)
        # per-generator bytes of the active layout (== model_mbytes for named_tensors)
        metrics["checkpoint_engine/layout_mbytes"] = layout.total_bytes / (1 << 20)
        logger.info(
            "sharded_nixl send v=%s done in %.2fs (%s, buckets=%d, payload=%.1fMB, first_sync=%s)",
            S,
            total,
            wire_format,
            layout.num_buckets,
            payload_bytes / (1 << 20),
            first_sync,
        )
        return metrics if self.rank == 0 else None

    def _encode_snx(self, S: int, k: int, table: list[tuple[int, int]], hashes: dict, g: _Peer) -> bytes:
        """The SNX notification for a WRITE to ``g``: the per-slice hashes inline, or, when they
        exceed ``notif_max_bytes``, a marker with the hashes sent over ZMQ."""
        hs = [hashes[t] for t in table]
        if 3 + _SNX_HDR.size + 8 * len(hs) <= self.notif_max_bytes:
            return b"SNX" + _SNX_HDR.pack(S, k, self.rank, len(hs)) + b"".join(_HASH.pack(h) for h in hs)
        self.agent.send_message(g.name, {"type": "hashes", "sync": S, "k": k, "rank": self.rank, "hashes": hs})
        return b"SNX" + _SNX_HDR.pack(S, k, self.rank, 0xFFFFFFFF)

    def _reduce_metrics(self, local: dict[str, float]) -> dict[str, float]:
        """MAX for times, SUM for counters across trainer ranks, plus the busiest rank's payload."""
        keys = sorted(local)
        vals = torch.tensor([local[k] for k in keys], dtype=torch.float64)
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
                dev = "cuda" if (dist.get_backend() == "nccl") else "cpu"
                v = vals.to(dev)
                is_time = torch.tensor([k.endswith("_s") for k in keys], device=dev)
                vmax, vsum = v.clone(), v.clone()
                dist.all_reduce(vmax, op=dist.ReduceOp.MAX)
                dist.all_reduce(vsum, op=dist.ReduceOp.SUM)
                vals = torch.where(is_time, vmax, vsum).cpu()
                pm = torch.tensor([local["payload_mbytes"]], device=dev)
                dist.all_reduce(pm, op=dist.ReduceOp.MAX)
                local_max = float(pm.item())
            else:
                local_max = local["payload_mbytes"]
        except Exception as e:  # noqa: BLE001
            logger.warning(f"sharded_nixl: metric reduction skipped: {e}")
            local_max = local["payload_mbytes"]
        out = {f"checkpoint_engine/{k}": float(v) for k, v in zip(keys, vals.tolist(), strict=True)}
        out["checkpoint_engine/local_payload_mbytes_max"] = local_max
        return out

    # ------------------------------------------------------------------ generator side
    @torch.no_grad()
    async def receive_weights(self, global_steps: int | None = None) -> AsyncGenerator[tuple[str, torch.Tensor], None]:
        async for name, weight in merge_weight_chunks(self._receive_chunks(global_steps), self.bucket_size):
            yield name, weight

    async def _receive_chunks(self, global_steps: int | None):
        assert self.role == "rollout", f"receive_weights called on role={self.role}"
        self._sync_task = asyncio.current_task()
        S = (global_steps if global_steps is not None else self._sync_counter) + self._debug_sync_offset
        self._sync_counter += 1  # per attempt, as on the trainer
        trainers = [self.peers[n] for n in self.peer_order]
        master = trainers[0].name
        nb = self.num_buffers
        t_start = time.time()

        # 0. hello to every trainer
        hello = {
            "type": "hello",
            "sync": S,
            "layout": self._rollout_layout.to_dict() if self._rollout_layout is not None else None,
            "known_digests": [self._layout.digest] if self._layout is not None else [],
            "known_plan_keys": {t: k for (t, k) in self._gen_plans.keys()},
        }
        for t in trainers:
            self.agent.send_message(t.name, hello)

        # 1. sync header (+ layout on first contact)
        msg = await self._read(master, "sync", self.notif_timeout_s)
        if msg["sync"] != S:
            raise RuntimeError(f"sharded_nixl: trainer is at sync {msg['sync']}, generator at {S}")
        if self._layout is None or self._layout.digest != msg["digest"]:
            if "layout" not in msg:
                raise RuntimeError("sharded_nixl: trainer assumed a cached layout this generator does not have")
            self._layout = BucketLayout.from_wire(msg["layout"])
            self._gen_plans = {}
        layout = self._layout
        num_buckets = msg["num_buckets"]
        wire_format = msg.get("wire_format", NAMED)
        if wire_format == RANK_LOCAL and (
            self._rollout_layout is None or RANK_LOCAL not in (self._rollout_layout.wire_formats or [])
        ):
            raise RuntimeError(
                "sharded_nixl: trainer chose rank_local_named_tensors, which this generator did not advertise"
            )
        # rank-local: the tensors yielded below have their local (box) shapes
        self.active_wire_format = wire_format
        owns = owned_name_filter(self._rollout_layout)
        plan_key = self._plan_key(layout.digest, self._rollout_layout, wire_format)

        # 2. per-trainer write tables, read only when not cached
        tables: dict[str, dict[int, list[tuple[int, int]]]] = {}
        for t in trainers:
            cached = self._gen_plans.get((t.name, plan_key))
            if cached is None:
                pm = await self._read(t.name, "plan", self.notif_timeout_s)
                if pm["plan_key"] != plan_key:
                    raise RuntimeError(f"sharded_nixl: plan key mismatch from {t.name}")
                cached = {int(k): [tuple(x) for x in v] for k, v in pm["tables"].items()}
                self._gen_plans = {kk: vv for kk, vv in self._gen_plans.items() if kk[0] != t.name}
                self._gen_plans[(t.name, plan_key)] = cached
            tables[t.name] = cached

        # 3. buckets: wait for every trainer's SNX, verify, yield, ACK
        total_bytes = total_params = 0
        for k in range(num_buckets):
            slot = k % nb
            buf = self.bufs[slot]
            notifs = {t.name: await self._wait_notif(t.name, b"SNX", S, k, self.notif_timeout_s) for t in trainers}
            for t in trainers:
                await self._verify_bucket(buf, tables[t.name].get(k, []), notifs[t.name], t, S, k)
            for cs in layout.slots_by_bucket[k]:
                if not owns(cs.name):
                    continue
                entry = layout.entry_by_name[cs.name]
                total_bytes += cs.chunk_size
                total_params += 1
                meta = TensorMeta(
                    name=cs.name,
                    shape=torch.Size(entry.shape),
                    dtype=entry.dtype,
                    chunk_offset=cs.chunk_offset,
                    chunk_size=cs.chunk_size,
                    offset=cs.offset,
                )
                yield meta, buf[cs.offset : cs.offset + cs.chunk_size]
            # the consumer has copied bucket k out (BucketedWeightSender copies into its IPC buffer
            # before asking for the next tensor); make those copies complete before handing the slot back.
            # With cpu staging they are async H2D copies out of the pinned slot, so they need it too
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            ack = b"ACK" + _ACK_HDR.pack(S, k)
            for t in trainers:
                self.agent.send_notif(t.name, ack)
        dt = time.time() - t_start
        logger.info(
            "sharded_nixl recv v=%s done in %.2fs (buckets=%d, params=%d, bytes=%.1fMB)",
            S,
            dt,
            num_buckets,
            total_params,
            total_bytes / (1 << 20),
        )

    async def _verify_bucket(
        self, buf: torch.Tensor, table: list[tuple[int, int]], notif: bytes, t: _Peer, S: int, k: int
    ):
        """Compare the hashes trainer ``t`` sent for bucket ``k`` with what landed in ``buf``."""
        _sync, _k, rank, n_inline = _SNX_HDR.unpack_from(notif, 3)
        if not table:
            return
        if n_inline == 0xFFFFFFFF:
            hm = await self._read(t.name, "hashes", self.notif_timeout_s)
            if hm["sync"] != S or hm["k"] != k:
                raise RuntimeError(f"sharded_nixl: out-of-order hashes message from trainer {rank}")
            expected = list(hm["hashes"])
        else:
            off = 3 + _SNX_HDR.size
            expected = [_HASH.unpack_from(notif, off + 8 * i)[0] for i in range(n_inline)]
        if len(expected) != len(table):
            raise RuntimeError(
                f"sharded_nixl: trainer {rank} sent {len(expected)} hashes for {len(table)} slices (bucket {k})"
            )
        got = _hash_u8_many(buf, table)
        bad = [i for i, (e, g) in enumerate(zip(expected, got, strict=True)) if e != g]
        if bad:
            dst, ln = table[bad[0]]
            raise RuntimeError(
                f"sharded_nixl: checksum mismatch in bucket {k} from trainer {rank}: {len(bad)} slice(s), "
                f"first at dst={dst} len={ln}"
            )


def _approx_size(obj) -> int:
    """Pickled size of a control message, for the metadata_bytes metric."""
    try:
        import pickle

        return len(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))
    except Exception:  # noqa: BLE001
        return 0
