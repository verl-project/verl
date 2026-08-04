# Copyright 2026 Amazon.com Inc and/or its affiliates
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
"""Parallel-sender NCCL checkpoint engine (``nccl_parallel``).

The stock ``nccl`` engine funnels every byte through actor rank 0: ranks 1..N-1
materialize the identical full-tensor weight stream (their iteration drives the
FSDP all-gathers) and then discard it, while rank 0 alone buckets and broadcasts
on a single NCCL communicator, so send bandwidth is capped at one GPU's NIC share.

This engine turns every actor rank into a sender over ONE GLOBAL BUCKET
SEQUENCE round-robined across the ranks:

- All actor ranks iterate the identical weight stream and therefore derive the
  identical bucket boundaries (bucketization depends only on tensor sizes).
  Global bucket ``b`` is owned by sender ``b % S``; owners copy their buckets
  into a transfer buffer, everyone else just drains (draining is mandatory
  anyway; producing each tensor drives a collective FSDP gather).
- Sender ``s`` broadcasts on its own NCCL group (``{group_name}_{s}``: the
  sender as rank 0 plus every rollout worker) with its own zmq PUB metadata
  channel; receivers join all S groups and consume buckets in ascending global
  order.
- Transfers run in BULK-SYNCHRONOUS PHASES of ``phase`` consecutive buckets
  (distinct owners): senders synchronize on a dedicated coordination group,
  the phase's owners broadcast concurrently (the fan-out), each waits for its
  own transfer to complete on-device, then all synchronize again and resume
  gathering. The receiver keeps exactly the phase's reads outstanding.

Why bulk-synchronous phases are load-bearing (do not remove them):
the trainer's weight-stream materialization issues on the order of a thousand
of its OWN per-tensor collectives during a sync. Free-running engine broadcasts
therefore interleave at a DIFFERENT position in each rank's collective stream,
violating NCCL's requirement that ranks sharing communicators issue collectives
in a consistent order; the job deadlocks with every GPU spinning at 100%
(diagnosed via NCCL_DEBUG_SUBSYS=COLL; multiple prior designs hit variants of
this). The phase barriers pin the ENQUEUE order: on every rank the sequence
is [same trainer prefix] → entry barrier → own broadcast (owners) →
exit barrier → [same trainer suffix]. Because the barrier and the broadcasts
live on different comm streams, an owner's broadcast KERNEL may start before a
lagging peer has enqueued its entry barrier. That is safe: the lagging peer's
own broadcast is locally ordered after its entry barrier and completes before
its exit, and post-phase trainer collectives are locally ordered after the
exit, so they can wait on a lagging peer but never prevent it from finishing
its transfer. (Waiting for entry-barrier COMPLETION before broadcasting was
tried and deadlocks the real integration; see _sender_barrier.)

Stream discipline: ray.util.collective creates its per-communicator streams as
BLOCKING with respect to the default stream, which would (a) serialize each
sender's next gathers behind its in-flight broadcast and (b) let receivers'
early-posted reads block their apply-path copies. The engine therefore patches
ray's stream factory to non-blocking (process-wide, applied once, before any
group exists) and replaces the lost implicit ordering with explicit
per-transfer CUDA events (senders) and a phase-boundary device synchronize
(receivers, safe because nothing else is in flight at a phase boundary).

Measured reference (2-node Qwen3.5-27B, 55.6 GB bf16, GEN_TP=4, bucket 2048MB,
8 senders, AWS EFA interconnect; n=20 refits): 2.51s ±0.27 per refit vs
3.01s ±0.25 for the stock ``nccl`` engine on the same workload; over 100-step
GSM8K runs, plateau accuracy (steps 61-100) matched the stock engine to within
0.0005. Also validated on 4-node clusters (same recipe, 10 refits each, one run
per topology): 1/2/3 trainer nodes (S=8/16/24) against 3/2/1 rollout nodes,
``phase=8``, all ran to completion with normal learning curves; per-refit sync
2.48/5.55/2.93 s; the 2-trainer/2-rollout split is the slow one, an
interaction not yet characterized. Every receiver still ingests every byte:
this engine aggregates sender-side NIC bandwidth; it does not reduce bytes on
the wire.
"""

import asyncio
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncGenerator, Generator
from unittest.mock import patch

with patch("importlib.metadata.distributions", return_value=[]):
    import cupy as cp

import ray
import ray.util.collective as collective
import torch
import zmq

from verl.checkpoint_engine.base import (
    CheckpointEngine,
    CheckpointEngineRegistry,
    TensorMeta,
    merge_weight_chunks,
    split_weight_chunks,
)
from verl.checkpoint_engine.nccl_checkpoint_engine import MasterMetadata
from verl.utils.net_utils import get_free_port, is_valid_ipv6_address

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_MIN_BUCKET = 32 << 20  # floor for the per-group bucket after dividing by S

# zmq PUB/SUB drops messages published before the subscription has PROPAGATED to
# the publisher (the "slow joiner" problem). Cross-node that window is real, and
# a lost first metadata message means the receiver never posts its NCCL recv ->
# the sender's kernel spin-waits -> the trainer stalls. Handshake: senders
# publish SYNC pings until a second NCCL barrier proves every receiver's
# subscription is live; receivers filter residual pings afterwards.
_ZMQ_SYNC = "__zmq_sync_ping__"
# End-of-stream marker, published once per sender per round. Carries no bucket,
# so the receiver must NOT post a matching NCCL broadcast for it.
_STREAM_END = "__stream_end__"
# Receiver-side metadata timeout: a protocol bug raises in minutes instead of
# hanging until the NCCL watchdog kills the trainer half an hour later.
_RECV_TIMEOUT_MS = 300_000
# Sender-side bound on one broadcast completing on-device.
_TRANSFER_TIMEOUT_S = 600.0
# Senders' coordination group for bulk-synchronous phases.
_SENDER_GROUP = "nccl_parallel_senders"

# Incremented every time the patched factory creates a comm stream; lets a test
# assert the non-blocking patch actually took effect.
_NONBLOCKING_STREAMS_CREATED = 0


def _force_nonblocking_comm_streams() -> bool:
    """Make ray.util.collective's per-communicator streams NON-BLOCKING.

    ray creates them as ``cupy.cuda.Stream(null=False, non_blocking=False)``
    (``ray/util/collective/collective_group/cuda_stream.py``): BLOCKING streams
    implicitly serialize with PyTorch's legacy default stream in both
    directions, which (a) stalls each sender's subsequent FSDP gathers behind
    its in-flight broadcast (serializing the senders the engine exists to
    parallelize) and (b) lets receivers' posted reads block apply-path copies.

    ray's own producer->comm ordering does not depend on the blocking flag: its
    ``_sync_streams`` explicitly records an event on the current stream and
    makes the comm stream wait for it. The consumer direction is this engine's
    job: per-transfer CUDA events on senders, a phase-boundary device
    synchronize on receivers.

    Must run before any collective group exists (ray builds one stream pool per
    device on first use and caches it). Returns True if the patch is in effect.
    """
    try:
        from ray.util.collective.collective_group import cuda_stream as _ray_cuda_stream
    except ImportError:  # pragma: no cover - ray layout changed
        logger.warning("nccl_parallel: ray cuda_stream module not found; comm streams stay blocking")
        return False

    if getattr(_ray_cuda_stream, "_verl_nonblocking_patched", False):
        return True
    # ray caches one StreamPool per device in _device_stream_pool_map on first
    # use; patching after any pool exists would leave those streams blocking.
    if getattr(_ray_cuda_stream, "_device_stream_pool_map", None):
        logger.warning("nccl_parallel: ray stream pool already built; comm streams stay blocking")
        return False

    original_stream_cls = _ray_cuda_stream.cupy.cuda.Stream

    def _nonblocking_stream(*args, **kwargs):
        # cupy.cuda.Stream is a Cython class; call it rather than subclass it.
        global _NONBLOCKING_STREAMS_CREATED
        kwargs["non_blocking"] = True
        stream = original_stream_cls(*args, **kwargs)
        _NONBLOCKING_STREAMS_CREATED += 1
        return stream

    class _CupyCudaShim:
        """Only the Stream attribute differs; everything else delegates."""

        Stream = staticmethod(_nonblocking_stream)

        def __init__(self, wrapped):
            self._wrapped = wrapped

        def __getattr__(self, name):
            return getattr(self._wrapped, name)

    class _CupyShim:
        def __init__(self, wrapped):
            self._wrapped = wrapped
            self.cuda = _CupyCudaShim(wrapped.cuda)

        def __getattr__(self, name):
            return getattr(self._wrapped, name)

    _ray_cuda_stream.cupy = _CupyShim(_ray_cuda_stream.cupy)
    _ray_cuda_stream._verl_nonblocking_patched = True
    logger.info("nccl_parallel: ray collective comm streams patched to non-blocking")
    return True


def _comm_stream(group_name: str):
    """Best-effort handle on ray's per-communicator cupy stream (None if absent).

    Needed to wait for ONE broadcast to finish; ``torch.cuda.synchronize()`` is
    not a substitute on senders: device-wide it would also wait for the other
    senders' work and re-serialize them.
    """
    try:
        group = collective.get_group_handle(group_name)
    except Exception:  # pragma: no cover - ray API drift
        return None
    for streams in (getattr(group, "_dev_streams_map", None) or {}).values():
        if streams:
            return streams[0]
    return None


class _SendOperation:
    """Publish bucket metadata, post the NCCL broadcast, record a completion event.

    The metadata is published BEFORE the broadcast is enqueued, so the receiver
    can always post the matching read, so the broadcast never waits on a receiver
    that has not been told about it.
    """

    def __init__(
        self,
        group_name: str,
        bucket,
        metadata: dict,
        socket: zmq.Socket,
        topic: str,
        device: int,
        executor: ThreadPoolExecutor | None = None,
    ) -> None:
        self.group_name = group_name
        self.bucket = bucket
        self.metadata = metadata
        self.socket = socket
        self.topic = topic
        self.device = device
        self.event = None
        loop = asyncio.get_running_loop()
        self._task = loop.run_in_executor(executor, self._run)

    def _run(self):
        # Pooled threads start on device 0; verl pins workers with
        # set_device(local_rank) when RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES
        # is set, so an unpinned thread would touch the wrong GPU.
        torch.cuda.set_device(self.device)
        self.socket.send_string(self.topic, flags=zmq.SNDMORE)
        self.socket.send_pyobj(self.metadata)
        collective.broadcast(self.bucket, src_rank=0, group_name=self.group_name)
        stream = _comm_stream(self.group_name)
        if stream is not None:
            event = cp.cuda.Event()
            event.record(stream)
            self.event = event

    async def wait_for_transfer(self, executor: ThreadPoolExecutor | None = None):
        """Block until this broadcast has actually completed on the device.

        Bounded end to end: a broadcast blocked inside NCCL (not just an
        unfinished kernel) also trips the timeout. Any raise here is
        JOB-FATAL by design -- it propagates through the Ray actor to the
        manager's ray.get, which is verl's coordinated-abort path; a partially
        synced rollout must not be resumed.
        """
        await asyncio.wait_for(asyncio.shield(self._task), timeout=_TRANSFER_TIMEOUT_S + 60)
        loop = asyncio.get_running_loop()
        if self.event is not None:
            await loop.run_in_executor(executor, self._await_event)
        else:
            # Fallback: no stream handle (ray internals changed). Correct but
            # serializes this rank against every other stream on the device.
            await loop.run_in_executor(executor, torch.cuda.synchronize, self.device)

    def _await_event(self):
        torch.cuda.set_device(self.device)  # pooled threads default to device 0
        deadline = time.monotonic() + _TRANSFER_TIMEOUT_S
        while not self.event.done:  # cupy Event exposes `done`, not query()
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"nccl_parallel: broadcast on {self.group_name} did not complete within "
                    f"{_TRANSFER_TIMEOUT_S:.0f}s (receiver stalled?)"
                )
            time.sleep(0.001)


class _RecvOperation:
    """Receive one metadata message (skipping residual handshake pings) and post
    the matching NCCL read, unless it is the end-of-stream marker, which
    carries no bucket. Runs in an executor thread so the event loop stays free
    to service the other groups in the phase."""

    def __init__(
        self,
        group_name: str,
        bucket: torch.Tensor,
        socket: zmq.Socket,
        topic: str,
        device: int,
        executor: ThreadPoolExecutor | None = None,
    ) -> None:
        self.group_name = group_name
        self.bucket = bucket
        self.socket = socket
        self.topic = topic
        self.device = device
        self.metadata = None
        loop = asyncio.get_running_loop()
        self._task = loop.run_in_executor(executor, self._run)

    def _run(self):
        torch.cuda.set_device(self.device)  # pooled threads default to device 0
        while True:
            try:
                self.socket.recv_string()
                metadata = self.socket.recv_pyobj()
            except zmq.Again as e:
                raise RuntimeError(
                    f"nccl_parallel: no metadata on {self.group_name} within "
                    f"{_RECV_TIMEOUT_MS / 1000:.0f}s; sender stalled or message lost"
                ) from e
            if metadata != _ZMQ_SYNC:
                break
        self.metadata = metadata
        if metadata == _STREAM_END:
            return
        collective.broadcast(self.bucket, src_rank=0, group_name=self.group_name)

    async def wait_for_complete(self, executor: ThreadPoolExecutor | None = None):
        """Wait for the metadata + the recv ENQUEUE (not kernel completion).

        Kernel completion is waited collectively at the phase boundary (the
        bounded device synchronize in receive_weights). Waiting per-bucket
        kernel events here was tried and wedged the real integration the same
        way the sender entry-barrier completion wait did (first sync, all
        receivers parked in the event poll): per-transfer completion coupling
        changes the receivers' pacing relative to the colocated vLLM processes.
        The enqueue-only pattern is the one validated by multi-step real
        training runs. The wait below is still bounded (metadata RCVTIMEO +
        grace), and the phase sync is bounded by its caller, so failure remains
        loud and job-fatal.
        """
        await asyncio.wait_for(asyncio.shield(self._task), timeout=_RECV_TIMEOUT_MS / 1000 + 60)
        return self.metadata


@CheckpointEngineRegistry.register("nccl_parallel")
class NCCLParallelCheckpointEngine(CheckpointEngine):
    """NCCL checkpoint engine with all actor ranks sending in bulk-synchronous phases.

    Args:
        bucket_size (int): TOTAL in-flight transfer budget in bytes; divided by
            the sender count to size the per-group bucket, so receiver device
            memory stays ~2 * bucket_size (S groups x 2 buffers x bucket_size/S).
            The per-group bucket is floored at 32MB, so a budget below S * 32MB
            is NOT honored: receiver buffers then total 2 * S * 32MB and the
            engine logs a warning at init_process_group. It is also rounded down
            to a 16-byte boundary so that a bucket-sized cut through a large
            tensor cannot leave the tensors packed behind it at a
            dtype-misaligned offset (see init_process_group).
        group_name (str): Base name for the per-sender NCCL groups.
        rebuild_group (bool): Rebuild the NCCL groups on each update round.
        is_master (bool): Accepted for ctor compatibility with the stock engine
            (actor rank 0 passes True); sender-ness is decided by the topology.
        phase (int | None): Concurrent transfers per bulk-synchronous phase,
            1..S (default S). Each in-phase transfer occupies NCCL kernel
            resources on every receiver GPU concurrently with the colocated
            vLLM processes; lower it (or cap NCCL_MAX_NCHANNELS) if receivers
            cannot co-schedule the full fan-out.
        rollout_dtype (torch.dtype): Unused; kept for kwarg compatibility.
    """

    def __init__(
        self,
        bucket_size: int,
        group_name: str = "nccl_parallel",
        rebuild_group: bool = False,
        is_master: bool = False,
        phase: int | None = None,
        rollout_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.total_bucket_size = bucket_size
        self.group_name = group_name
        self.rebuild_group = rebuild_group
        self.configured_phase = phase
        self.topic = "bucket_metadata"
        # Patch here: the engine is constructed before any collective group, and
        # ray caches one stream pool per device on first use. FAIL CLOSED if the
        # patch cannot take effect: this engine's ordering discipline is built
        # for non-blocking comm streams, and running it on blocking streams
        # wedges (receivers' posted reads block their apply-path copies).
        # Note for mixed use: the stock `nccl` engine remains correct under
        # non-blocking streams -- its device-wide torch.cuda.synchronize()
        # calls cover streams of either kind -- so patching process-wide is
        # safe for processes that also construct other backends.
        self.nonblocking_streams = _force_nonblocking_comm_streams()
        if not self.nonblocking_streams:
            raise RuntimeError(
                "nccl_parallel requires ray's collective comm streams to be non-blocking, but the "
                "stream factory could not be patched (a stream pool already exists in this process). "
                "Construct this engine before any other ray collective group is used."
            )

        # Filled by init_process_group (topology-dependent).
        self.num_senders: int | None = None
        self.bucket_size: int | None = None
        self.phase: int | None = None
        self.stripe: int | None = None  # sender's group index; None = not a sender
        self.group_ranks: list[int] | None = None
        self.world_size: int | None = None
        self.pub_socket: zmq.Socket | None = None
        self._master_metadata: MasterMetadata | None = None
        self.sub_sockets: list[zmq.Socket] | None = None
        self._executor: ThreadPoolExecutor | None = None
        self._device: int | None = None
        self.send_bufs = None  # sender: (front, back) cupy buffers
        self.recv_bufs: list[list[torch.Tensor]] | None = None  # receiver: per group [front, back]

    # ------------------------------------------------------------------ setup

    def prepare(self) -> MasterMetadata:
        """Bind a zmq PUB endpoint on EVERY worker (ONCE; reused across rounds).

        Topology is not known yet (the manager calls prepare() before
        build_topology), so all workers advertise an endpoint and build_topology
        uses the actor entries as the sender masters. The manager calls
        prepare() every round (rebinding each round leaks sockets and races
        get_free_port's probe), so bind exactly once and return the cached
        endpoint.
        """
        if self.pub_socket is not None:
            return self._master_metadata

        self.ip = ray.util.get_node_ip_address().strip("[]")
        self.listen_port, _ = get_free_port(self.ip)

        context = zmq.Context()
        self.pub_socket = context.socket(zmq.PUB)
        if is_valid_ipv6_address(self.ip):
            address = f"tcp://[{self.ip}]:{self.listen_port}"
            self.pub_socket.setsockopt(zmq.IPV6, 1)
        else:
            address = f"tcp://{self.ip}:{self.listen_port}"
        self.pub_socket.bind(address)

        self._master_metadata = MasterMetadata(zmq_ip=self.ip, zmq_port=self.listen_port)
        return self._master_metadata

    @classmethod
    def build_topology(cls, actor_wg_world_size: int, rollout_world_size: int, metadata: list[dict]):
        """S = actor_wg_world_size sender groups; every rollout worker joins all S.

        metadata layout (from CheckpointEngineManager.build_process_group):
        [actor_0 .. actor_{A-1}, rollout_0 .. rollout_{R-1}] prepare() returns.
        """
        num_senders = actor_wg_world_size
        sender_metadata = list(metadata[:num_senders])

        actor_wg_kwargs = {
            "group_ranks": [[0 if s == a else -1 for s in range(num_senders)] for a in range(actor_wg_world_size)],
            "world_size": [rollout_world_size + 1] * actor_wg_world_size,
            "master_metadata": [sender_metadata] * actor_wg_world_size,
        }
        rollout_kwargs = {
            "group_ranks": [[j + 1] * num_senders for j in range(rollout_world_size)],
            "world_size": [rollout_world_size + 1] * rollout_world_size,
            "master_metadata": [sender_metadata] * rollout_world_size,
        }
        return actor_wg_kwargs, rollout_kwargs

    def _group(self, s: int) -> str:
        return f"{self.group_name}_{s}"

    def init_process_group(self, group_ranks: list[int], world_size: int, master_metadata: list[MasterMetadata]):
        """Join this worker's groups and allocate transfer buffers.

        Args:
            group_ranks: This worker's rank in group s (or -1 if not a member).
            world_size: World size of every group (1 sender + R receivers).
            master_metadata: zmq endpoints of the S senders, index-aligned.
        """
        self.num_senders = len(master_metadata)
        self.bucket_size = max(self.total_bucket_size // self.num_senders, _MIN_BUCKET)
        # Align DOWN to a 16-byte boundary. split_weight_chunks cuts a large
        # tensor at multiples of bucket_size, and the receive path views the
        # reassembled bytes as the tensor dtype -- torch requires the storage
        # offset to divide the element size, so an odd bucket size (e.g.
        # 2048MB / 24 senders) shifts every tensor after the cut to a
        # misaligned offset and the view raises. A multiple of 16 divides every
        # dtype itemsize torch ships (largest is complex128), which removes the
        # bucket cut as a source of misalignment. It is not a general alignment
        # guarantee: back-to-back packing of a MIXED-dtype stream can still land
        # a later tensor at a misaligned offset whatever the bucket size --
        # behavior shared with the stock nccl engine, and not reachable from the
        # homogeneous-dtype (bf16) stream the export path produces today.
        self.bucket_size -= self.bucket_size % 16
        if self.bucket_size > self.total_bucket_size // self.num_senders:
            logger.warning(
                f"nccl_parallel: per-group bucket floored to {_MIN_BUCKET >> 20}MB; receiver transfer buffers "
                f"total 2 x {self.num_senders} x {self.bucket_size >> 20}MB, exceeding the configured "
                f"update_weights_bucket_megabytes budget"
            )
        if self.configured_phase is None:
            self.phase = self.num_senders
        else:
            if not isinstance(self.configured_phase, int) or not 1 <= self.configured_phase <= self.num_senders:
                raise ValueError(
                    f"nccl_parallel: phase must be an int in [1, {self.num_senders}], got {self.configured_phase!r}"
                )
            self.phase = self.configured_phase
        self.group_ranks = group_ranks
        self.world_size = world_size
        # Captured on the main thread, where verl has already pinned the worker's
        # device; executor threads must be pinned to the same one.
        self._device = torch.cuda.current_device()
        if self._executor is None:
            # Dedicated pool for blocking recvs / event waits: a phase parks up
            # to S threads, which would otherwise squat on the default executor
            # shared with the rest of the worker.
            self._executor = ThreadPoolExecutor(max_workers=self.num_senders + 2, thread_name_prefix="ckpt_parallel")

        member_groups = [s for s, r in enumerate(group_ranks) if r >= 0]
        is_rollout = bool(member_groups) and all(group_ranks[s] > 0 for s in member_groups)

        if is_rollout:
            # Receiver: joins all S groups. SUB sockets connect ONCE (sender PUB
            # endpoints are bound once and stable across rounds); buffers are
            # re-allocated when finalize() has freed them.
            if self.sub_sockets is None:
                context = zmq.Context()
                self.sub_sockets = []
                for s in range(self.num_senders):
                    sock = context.socket(zmq.SUB)
                    meta = master_metadata[s]
                    if is_valid_ipv6_address(meta.zmq_ip):
                        sock.setsockopt(zmq.IPV6, 1)
                        sock.connect(f"tcp://[{meta.zmq_ip}]:{meta.zmq_port}")
                    else:
                        sock.connect(f"tcp://{meta.zmq_ip}:{meta.zmq_port}")
                    sock.setsockopt_string(zmq.SUBSCRIBE, self.topic)
                    sock.setsockopt(zmq.RCVTIMEO, _RECV_TIMEOUT_MS)
                    self.sub_sockets.append(sock)
            if self.recv_bufs is None:
                self.recv_bufs = [
                    [
                        torch.zeros(self.bucket_size, dtype=torch.uint8, device="cuda"),
                        torch.zeros(self.bucket_size, dtype=torch.uint8, device="cuda"),
                    ]
                    for _ in range(self.num_senders)
                ]
        elif member_groups:
            # Sender: exactly one group, where our rank is 0.
            assert len(member_groups) == 1 and group_ranks[member_groups[0]] == 0, (
                f"actor worker must master exactly one group, got group_ranks={group_ranks}"
            )
            self.stripe = member_groups[0]
            # cupy buffers on senders: sidestep the torch expandable_segments
            # memory-registration issue (same reason as the stock engine).
            self.send_bufs = (
                cp.zeros(self.bucket_size, dtype=cp.uint8),
                cp.zeros(self.bucket_size, dtype=cp.uint8),
            )
            # Senders' phase-coordination group (see module docstring).
            if self.rebuild_group or not collective.is_group_initialized(_SENDER_GROUP):
                collective.init_collective_group(self.num_senders, self.stripe, "nccl", _SENDER_GROUP)
            collective.barrier(_SENDER_GROUP)

        # Join groups in ascending s on every member; each group completes
        # independently (sender s + all receivers), so there is no circular wait.
        for s in member_groups:
            if self.rebuild_group or not collective.is_group_initialized(self._group(s)):
                collective.init_collective_group(world_size, group_ranks[s], "nccl", self._group(s))
        for s in member_groups:
            collective.barrier(self._group(s))

        # zmq slow-joiner handshake (see _ZMQ_SYNC above). Sender: ping until
        # barrier #2 proves every receiver's subscription is live. Receiver:
        # block for the first ping on each channel, then barrier. Ascending
        # group order on receivers keeps this deadlock-free.
        if self.stripe is not None:
            import threading

            stop = threading.Event()

            def _ping():
                while not stop.is_set():
                    self.pub_socket.send_string(self.topic, flags=zmq.SNDMORE)
                    self.pub_socket.send_pyobj(_ZMQ_SYNC)
                    time.sleep(0.05)

            pinger = threading.Thread(target=_ping, daemon=True)
            pinger.start()
            try:
                collective.barrier(self._group(self.stripe))
                # With non-blocking comm streams, barrier() returns at ENQUEUE.
                # The pings must continue until the barrier has COMPLETED --
                # completion proves every receiver posted it, which it only
                # does after receiving a ping. Wait on the comm stream event.
                stream = _comm_stream(self._group(self.stripe))
                if stream is not None:
                    barrier_done = cp.cuda.Event()
                    barrier_done.record(stream)
                    deadline = time.monotonic() + _TRANSFER_TIMEOUT_S
                    while not barrier_done.done:
                        if time.monotonic() > deadline:
                            raise RuntimeError("nccl_parallel: zmq handshake barrier did not complete")
                        time.sleep(0.001)
                else:  # pragma: no cover - ray API drift
                    torch.cuda.synchronize()
            finally:
                # Always reclaim the PUB socket, even on handshake failure -- a
                # live pinger would race any later publish.
                stop.set()
                pinger.join()  # the PUB socket is single-threaded again from here on
            logger.info(f"nccl_parallel stripe {self.stripe}: zmq handshake complete")
        elif is_rollout:
            for s in range(self.num_senders):
                try:
                    self.sub_sockets[s].recv_string()
                    msg = self.sub_sockets[s].recv_pyobj()
                except zmq.Again as e:
                    raise RuntimeError(
                        f"nccl_parallel: no zmq handshake ping from sender {s} within "
                        f"{_RECV_TIMEOUT_MS / 1000:.0f}s (check the PUB endpoint is reachable cross-node)"
                    ) from e
                assert msg == _ZMQ_SYNC, f"expected sync ping on group {s} during handshake, got {type(msg)}"
                collective.barrier(self._group(s))
            logger.info("nccl_parallel receiver: zmq handshake complete on all groups")

        logger.info(
            f"nccl_parallel init: S={self.num_senders}, bucket={self.bucket_size >> 20}MB, "
            f"phase={self.phase}, stripe={self.stripe}, member_groups={member_groups}"
        )

    def finalize(self):
        if self.rebuild_group and self.group_ranks is not None:
            for s, r in enumerate(self.group_ranks):
                if r >= 0:
                    collective.destroy_collective_group(self._group(s))
            if self.stripe is not None:
                collective.destroy_collective_group(_SENDER_GROUP)
            self.group_ranks = None
        self.send_bufs = None
        self.recv_bufs = None
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------ send

    def _publish(self, payload):
        """Publish one metadata object on our PUB channel (main thread only)."""
        self.pub_socket.send_string(self.topic, flags=zmq.SNDMORE)
        self.pub_socket.send_pyobj(payload)

    @torch.no_grad()
    async def send_weights(
        self,
        weights: Generator[tuple[str, torch.Tensor], None, None],
        global_steps: int | None = None,
    ):
        """Broadcast the global buckets we own in bulk-synchronous phases.

        Every actor rank consumes the whole stream (the iteration drives the
        FSDP gathers); owners additionally copy their buckets into the transfer
        buffer and broadcast them at phase boundaries. Bucket boundaries and
        phase boundaries derive ONLY from the shared stream, so every rank
        reaches every barrier the same number of times; owner-local state must
        never influence control flow here.
        """
        if self.stripe is None:
            for _name, _weight in weights:  # still must drive the FSDP gathers
                pass
            return {}

        S = self.num_senders
        loop = asyncio.get_running_loop()
        start_time = time.time()
        front, back = self.send_bufs
        bucket_idx = 0
        offset = 0  # position within the CURRENT global bucket (tracked by every sender)
        bucket_meta: dict[str, TensorMeta] = {}
        pending_own: tuple[int, dict[str, TensorMeta]] | None = None
        total_bytes = 0
        my_buckets = 0
        num_global_buckets = 0

        def _sender_barrier():
            """Enqueue the senders' phase barrier (deliberately NOT completion-waited).

            The correctness invariant is that every rank issues collectives in
            a CONSISTENT ENQUEUE ORDER across communicators; the barrier's
            position in each rank's stream provides that. NOTE the barrier and
            the broadcasts are on DIFFERENT comm streams, so an owner's
            broadcast kernel may run before a lagging peer enqueues its entry
            barrier; safety comes from local ordering (each peer's own
            broadcast follows its entry barrier and precedes its exit; trainer
            collectives follow the exit), not from an on-device rendezvous.
            Waiting for entry-barrier COMPLETION before broadcasting was tried
            and deadlocks the real integration -- a real run timed out on
            exactly that wait in its first sync, while the enqueue-only
            discipline repeatedly completed 20-step real benchmarks.
            """
            torch.cuda.set_device(self._device)  # pooled threads default to device 0
            collective.barrier(_SENDER_GROUP)

        async def phase_flush() -> None:
            """One bulk-synchronous phase (see module docstring)."""
            nonlocal pending_own, my_buckets, front, back
            await loop.run_in_executor(self._executor, _sender_barrier)
            if pending_own is not None:
                idx, meta = pending_own
                op = _SendOperation(
                    group_name=self._group(self.stripe),
                    bucket=front,
                    metadata={"bucket_meta": meta, "global_idx": idx},
                    socket=self.pub_socket,
                    topic=self.topic,
                    device=self._device,
                    executor=self._executor,
                )
                await op.wait_for_transfer(self._executor)
                front, back = back, front
                my_buckets += 1
                pending_own = None
            await loop.run_in_executor(self._executor, _sender_barrier)

        async for tensor_meta, chunk in split_weight_chunks(weights, self.bucket_size):
            if offset + tensor_meta.chunk_size > self.bucket_size:
                if bucket_idx % S == self.stripe:
                    # Our bucket of the current phase is full; broadcast at the
                    # phase boundary below.
                    pending_own = (bucket_idx, bucket_meta)
                if (bucket_idx + 1) % self.phase == 0:
                    await phase_flush()
                bucket_idx += 1
                num_global_buckets = bucket_idx
                offset = 0
                bucket_meta = {}

            if bucket_idx % S == self.stripe:
                assert tensor_meta.name not in bucket_meta, f"duplicate {tensor_meta.name} in bucket {bucket_idx}"
                tensor_meta.offset = offset
                bucket_meta[tensor_meta.name] = tensor_meta
                front[offset : offset + tensor_meta.chunk_size] = cp.asarray(chunk)
                total_bytes += tensor_meta.chunk_size
            offset += tensor_meta.chunk_size

        # Tail bucket (non-empty unless the stream ended exactly on a boundary),
        # then the final (possibly partial) phase. Deterministic on every rank:
        # derived from the shared stream only (bucket count / tail offset),
        # never from owner-local state like pending_own, which would diverge the
        # barrier count across ranks.
        if offset > 0:
            num_global_buckets = bucket_idx + 1
            if bucket_idx % S == self.stripe:
                pending_own = (bucket_idx, bucket_meta)
        if num_global_buckets % self.phase != 0 or offset > 0:
            await phase_flush()

        # The FINAL exit barrier must have COMPLETED before this round is done:
        # intermediate exit barriers are retired by later enqueues on the same
        # stream, but nothing follows the last one, and finalize() with
        # rebuild_group=True would otherwise destroy the senders communicator
        # with the barrier still in flight (ray's destroy does not synchronize).
        def _wait_sender_stream():
            torch.cuda.set_device(self._device)
            stream = _comm_stream(_SENDER_GROUP)
            if stream is None:  # pragma: no cover - ray API drift
                torch.cuda.synchronize(self._device)
                return
            done = cp.cuda.Event()
            done.record(stream)
            deadline = time.monotonic() + _TRANSFER_TIMEOUT_S
            while not done.done:
                if time.monotonic() > deadline:
                    raise RuntimeError("nccl_parallel: final sender phase barrier did not complete")
                time.sleep(0.001)

        await loop.run_in_executor(self._executor, _wait_sender_stream)

        # Exactly one end-of-stream marker per sender per round; the receiver
        # consumes exactly one per group, keeping SUB channels clean across rounds.
        self._publish(_STREAM_END)

        elapsed = time.time() - start_time
        logger.info(
            f"nccl_parallel stripe {self.stripe}: sent {total_bytes / (1 << 30):.2f} GiB in {my_buckets} buckets "
            f"({num_global_buckets} global, S={S}, phase={self.phase}, "
            f"bucket={self.bucket_size >> 20}MB) in {elapsed:.2f}s"
        )
        return {f"sync/parallel_send_gib_stripe{self.stripe}": total_bytes / (1 << 30)}

    # --------------------------------------------------------------- receive

    async def receive_weights(
        self,
        global_steps: int | None = None,
    ) -> AsyncGenerator[tuple[str, torch.Tensor], None]:
        """Consume global buckets in ascending order, one sender phase at a time.

        The receiver keeps exactly the current phase's reads outstanding
        (buckets p*phase .. p*phase+phase-1, distinct groups), applies them in
        global order after a device synchronize, then posts the next phase's
        reads. Mirrors the sender's phase discipline: no read is outstanding
        for a bucket whose sender has not reached the same phase.
        """
        start_time = time.time()
        S = self.num_senders
        parity = [0] * S
        pending: dict[int, _RecvOperation] = {}

        def post(idx: int):
            s = idx % S
            buf = self.recv_bufs[s][parity[s]]
            parity[s] ^= 1
            pending[idx] = _RecvOperation(
                group_name=self._group(s),
                bucket=buf,
                socket=self.sub_sockets[s],
                topic=self.topic,
                device=self._device,
                executor=self._executor,
            )

        loop = asyncio.get_running_loop()

        async def chunk_stream() -> AsyncGenerator[tuple[TensorMeta, torch.Tensor], None]:
            idx = 0
            done = False
            while not done:
                if idx:
                    # The consumer's copies out of earlier phases' buffers
                    # (merge_weight_chunks / the apply path) are ISSUED on the
                    # default stream during the yields but complete
                    # asynchronously; each group's double buffer is reused two
                    # phases later, and with non-blocking comm streams nothing
                    # orders a still-queued consumer copy against the recv
                    # broadcast that would overwrite its source buffer. Nothing
                    # of ours is in flight at a phase boundary, so a device-wide
                    # synchronize here closes that window cheaply. Bounded like
                    # every other wait: fail loudly, never hang to the watchdog.
                    await asyncio.wait_for(
                        asyncio.shield(loop.run_in_executor(self._executor, torch.cuda.synchronize, self._device)),
                        timeout=_TRANSFER_TIMEOUT_S,
                    )
                for k in range(self.phase):
                    post(idx + k)
                ops = dict(sorted(pending.items()))
                pending.clear()
                metas = {}
                failures: list[BaseException] = []
                for b, op in ops.items():
                    # Await EVERY started operation even after a failure, so no
                    # executor future goes unobserved.
                    try:
                        metas[b] = await op.wait_for_complete(self._executor)
                    except BaseException as exc:  # noqa: BLE001 - collected and re-raised
                        failures.append(exc)
                if failures:
                    raise failures[0]

                if any(m != _STREAM_END for m in metas.values()):
                    # Wait for the phase's recv kernels, ALL AT ONCE. Device-wide
                    # is safe here (and only here): at a phase boundary nothing
                    # else of ours is pending on this device's streams. Bounded:
                    # a never-completing phase raises instead of hanging to the
                    # NCCL watchdog (the sync thread itself cannot be cancelled,
                    # but the job fails loudly, which is the requirement).
                    await asyncio.wait_for(
                        asyncio.shield(loop.run_in_executor(self._executor, torch.cuda.synchronize, self._device)),
                        timeout=_TRANSFER_TIMEOUT_S,
                    )

                end_groups = {b % S for b, m in metas.items() if m == _STREAM_END}
                for b, metadata in metas.items():
                    if metadata == _STREAM_END:
                        # The stream ended at bucket b-1. Consume exactly one
                        # marker per group or leftovers corrupt the next round.
                        for s in range(S):
                            if s not in end_groups:
                                tail = await _RecvOperation(
                                    group_name=self._group(s),
                                    bucket=self.recv_bufs[s][0],
                                    socket=self.sub_sockets[s],
                                    topic=self.topic,
                                    device=self._device,
                                    executor=self._executor,
                                ).wait_for_complete(self._executor)
                                assert tail == _STREAM_END, f"expected end-of-stream on group {s}, got data"
                        done = True
                        break
                    for tensor_meta in metadata["bucket_meta"].values():
                        yield (
                            tensor_meta,
                            ops[b].bucket[tensor_meta.offset : tensor_meta.offset + tensor_meta.chunk_size],
                        )
                idx += self.phase

        async for name, weight in merge_weight_chunks(chunk_stream(), self.bucket_size):
            yield name, weight

        logger.info(
            f"nccl_parallel receive done in {time.time() - start_time:.2f}s "
            f"(S={S}, phase={self.phase}, bucket={self.bucket_size >> 20}MB)"
        )
