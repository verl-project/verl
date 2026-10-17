# Copyright 2026 verl contributors
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

"""Opt-in diagnostics for Megatron's dynamic pipeline shape exchange."""

import inspect
import json
import math
import os
import socket
import time
from functools import wraps


def _sentinel_shape_exchange(self, tensor_send_next, tensor_send_prev, recv_prev, recv_next, mode, emit):
    """Mirror MCore's three-int64 header protocol, with initialized receive buffers.

    This deliberately synchronizes header initialization before communication.
    Compare sentinel_batch and sentinel_p2p in separate runs on every rank;
    never retry a failed exchange on the same communicator.
    """
    import torch
    import torch.distributed as dist
    from megatron.core.pipeline_parallel.p2p_communication import _p2p_ops

    from verl.utils.device import get_device_id, get_device_name, get_torch_device

    if self.config.use_ring_exchange_p2p:
        raise RuntimeError("Sentinel shape diagnostics do not support ring exchange")
    device = torch.device(get_device_name(), get_device_id())
    sentinel = [-1, -2, -3]
    headers = {}
    for name, enabled in (("recv_prev", recv_prev), ("recv_next", recv_next)):
        headers[name] = torch.tensor(sentinel, dtype=torch.int64, device=device) if enabled else None
    for name, tensor in (("send_prev", tensor_send_prev), ("send_next", tensor_send_next)):
        if tensor is not None and tensor.ndim != 3:
            raise RuntimeError(f"Expected a three-dimensional pipeline payload for {name}, got {tensor.shape}")
        headers[name] = (
            torch.tensor(list(tensor.shape), dtype=torch.int64, device=device) if tensor is not None else None
        )

    get_torch_device().synchronize()
    emit(
        "headers_ready",
        device=str(device),
        headers={name: tensor.tolist() if tensor is not None else None for name, tensor in headers.items()},
        pointers={name: tensor.data_ptr() if tensor is not None else None for name, tensor in headers.items()},
        transport_implementation=(
            f"{dist.batch_isend_irecv.__module__}.{dist.batch_isend_irecv.__qualname__}"
            if mode == "sentinel_batch"
            else f"{_p2p_ops.__module__}.{_p2p_ops.__qualname__}"
        ),
    )
    if mode == "sentinel_batch":
        # Preserve upstream's operation order and process group.
        ops = []
        for name, op, peer in (
            ("send_prev", dist.isend, self.prev_rank),
            ("recv_prev", dist.irecv, self.prev_rank),
            ("send_next", dist.isend, self.next_rank),
            ("recv_next", dist.irecv, self.next_rank),
        ):
            if headers[name] is not None:
                ops.append(dist.P2POp(op, headers[name], peer, self.pp_group))
        reqs = dist.batch_isend_irecv(ops) if ops else []
    else:
        # Use MCore's parity-ordered ordinary P2P operations to avoid send/send deadlocks.
        reqs = _p2p_ops(
            tensor_send_prev=headers["send_prev"],
            tensor_recv_prev=headers["recv_prev"],
            tensor_send_next=headers["send_next"],
            tensor_recv_next=headers["recv_next"],
            group=self.pp_group,
            prev_pipeline_rank=self.prev_rank,
            next_pipeline_rank=self.next_rank,
        )
    emit("headers_submitted", request_count=len(reqs))
    for req in reqs.values() if isinstance(reqs, dict) else reqs:
        req.wait()
    get_torch_device().synchronize()
    values = {name: tensor.tolist() if tensor is not None else None for name, tensor in headers.items()}
    emit(
        "headers_completed",
        headers=values,
        sentinel_positions={
            name: [i for i, value in enumerate(values[name]) if value == sentinel[i]]
            for name in ("recv_prev", "recv_next")
            if values[name] is not None
        },
    )
    return values["recv_prev"] or [0, 0, 0], values["recv_next"] or [0, 0, 0]


def install_p2p_shape_trace():
    """Trace the active shape method, optionally replacing it for a controlled experiment.

    Enable with VERL_P2P_SHAPE_TRACE=1. Records go to worker stdout so Ray job
    logs retain them even when the worker fails. The diagnostic allocation cap
    defaults to 1 TiB and can be set with VERL_P2P_SHAPE_MAX_BYTES.
    VERL_P2P_SHAPE_MODE selects original (unchanged transport), sentinel_batch,
    or sentinel_p2p. Both sentinel modes add initialization and synchronization;
    a passing run alone does not establish which underlying bug was responsible.
    """
    if os.environ.get("VERL_P2P_SHAPE_TRACE") != "1":
        return

    import torch.distributed as dist
    from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator

    original = P2PCommunicator._communicate_shapes
    if getattr(original, "_verl_shape_trace", False):
        return
    limit = int(os.environ.get("VERL_P2P_SHAPE_MAX_BYTES", str(1 << 40)))
    if limit <= 0:
        raise ValueError("VERL_P2P_SHAPE_MAX_BYTES must be positive")
    mode = os.environ.get("VERL_P2P_SHAPE_MODE", "original")
    if mode not in ("original", "sentinel_batch", "sentinel_p2p"):
        raise ValueError(f"Unknown VERL_P2P_SHAPE_MODE: {mode}")
    if mode != "original" and original.__module__ != P2PCommunicator.__module__:
        raise RuntimeError("Sentinel diagnostics require MCore's native three-int64 shape exchange")

    @wraps(original)
    def traced(self, tensor_send_next, tensor_send_prev, recv_prev, recv_next):
        call = getattr(self, "_verl_shape_trace_call", 0) + 1
        self._verl_shape_trace_call = call
        context = {
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "rank": dist.get_rank(),
            "pp_ranks": dist.get_process_group_ranks(self.pp_group),
            "pp_rank": self.pp_group.rank(),
            "prev_rank": self.prev_rank,
            "next_rank": self.next_rank,
            "communicator": id(self),
            "call": call,
            "backend": str(dist.get_backend(self.pp_group)),
            "shape_mode": mode,
            "batch_p2p_comm": self.config.batch_p2p_comm,
            "variable_seq_lengths": self.config.variable_seq_lengths,
            "dtype": str(self.config.pipeline_dtype),
            "send_prev_shape": list(tensor_send_prev.shape) if tensor_send_prev is not None else None,
            "send_next_shape": list(tensor_send_next.shape) if tensor_send_next is not None else None,
            "recv_prev": recv_prev,
            "recv_next": recv_next,
        }

        def emit(event, **fields):
            # RAY_DEDUP_LOGS=0 must be set before the driver imports Ray to retain every record.
            print(
                "[P2P_SHAPE_TRACE] "
                + json.dumps({**context, "event": event, "time_ns": time.time_ns(), **fields}, sort_keys=True),
                flush=True,
            )

        emit("exchange_start", implementation=f"{inspect.getsourcefile(original)}:{original.__qualname__}")
        try:
            if mode == "original":
                shapes = original(self, tensor_send_next, tensor_send_prev, recv_prev, recv_next)
            else:
                shapes = _sentinel_shape_exchange(
                    self, tensor_send_next, tensor_send_prev, recv_prev, recv_next, mode, emit
                )
        except Exception as exc:
            emit("exchange_error", error=repr(exc))
            raise

        emit("exchange_done", recv_prev_shape=list(shapes[0]), recv_next_shape=list(shapes[1]))
        for direction, enabled, shape in zip(("prev", "next"), (recv_prev, recv_next), shapes, strict=True):
            if not enabled:
                continue
            # Use Python integers so a corrupt header cannot overflow the byte calculation.
            dimensions = [int(dim) for dim in shape]
            nbytes = math.prod(dimensions) * self.config.pipeline_dtype.itemsize
            if not dimensions or any(dim < 0 for dim in dimensions) or nbytes > limit:
                emit(
                    "invalid_shape",
                    direction=direction,
                    shape=dimensions,
                    shape_hex=[hex(dim) for dim in dimensions],
                    requested_bytes=nbytes,
                    limit_bytes=limit,
                )
                raise RuntimeError(
                    f"Invalid pipeline receive shape: rank={context['rank']} direction={direction} "
                    f"shape={dimensions} bytes={nbytes} diagnostic_limit={limit}; see P2P_SHAPE_TRACE logs"
                )
        return shapes

    traced._verl_shape_trace = True
    P2PCommunicator._communicate_shapes = traced
