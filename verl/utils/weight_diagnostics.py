# Copyright 2025 Individual Contributor: Steven Zhang
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
"""Opt-in, per-process weight diagnostics; metadata mode never reads tensor values."""

import hashlib
import itertools
import json
import logging
import os
import socket
import sys
import time
from collections import Counter
from pathlib import Path

_counts = Counter()
_warned = False


def _tensor_record(name, tensor, values):
    """Describe a local shard without invoking DTensor collectives."""
    local = getattr(tensor, "_local_tensor", tensor)
    record = {
        "name": name,
        "type": type(tensor).__name__,
        "shape": list(tensor.shape),
        "local_shape": list(local.shape),
        "stride": list(local.stride()),
        "dtype": str(local.dtype),
        "device": str(local.device),
        "placements": str(getattr(tensor, "placements", "")),
        "pinned": local.is_pinned() if local.device.type == "cpu" else None,
    }
    if values and local.device.type != "meta":
        # Explicitly intrusive mode: D2H and host access wait for device work.
        # No full_tensor(): hashes describe LOCAL shards, not global weights.
        cpu = local.detach().to("cpu").contiguous()
        import torch

        raw = cpu.reshape(-1).view(torch.uint8).numpy()
        record["sha256"] = hashlib.sha256(memoryview(raw)).hexdigest()
        if cpu.is_floating_point() or cpu.is_complex():
            record["nonfinite"] = int((~torch.isfinite(cpu)).sum().item())
    return record


def trace_weights(stage, *, model=None, tensors=None, **details):
    """Write bounded local diagnostics, preserving training errors and return values.

    Enabled only with VERL_WEIGHT_DIAGNOSTICS_DIR. Metadata mode performs no
    tensor copies, device reductions, synchronize, or distributed collectives.
    Values mode is intentionally intrusive and must be run as a separate control.
    """
    global _warned
    directory = os.environ.get("VERL_WEIGHT_DIAGNOSTICS_DIR")
    if not directory:
        return
    try:
        limit = int(os.environ.get("VERL_WEIGHT_DIAGNOSTICS_LIMIT", "2"))
        key = (stage, id(model) if model is not None else None)
        if _counts[key] >= limit:
            return
        _counts[key] += 1
        mode = os.environ.get("VERL_WEIGHT_DIAGNOSTICS_MODE", "metadata")
        if mode not in ("metadata", "values"):
            raise ValueError("diagnostics mode must be metadata or values")
        import torch.distributed as dist

        if mode == "values":
            from verl.utils.device import get_torch_device

            get_torch_device().synchronize()
        rank = dist.get_rank() if dist.is_initialized() else None
        event = {
            "stage": stage,
            "call": _counts[key],
            "mode": mode,
            "time_ns": time.time_ns(),
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "rank": rank,
            "details": details,
            "model_id": id(model) if model is not None else None,
            "model_class": type(model).__name__ if model is not None else None,
            "modules": {
                name: {
                    "file": getattr(sys.modules.get(name), "__file__", None),
                    "version": getattr(sys.modules.get(name), "__version__", None),
                }
                for name in ("torch", "torch_npu", "acl", "vllm", "vllm_ascend", "transformers", "verl")
            },
        }
        try:
            event["mapped_libraries"] = sorted(
                {
                    line.split()[-1]
                    for line in Path("/proc/self/maps").read_text().splitlines()
                    if "/" in line and ".so" in line.split()[-1]
                }
            )
        except OSError as exc:
            event["maps_error"] = str(exc)
        if model is not None:
            tensors = itertools.chain(
                (("parameter:" + n, t) for n, t in model.named_parameters()),
                (("buffer:" + n, t) for n, t in model.named_buffers()),
            )
        event["tensors"] = [_tensor_record(n, t, mode == "values") for n, t in tensors or ()]
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        with (path / f"{socket.gethostname()}-{os.getpid()}.jsonl").open("a") as stream:
            stream.write(json.dumps(event, default=str) + "\n")
    except Exception:
        if not _warned:
            logging.getLogger(__name__).warning("Weight diagnostics failed; training continues", exc_info=True)
            _warned = True
