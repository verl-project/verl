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
"""Opt-in test diagnostics. Fingerprinting copies to CPU and changes device timing."""

import hashlib
import json
import os
import socket
from pathlib import Path

import torch


def diagnostic_env(case):
    """Explicitly propagate only diagnostic settings to Ray workers."""
    settings = {
        key: os.environ[key]
        for key in (
            "VERL_CHECKPOINT_DIAGNOSTICS_DIR",
            "VERL_CHECKPOINT_WEIGHT_FINGERPRINTS",
        )
        if key in os.environ
    }
    if settings:
        settings["VERL_CHECKPOINT_DIAGNOSTIC_CASE"] = case
    return settings


def fingerprints_enabled():
    return os.getenv("VERL_CHECKPOINT_WEIGHT_FINGERPRINTS") == "1"


def fingerprint(tensor, dtype=None, chunk_bytes=4 << 20):
    """Hash logical tensor bytes with bounded CPU staging, including BF16 tensors."""
    dtype = dtype or tensor.dtype
    flat = tensor.detach().reshape(-1)
    chunk_elements = max(1, chunk_bytes // max(tensor.element_size(), dtype.itemsize))
    digest = hashlib.sha256()
    for offset in range(0, flat.numel(), chunk_elements):
        chunk = flat[offset : offset + chunk_elements].to(device="cpu", dtype=dtype).contiguous()
        digest.update(memoryview(chunk.view(torch.uint8).numpy()))
    return {"dtype": str(dtype), "shape": list(tensor.shape), "sha256": digest.hexdigest()}


def record(stage, name, update_index, **details):
    """Write a separate JSONL file per worker; never dump tensors or environment values."""
    directory = os.getenv("VERL_CHECKPOINT_DIAGNOSTICS_DIR")
    if not directory:
        return
    entry = {
        "stage": stage,
        "name": name,
        "update_index": update_index,
        "case": os.getenv("VERL_CHECKPOINT_DIAGNOSTIC_CASE", "unspecified"),
        "host": socket.gethostname(),
        "pid": os.getpid(),
        **details,
    }
    try:
        folder = Path(directory)
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / f"{entry['host']}-{entry['pid']}.jsonl").open("a", encoding="utf-8") as output:
            output.write(json.dumps(entry, allow_nan=False) + "\n")
    except (OSError, ValueError) as error:
        print(f"CHECKPOINT_WEIGHT_DIAGNOSTIC write_failed={error!r}", flush=True)


def diagnose_sender(weights, update_index):
    """Observe the actual export generator without replacing its tensors."""
    for name, weight in weights:
        record("sender", name, update_index, **fingerprint(weight))
        yield name, weight


def record_loaded_weights(weights, shard_count):
    """Record rank-zero HF weights before FSDP and the expected Shard(0) slices."""
    for name, weight in weights:
        record("loaded", name, -1, **fingerprint(weight))
        if weight.ndim == 0:
            continue
        # Match DTensor Shard(0), including uneven and empty trailing shards.
        chunks = weight.chunk(shard_count, dim=0)
        for rank in range(shard_count):
            shard = chunks[rank] if rank < len(chunks) else weight[:0]
            record("loaded_shard", name, -1, shard_rank=rank, **fingerprint(shard))


def record_export_shards(params, update_index, rank):
    """Observe FSDP's real state_dict before the later DTensor full_tensor call."""
    from torch.distributed.tensor import DTensor, Shard

    for name, value in params.items():
        if isinstance(value, DTensor):
            supported = value.device_mesh.ndim == 1 and value.placements == (Shard(0),)
            if not supported:
                record("unsupported_shard", name, update_index, shard_rank=rank)
                continue
            record("export_shard", name, update_index, shard_rank=rank, **fingerprint(value.to_local()))


def mismatch_details(expected, received, chunk_bytes=4 << 20):
    """Describe an existing failed assertion, using the same default allclose tolerances."""
    result = {
        "expected_dtype": str(expected.dtype),
        "received_dtype": str(received.dtype),
        "expected_shape": list(expected.shape),
        "received_shape": list(received.shape),
        "compared_dtype": str(received.dtype),
        "numel": received.numel(),
    }
    if expected.shape != received.shape:
        return result
    expected_flat, received_flat = expected.detach().reshape(-1), received.detach().reshape(-1)
    chunk_elements = max(1, chunk_bytes // max(expected.element_size(), received.element_size()))
    mismatched, nonfinite_mismatched, max_abs_finite = 0, 0, 0.0
    for offset in range(0, received.numel(), chunk_elements):
        reference = expected_flat[offset : offset + chunk_elements].to(device="cpu", dtype=received.dtype)
        actual = received_flat[offset : offset + chunk_elements].to(device="cpu")
        bad = ~torch.isclose(reference, actual, rtol=1e-5, atol=1e-8)
        finite = torch.isfinite(reference) & torch.isfinite(actual)
        mismatched += int(bad.sum().item())
        nonfinite_mismatched += int((bad & ~finite).sum().item())
        if (bad & finite).any():
            delta = (reference.double() - actual.double()).abs()[bad & finite]
            max_abs_finite = max(max_abs_finite, float(delta.max().item()))
    result.update(
        cpu_isclose_mismatches=mismatched,
        nonfinite_mismatches=nonfinite_mismatched,
        max_abs_finite=max_abs_finite,
    )
    return result


def summarize(directory):
    """Compare each receiver with the sender and its dtype-normalized HF reference."""
    root = Path(directory)
    senders, references, receivers, failures = {}, {}, [], []
    loaded, loaded_shards, export_shards = {}, {}, []
    unsupported_shards = 0
    unreadable = 0
    for path in sorted(root.rglob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                unreadable += 1
                continue
            key = (entry["case"], entry["update_index"], entry["name"])
            peer = (entry["host"], entry["pid"])
            if entry["stage"] == "sender":
                senders[key] = entry
            elif entry["stage"] == "reference":
                references[key + peer] = entry
            elif entry["stage"] == "received":
                receivers.append((key, peer, entry))
            elif entry["stage"] == "mismatch":
                failures.append({**entry, "record_file": path.relative_to(root).as_posix()})
            elif entry["stage"] == "loaded":
                loaded[(entry["case"], entry["name"])] = entry
            elif entry["stage"] == "loaded_shard":
                loaded_shards[(entry["case"], entry["name"], entry["shard_rank"])] = entry
            elif entry["stage"] == "export_shard":
                export_shards.append(entry)
            elif entry["stage"] == "unsupported_shard":
                unsupported_shards += 1

    def signature(item):
        return item["dtype"], item["shape"], item["sha256"]

    # Report all observed source boundaries, even after receiver assertions stop.
    source_checks = []
    for key, sender in senders.items():
        initial = loaded.get((key[0], key[2]))
        source_checks.append(
            {
                "case": key[0],
                "update_index": key[1],
                "name": key[2],
                "boundary": "loaded_to_sender",
                "equal": signature(initial) == signature(sender) if initial is not None else None,
            }
        )
    for entry in export_shards:
        initial = loaded_shards.get((entry["case"], entry["name"], entry["shard_rank"]))
        source_checks.append(
            {
                "case": entry["case"],
                "update_index": entry["update_index"],
                "name": entry["name"],
                "shard_rank": entry["shard_rank"],
                "boundary": "loaded_to_export_shard",
                "equal": signature(initial) == signature(entry) if initial is not None else None,
            }
        )

    differences, compared, incomplete = [], 0, 0
    for key, peer, received in receivers:
        sender, reference = senders.get(key), references.get(key + peer)
        if sender is None or reference is None:
            incomplete += 1
            continue
        compared += 1

        source_matches = signature(sender) == signature(received)
        reference_matches = signature(reference) == signature(received)
        initial = loaded.get((key[0], key[2]))
        source_checks.append(
            {
                "case": key[0],
                "update_index": key[1],
                "name": key[2],
                "receiver": list(peer),
                "boundary": "loaded_to_reference",
                "equal": signature(initial) == signature(reference) if initial is not None else None,
            }
        )
        if not source_matches or not reference_matches:
            differences.append(
                {
                    "case": key[0],
                    "update_index": key[1],
                    "name": key[2],
                    "receiver": list(peer),
                    "sender_equals_received": source_matches,
                    "received_equals_reference": reference_matches,
                }
            )
    result = {
        "compared": compared,
        "incomplete": incomplete,
        "unreadable": unreadable,
        "differences": differences,
        "assertion_failures": failures,
        "source_checks": source_checks,
        "unsupported_shards": unsupported_shards,
        "source_record_counts": {
            "loaded": len(loaded),
            "loaded_shards": len(loaded_shards),
            "export_shards": len(export_shards),
        },
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "comparison.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    brief = {
        "compared": compared,
        "incomplete": incomplete,
        "unreadable": unreadable,
        "differences": len(differences),
        "assertion_failures": len(failures),
        "source_differences": sum(entry["equal"] is False for entry in source_checks),
        "source_incomplete": sum(entry["equal"] is None for entry in source_checks),
        "export_shards": len(export_shards),
        "unsupported_shards": unsupported_shards,
    }
    print(f"CHECKPOINT_WEIGHT_DIAGNOSTIC {json.dumps(brief)}", flush=True)
    return result
