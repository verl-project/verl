#!/usr/bin/env python3
"""Aggregate active CUDA allocations in a ``memory_snapshot`` JSON file.

The profiler writes the return value of ``torch.cuda.memory_snapshot`` as JSON.
PyTorch has changed the exact frame container a few times, so this reader
accepts both the current ``segments[].blocks[]`` shape and older block records
with a ``trace`` field.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def _frames(record: dict[str, Any]) -> list[Any]:
    value = record.get("frames", record.get("trace", []))
    return value if isinstance(value, list) else []


def _blocks(snapshot: Any) -> Iterable[dict[str, Any]]:
    if isinstance(snapshot, dict):
        blocks = snapshot.get("blocks")
        if isinstance(blocks, list):
            yield from (block for block in blocks if isinstance(block, dict))
            return
        segments = snapshot.get("segments")
        if isinstance(segments, list):
            for segment in segments:
                if isinstance(segment, dict):
                    blocks = segment.get("blocks", [])
                    if isinstance(blocks, list):
                        yield from (block for block in blocks if isinstance(block, dict))
        return
    if isinstance(snapshot, list):
        for segment in snapshot:
            yield from _blocks(segment)


def _frame_label(frame: Any) -> str:
    if isinstance(frame, dict):
        filename = frame.get("filename", frame.get("file", "<unknown>"))
        line = frame.get("line", frame.get("lineno", "?"))
        return f"{filename}:{line}"
    if isinstance(frame, str):
        return frame
    return "<unknown>"


def aggregate(snapshot: Any) -> dict[str, int]:
    totals: defaultdict[str, int] = defaultdict(int)
    for block in _blocks(snapshot):
        if block.get("state") not in (None, "active_allocated"):
            continue
        frames = _frames(block)
        label = _frame_label(frames[0]) if frames else "<unknown>"
        size = block.get("requested_size", block.get("size", block.get("real_size", 0)))
        if isinstance(size, int):
            totals[label] += size
    return dict(totals)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path, help="JSON written by DSV41_MEM_SNAPSHOT")
    parser.add_argument("--limit", type=int, default=20, help="number of allocation points to print")
    args = parser.parse_args()
    with args.snapshot.open(encoding="utf-8") as handle:
        snapshot = json.load(handle)
    totals = aggregate(snapshot)
    print(f"[mem-snapshot] allocations={sum(totals.values())} bytes points={len(totals)}")
    for label, size in sorted(totals.items(), key=lambda item: item[1], reverse=True)[: args.limit]:
        print(f"[mem-snapshot] {label} bytes={size} ({size / (1 << 30):.2f}GB)")


if __name__ == "__main__":
    main()
