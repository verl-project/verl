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

"""Merge independently evaluated LoCoMo question shards into one seed result."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from examples.tmem.locomo import load_locomo, score_breakdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("shard_dirs", nargs="+")
    return parser.parse_args()


def merge_rollout_stats(shard_stats: list[dict[str, Any]]) -> dict[str, float | int]:
    block_sizes = {int(stats["dflash_block_size"]) for stats in shard_stats}
    if len(block_sizes) != 1:
        raise ValueError(f"DFlash block-size mismatch across shards: {sorted(block_sizes)}")
    accept_count = sum(int(stats.get("spec_accept_length_count", 0)) for stats in shard_stats)
    accept_sum = sum(
        float(stats.get("mean_spec_accept_length", 0.0)) * int(stats.get("spec_accept_length_count", 0))
        for stats in shard_stats
    )
    return {
        "dflash_block_size": block_sizes.pop(),
        "generation_calls": sum(int(stats.get("generation_calls", 0)) for stats in shard_stats),
        "resumed_request_count": sum(int(stats.get("resumed_request_count", 0)) for stats in shard_stats),
        "generation_gpu_seconds": sum(float(stats.get("generation_seconds", 0.0)) for stats in shard_stats),
        "completion_tokens": sum(int(stats.get("completion_tokens", 0)) for stats in shard_stats),
        "spec_verify_count": sum(int(stats.get("spec_verify_count", 0)) for stats in shard_stats),
        "spec_accept_length_count": accept_count,
        "mean_spec_accept_length": accept_sum / accept_count if accept_count else 0.0,
    }


def main() -> None:
    args = parse_args()
    dataset = load_locomo(args.data)
    ordered_keys = [(sample["sample_id"], qa_index) for sample in dataset for qa_index, _ in enumerate(sample["qa"])]
    expected_order = {key: order for order, key in enumerate(ordered_keys)}
    expected_keys = set(expected_order)

    records_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    shard_payloads = []
    for shard_dir_value in args.shard_dirs:
        shard_dir = Path(shard_dir_value)
        records_path = shard_dir / f"seed_{args.seed}.jsonl"
        payload_path = shard_dir / f"seed_{args.seed}.json"
        for line in records_path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            record = json.loads(line)
            key = (record["sample_id"], int(record["qa_index"]))
            if key in records_by_key:
                raise ValueError(f"Duplicate LoCoMo question across shards: {key}")
            records_by_key[key] = record
        shard_payloads.append(json.loads(payload_path.read_text(encoding="utf-8")))

    actual_keys = set(records_by_key)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        unexpected = sorted(actual_keys - expected_keys)
        raise ValueError(f"Shard coverage mismatch: missing={missing[:10]}, unexpected={unexpected[:10]}")

    records = sorted(
        records_by_key.values(),
        key=lambda record: expected_order[(record["sample_id"], record["qa_index"])],
    )
    shard_metrics = [payload["metrics"] for payload in shard_payloads]
    metrics = score_breakdown(records)
    metrics.update(
        {
            "seed": args.seed,
            "elapsed_seconds": max(float(shard["elapsed_seconds"]) for shard in shard_metrics),
            "rollout": merge_rollout_stats([shard["rollout"] for shard in shard_metrics]),
        }
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / f"seed_{args.seed}.jsonl"
    records_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    payload = {
        "config": {
            "seed": args.seed,
            "parallel_question_shards": [str(Path(path)) for path in args.shard_dirs],
        },
        "paper_target": {"f1": 25.72, "em": 15.40},
        "dataset_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
        "metrics": metrics,
        "records_file": records_path.name,
        "shard_results": shard_payloads,
    }
    (output_dir / f"seed_{args.seed}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    summary = {
        "runs": [metrics],
        "mean_f1": metrics["f1"],
        "mean_em": metrics["em"],
        "std_f1": 0.0,
        "std_em": 0.0,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
