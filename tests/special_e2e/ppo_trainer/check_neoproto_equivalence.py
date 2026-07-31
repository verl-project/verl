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
"""Fail-closed DataProto/NeoProto deterministic E2E equivalence checker."""

from __future__ import annotations

import argparse
import json
import math
import re
import socket
from pathlib import Path
from typing import Any

import numpy as np
import torch

_STEP_RE = re.compile(r"(?:^|\s)step:(\d+)\s+-\s+")
_FORBIDDEN_LOG_MARKERS = (
    "Traceback (most recent call last)",
    "CUDA out of memory",
    "RayTaskError",
    "ActorDiedError",
    "Error executing job",
)
_REQUIRED_METRICS = (
    "actor/pg_loss",
    "actor/grad_norm",
    "critic/vf_loss",
    "critic/score/mean",
    "critic/rewards/mean",
    "critic/advantages/mean",
    "critic/returns/mean",
    "critic/values/mean",
    "response_length/mean",
    "prompt_length/mean",
)
_IGNORED_METRIC_PREFIXES = (
    "timing_s/",
    "timing_per_token_ms/",
    "perf/mfu/",
    "perf/throughput",
    "perf/max_memory",
    "perf/time_per_step",
)
_RESOURCE_TRACKER_PSM_RE = re.compile(r"KeyError: '/psm_[0-9a-f]+'")


def _strip_known_resource_tracker_noise(text: str) -> tuple[str, int]:
    """Remove only the known vLLM/Python shared-memory tracker traceback.

    Python's resource tracker can emit a non-fatal ``KeyError('/psm_*')`` when
    vLLM workers unregister the same shared-memory segment during teardown.  Do
    not suppress generic tracebacks: the four fingerprinted line groups must be
    present one-for-one, and every traceback must come from a vLLM HTTP server.
    """
    lines = text.splitlines()
    traceback_lines = [line for line in lines if "Traceback (most recent call last)" in line]
    if not traceback_lines:
        return text, 0
    count = len(traceback_lines)
    if any("vLLMHttpServer" not in line for line in traceback_lines):
        return text, 0

    resource_file_lines = [line for line in lines if 'multiprocessing/resource_tracker.py", line 239, in main' in line]
    cache_remove_lines = [line for line in lines if "cache[rtype].remove(name)" in line]
    psm_keyerror_lines = [line for line in lines if _RESOURCE_TRACKER_PSM_RE.search(line)]
    if not (len(resource_file_lines) == len(cache_remove_lines) == len(psm_keyerror_lines) == count):
        return text, 0

    filtered = [
        line
        for line in lines
        if not (
            ("Traceback (most recent call last)" in line and "vLLMHttpServer" in line)
            or 'multiprocessing/resource_tracker.py", line 239, in main' in line
            or "cache[rtype].remove(name)" in line
            or _RESOURCE_TRACKER_PSM_RE.search(line)
        )
    ]
    return "\n".join(filtered), count


def _parse_float(text: str) -> float:
    text = text.strip()
    wrapper = re.fullmatch(r"(?:np\.)?float(?:16|32|64)?\((.*)\)", text)
    if wrapper:
        text = wrapper.group(1)
    return float(text)


def _parse_metrics(path: Path, expected_steps: int) -> tuple[str, dict[int, dict[str, float]]]:
    if not path.is_file():
        raise AssertionError(f"Missing training log: {path}")
    text = path.read_text(errors="replace")
    scan_text, ignored_resource_tracker_count = _strip_known_resource_tracker_noise(text)
    for marker in _FORBIDDEN_LOG_MARKERS:
        if marker in scan_text:
            raise AssertionError(f"{path} contains failure marker {marker!r}")
    if ignored_resource_tracker_count:
        print(f"IGNORED_KNOWN_RESOURCE_TRACKER_TRACEBACKS path={path} count={ignored_resource_tracker_count}")

    steps: dict[int, dict[str, float]] = {}
    for line in text.splitlines():
        match = _STEP_RE.search(line)
        if match is None:
            continue
        step = int(match.group(1))
        metrics: dict[str, float] = {}
        for field in line[match.end() :].split(" - "):
            if ":" not in field:
                continue
            key, raw_value = field.split(":", 1)
            try:
                metrics[key.strip()] = _parse_float(raw_value)
            except ValueError:
                continue
        if step in steps:
            raise AssertionError(f"{path} contains duplicate metrics for step {step}")
        steps[step] = metrics

    expected = list(range(1, expected_steps + 1))
    if sorted(steps) != expected:
        raise AssertionError(f"{path} steps are {sorted(steps)}, expected {expected}")
    for step, metrics in steps.items():
        if not metrics:
            raise AssertionError(f"{path} step {step} has no numeric metrics")
        bad = {key: value for key, value in metrics.items() if not math.isfinite(value)}
        if bad:
            raise AssertionError(f"{path} step {step} has non-finite metrics: {bad}")
        missing = sorted(set(_REQUIRED_METRICS) - metrics.keys())
        if missing:
            raise AssertionError(f"{path} step {step} is missing correctness metrics: {missing}")
    if not any(abs(metrics["actor/grad_norm"]) > 0 for metrics in steps.values()):
        raise AssertionError(f"{path} has zero actor gradient on every step")
    return text, steps


def _is_semantic_metric(key: str) -> bool:
    return not key.startswith(_IGNORED_METRIC_PREFIXES) and "/perf/" not in key


def _compare_metrics(
    baseline: dict[int, dict[str, float]],
    neo: dict[int, dict[str, float]],
    *,
    rtol: float,
    atol: float,
) -> int:
    compared = 0
    for step in baseline:
        baseline_metrics = {key: value for key, value in baseline[step].items() if _is_semantic_metric(key)}
        neo_metrics = {key: value for key, value in neo[step].items() if _is_semantic_metric(key)}
        if baseline_metrics.keys() != neo_metrics.keys():
            missing_in_neo = sorted(baseline_metrics.keys() - neo_metrics.keys())
            extra_in_neo = sorted(neo_metrics.keys() - baseline_metrics.keys())
            raise AssertionError(
                f"Semantic metric key mismatch at step {step}: "
                f"missing_in_neo={missing_in_neo}, extra_in_neo={extra_in_neo}"
            )
        for key, baseline_value in baseline_metrics.items():
            neo_value = neo_metrics[key]
            if not math.isclose(baseline_value, neo_value, rel_tol=rtol, abs_tol=atol):
                raise AssertionError(
                    f"Metric mismatch at step {step}, {key}: "
                    f"DataProto={baseline_value}, NeoProto={neo_value}, rtol={rtol}, atol={atol}"
                )
            compared += 1
    return compared


def _assert_neoproto_active(baseline_text: str, neo_text: str, neo_metrics: dict[int, dict[str, float]]) -> None:
    marker = "RayPPOTrainer data_proto_cls=verl.experimental.neoproto.views.data_proto.DataProto"
    strict_marker = "NEOPROTO_STRICT_MODE=enabled dispatch=enabled full_materialize=disabled"
    if marker in baseline_text or strict_marker in baseline_text:
        raise AssertionError("DataProto baseline unexpectedly enabled NeoProto")
    if marker not in neo_text:
        raise AssertionError("NeoProto class marker is missing")
    if strict_marker not in neo_text:
        raise AssertionError("NeoProto strict dispatch marker is missing")
    if "timing_s/dataplane/materialize_calls" in baseline_text:
        raise AssertionError("DataProto baseline unexpectedly emitted Neo materialize counters")
    for step, metrics in neo_metrics.items():
        calls = metrics.get("timing_s/dataplane/materialize_calls")
        if calls is None or calls <= 0:
            raise AssertionError(f"NeoProto step {step} did not materialize any Neo refs: {calls}")
        if "timing_s/dataplane/prefetch_gen" not in metrics:
            raise AssertionError(f"NeoProto step {step} is missing the Neo prefetch marker")


def _assert_tensor_finite(tensor: torch.Tensor, path: str) -> None:
    if tensor.is_floating_point() or tensor.is_complex():
        if not torch.isfinite(tensor).all().item():
            raise AssertionError(f"Non-finite tensor at {path}")


def _local_tensor(value: Any) -> Any:
    if hasattr(value, "to_local") and callable(value.to_local):
        return value.to_local()
    if hasattr(value, "local_tensor") and callable(value.local_tensor):
        return value.local_tensor()
    return value


def _compare_objects(
    baseline: Any,
    neo: Any,
    *,
    path: str,
    rtol: float,
    atol: float,
) -> int:
    baseline = _local_tensor(baseline)
    neo = _local_tensor(neo)

    if isinstance(baseline, torch.Tensor) or isinstance(neo, torch.Tensor):
        if not isinstance(baseline, torch.Tensor) or not isinstance(neo, torch.Tensor):
            raise AssertionError(f"Type mismatch at {path}: {type(baseline)} vs {type(neo)}")
        if baseline.dtype != neo.dtype or baseline.shape != neo.shape:
            raise AssertionError(
                f"Tensor metadata mismatch at {path}: "
                f"{baseline.dtype}/{tuple(baseline.shape)} vs {neo.dtype}/{tuple(neo.shape)}"
            )
        baseline = baseline.detach().cpu()
        neo = neo.detach().cpu()
        _assert_tensor_finite(baseline, f"{path} (DataProto)")
        _assert_tensor_finite(neo, f"{path} (NeoProto)")
        if rtol == 0 and atol == 0:
            equal = torch.equal(baseline, neo)
        else:
            equal = torch.allclose(baseline, neo, rtol=rtol, atol=atol, equal_nan=False)
        if not equal:
            max_abs = (
                torch.max(torch.abs(baseline.to(torch.float64) - neo.to(torch.float64))).item()
                if baseline.numel()
                else 0.0
            )
            raise AssertionError(f"Tensor mismatch at {path}: max_abs={max_abs}, rtol={rtol}, atol={atol}")
        return 1

    if hasattr(baseline, "local_shards") or hasattr(neo, "local_shards"):
        if not hasattr(baseline, "local_shards") or not hasattr(neo, "local_shards"):
            raise AssertionError(f"Sharded tensor type mismatch at {path}")
        baseline_shards = baseline.local_shards()
        neo_shards = neo.local_shards()
        if len(baseline_shards) != len(neo_shards):
            raise AssertionError(f"Local shard count mismatch at {path}")
        count = 0
        for index, (baseline_shard, neo_shard) in enumerate(zip(baseline_shards, neo_shards, strict=True)):
            if repr(baseline_shard.metadata) != repr(neo_shard.metadata):
                raise AssertionError(f"Shard metadata mismatch at {path}[{index}]")
            count += _compare_objects(
                baseline_shard.tensor,
                neo_shard.tensor,
                path=f"{path}.local_shards[{index}]",
                rtol=rtol,
                atol=atol,
            )
        return count

    if type(baseline) is not type(neo):
        raise AssertionError(f"Type mismatch at {path}: {type(baseline)} vs {type(neo)}")
    if isinstance(baseline, dict):
        if baseline.keys() != neo.keys():
            raise AssertionError(f"Dictionary key mismatch at {path}")
        return sum(
            _compare_objects(
                baseline[key],
                neo[key],
                path=f"{path}.{key}",
                rtol=rtol,
                atol=atol,
            )
            for key in baseline
        )
    if isinstance(baseline, list | tuple):
        if len(baseline) != len(neo):
            raise AssertionError(f"Sequence length mismatch at {path}")
        return sum(
            _compare_objects(a, b, path=f"{path}[{index}]", rtol=rtol, atol=atol)
            for index, (a, b) in enumerate(zip(baseline, neo, strict=True))
        )
    if isinstance(baseline, np.ndarray):
        if baseline.dtype != neo.dtype or baseline.shape != neo.shape:
            raise AssertionError(f"Array metadata mismatch at {path}")
        if baseline.dtype == object:
            return sum(
                _compare_objects(a, b, path=f"{path}[{index}]", rtol=rtol, atol=atol)
                for index, (a, b) in enumerate(zip(baseline.flat, neo.flat, strict=True))
            )
        if np.issubdtype(baseline.dtype, np.floating):
            if not np.isfinite(baseline).all() or not np.isfinite(neo).all():
                raise AssertionError(f"Non-finite array at {path}")
            equal = np.allclose(baseline, neo, rtol=rtol, atol=atol, equal_nan=False)
        else:
            equal = np.array_equal(baseline, neo)
        if not equal:
            raise AssertionError(f"Array mismatch at {path}")
        return 1
    if isinstance(baseline, float):
        if not math.isfinite(baseline) or not math.isfinite(neo):
            raise AssertionError(f"Non-finite scalar at {path}")
        if not math.isclose(baseline, neo, rel_tol=rtol, abs_tol=atol):
            raise AssertionError(f"Scalar mismatch at {path}: {baseline} vs {neo}")
        return 1
    if baseline != neo:
        raise AssertionError(f"Value mismatch at {path}: {baseline!r} vs {neo!r}")
    return 1


def _compare_torch_directories(
    baseline_dir: Path,
    neo_dir: Path,
    *,
    expected_steps: int,
    rtol: float,
    atol: float,
    checkpoint: bool,
) -> int:
    if not baseline_dir.is_dir() or not neo_dir.is_dir():
        raise AssertionError(f"Missing comparison directories: {baseline_dir}, {neo_dir}")
    if checkpoint:
        baseline_files = {path.relative_to(baseline_dir) for path in baseline_dir.rglob("*.pt")}
        neo_files = {path.relative_to(neo_dir) for path in neo_dir.rglob("*.pt")}
    else:
        baseline_files = {Path(f"{step}.pt") for step in range(1, expected_steps + 1)}
        neo_files = set(baseline_files)
    if baseline_files != neo_files:
        raise AssertionError(
            f"Torch artifact file mismatch: "
            f"missing_in_neo={sorted(baseline_files - neo_files)}, "
            f"extra_in_neo={sorted(neo_files - baseline_files)}"
        )
    if not baseline_files:
        raise AssertionError(f"No .pt artifacts found under {baseline_dir}")
    if checkpoint:
        required = ("actor/model_", "actor/optim_", "critic/model_", "critic/optim_")
        relative_names = [str(path) for path in baseline_files]
        missing = [fragment for fragment in required if not any(fragment in name for name in relative_names)]
        if missing:
            raise AssertionError(f"Native checkpoint is incomplete; missing patterns: {missing}")

    count = 0
    for relative_path in sorted(baseline_files):
        baseline_value = torch.load(baseline_dir / relative_path, map_location="cpu", weights_only=False)
        neo_value = torch.load(neo_dir / relative_path, map_location="cpu", weights_only=False)
        count += _compare_objects(
            baseline_value,
            neo_value,
            path=str(relative_path),
            rtol=rtol,
            atol=atol,
        )
    return count


def _read_torch_snapshots(path: Path, expected_steps: int) -> dict[str, Any] | None:
    """Read a complete per-step snapshot set, or return ``None`` for an empty node."""
    expected_files = {Path(f"{step}.pt") for step in range(1, expected_steps + 1)}
    actual_files = {file_path.relative_to(path) for file_path in path.glob("*.pt")} if path.is_dir() else set()
    if not actual_files:
        return None
    if actual_files != expected_files:
        raise AssertionError(
            f"Incomplete tensor snapshots under {path}: "
            f"missing={sorted(expected_files - actual_files)}, extra={sorted(actual_files - expected_files)}"
        )
    return {
        str(relative_path): torch.load(path / relative_path, map_location="cpu", weights_only=False)
        for relative_path in sorted(actual_files)
    }


def _compare_loaded_torch_files(
    baseline: dict[str, Any],
    neo: dict[str, Any],
    *,
    rtol: float,
    atol: float,
) -> int:
    if baseline.keys() != neo.keys():
        raise AssertionError(
            "Loaded torch artifact mismatch: "
            f"missing_in_neo={sorted(baseline.keys() - neo.keys())}, "
            f"extra_in_neo={sorted(neo.keys() - baseline.keys())}"
        )
    return sum(
        _compare_objects(
            baseline[name],
            neo[name],
            path=name,
            rtol=rtol,
            atol=atol,
        )
        for name in baseline
    )


def _collect_distributed_node(
    baseline_dump_dir: str | None,
    neo_dump_dir: str | None,
    baseline_rollout_dir: str | None,
    neo_rollout_dir: str | None,
    baseline_checkpoint_dir: str | None,
    neo_checkpoint_dir: str | None,
    expected_steps: int,
    tensor_rtol: float,
    tensor_atol: float,
) -> dict[str, Any]:
    """Collect small artifacts and compare rank-local checkpoint shards on one Ray node."""
    output: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "baseline_snapshots": None,
        "neo_snapshots": None,
        "baseline_rollouts": None,
        "neo_rollouts": None,
        "baseline_checkpoint_globals": None,
        "neo_checkpoint_globals": None,
        "checkpoint_shard_files": [],
        "checkpoint_values_compared": 0,
    }
    if baseline_dump_dir is not None:
        output["baseline_snapshots"] = _read_torch_snapshots(Path(baseline_dump_dir), expected_steps)
        output["neo_snapshots"] = _read_torch_snapshots(Path(neo_dump_dir), expected_steps)
    if baseline_rollout_dir is not None:
        baseline_rollout_path = Path(baseline_rollout_dir)
        neo_rollout_path = Path(neo_rollout_dir)
        if baseline_rollout_path.is_dir() and any(baseline_rollout_path.glob("*.jsonl")):
            output["baseline_rollouts"] = _read_jsonl_dir(baseline_rollout_path, expected_steps)
        if neo_rollout_path.is_dir() and any(neo_rollout_path.glob("*.jsonl")):
            output["neo_rollouts"] = _read_jsonl_dir(neo_rollout_path, expected_steps)
    if baseline_checkpoint_dir is None:
        return output

    baseline_checkpoint_path = Path(baseline_checkpoint_dir)
    neo_checkpoint_path = Path(neo_checkpoint_dir)
    baseline_files = (
        {path.relative_to(baseline_checkpoint_path) for path in baseline_checkpoint_path.rglob("*.pt")}
        if baseline_checkpoint_path.is_dir()
        else set()
    )
    neo_files = (
        {path.relative_to(neo_checkpoint_path) for path in neo_checkpoint_path.rglob("*.pt")}
        if neo_checkpoint_path.is_dir()
        else set()
    )
    baseline_shards = {path for path in baseline_files if "_rank_" in path.name}
    neo_shards = {path for path in neo_files if "_rank_" in path.name}
    if baseline_shards != neo_shards:
        raise AssertionError(
            f"Rank-local checkpoint mismatch on {socket.gethostname()}: "
            f"missing_in_neo={sorted(baseline_shards - neo_shards)}, "
            f"extra_in_neo={sorted(neo_shards - baseline_shards)}"
        )
    for relative_path in sorted(baseline_shards):
        baseline_value = torch.load(baseline_checkpoint_path / relative_path, map_location="cpu", weights_only=False)
        neo_value = torch.load(neo_checkpoint_path / relative_path, map_location="cpu", weights_only=False)
        output["checkpoint_values_compared"] += _compare_objects(
            baseline_value,
            neo_value,
            path=str(relative_path),
            rtol=tensor_rtol,
            atol=tensor_atol,
        )
    output["checkpoint_shard_files"] = [str(path) for path in sorted(baseline_shards)]

    baseline_globals = baseline_files - baseline_shards
    neo_globals = neo_files - neo_shards
    if baseline_globals:
        output["baseline_checkpoint_globals"] = {
            str(relative_path): torch.load(
                baseline_checkpoint_path / relative_path, map_location="cpu", weights_only=False
            )
            for relative_path in sorted(baseline_globals)
        }
    if neo_globals:
        output["neo_checkpoint_globals"] = {
            str(relative_path): torch.load(neo_checkpoint_path / relative_path, map_location="cpu", weights_only=False)
            for relative_path in sorted(neo_globals)
        }
    return output


def _exactly_one_node_value(results: list[dict[str, Any]], key: str) -> Any:
    matches = [(result["hostname"], result[key]) for result in results if result[key] is not None]
    if len(matches) != 1:
        raise AssertionError(f"Expected {key} on exactly one Ray node, found {[host for host, _ in matches]}")
    return matches[0][1]


def _read_jsonl_dir(path: Path, expected_steps: int) -> dict[int, list[dict[str, Any]]]:
    output = {}
    for step in range(1, expected_steps + 1):
        file_path = path / f"{step}.jsonl"
        if not file_path.is_file():
            raise AssertionError(f"Missing rollout dump: {file_path}")
        records = []
        for line in file_path.read_text().splitlines():
            record = json.loads(line)
            record.pop("request_id", None)
            records.append(record)
        if not records:
            raise AssertionError(f"Empty rollout dump: {file_path}")
        output[step] = records
    return output


def _compare_rollouts(baseline_dir: Path, neo_dir: Path, expected_steps: int) -> int:
    baseline = _read_jsonl_dir(baseline_dir, expected_steps)
    neo = _read_jsonl_dir(neo_dir, expected_steps)
    for step in baseline:
        if baseline[step] != neo[step]:
            raise AssertionError(f"Decoded rollout JSONL differs at step {step}")
    return sum(len(records) for records in baseline.values())


def _compare_loaded_rollouts(baseline: dict[int, list[dict[str, Any]]], neo: dict[int, list[dict[str, Any]]]) -> int:
    if baseline.keys() != neo.keys():
        raise AssertionError(
            f"Rollout step mismatch: missing_in_neo={sorted(baseline.keys() - neo.keys())}, "
            f"extra_in_neo={sorted(neo.keys() - baseline.keys())}"
        )
    for step in baseline:
        if baseline[step] != neo[step]:
            raise AssertionError(f"Decoded rollout JSONL differs at step {step}")
    return sum(len(records) for records in baseline.values())


def _validate_checkpoint_shard_coverage(shard_files: list[str]) -> None:
    if len(shard_files) != len(set(shard_files)):
        raise AssertionError("The same checkpoint shard exists on more than one Ray node")
    for role in ("actor", "critic"):
        for kind in ("model", "optim", "extra_state"):
            pattern = re.compile(rf"^{role}/{kind}_world_size_(\d+)_rank_(\d+)\.pt$")
            matches = [pattern.fullmatch(path) for path in shard_files]
            parsed = [(int(match.group(1)), int(match.group(2))) for match in matches if match is not None]
            if not parsed:
                raise AssertionError(f"Distributed checkpoint is missing {role}/{kind} shards")
            world_sizes = {world_size for world_size, _ in parsed}
            if len(world_sizes) != 1:
                raise AssertionError(f"Inconsistent world sizes for {role}/{kind}: {sorted(world_sizes)}")
            world_size = world_sizes.pop()
            ranks = {rank for _, rank in parsed}
            if ranks != set(range(world_size)):
                raise AssertionError(
                    f"Incomplete ranks for {role}/{kind}: got {sorted(ranks)}, expected {list(range(world_size))}"
                )


def _compare_distributed_artifacts(args: argparse.Namespace) -> tuple[int, int, int]:
    """Compare node-local artifacts across every live GPU node in an existing Ray cluster."""
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    ray.init(address=args.distributed_ray_address, ignore_reinit_error=True, logging_level="ERROR")
    nodes = [node for node in ray.nodes() if node["Alive"] and node["Resources"].get("GPU", 0) > 0]
    if not nodes:
        raise AssertionError("No live GPU nodes found for distributed artifact comparison")
    collector = ray.remote(num_cpus=0)(_collect_distributed_node)
    refs = [
        collector.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node["NodeID"], soft=False)
        ).remote(
            str(args.baseline_dump_dir) if args.baseline_dump_dir is not None else None,
            str(args.neo_dump_dir) if args.neo_dump_dir is not None else None,
            str(args.baseline_rollout_dir) if args.baseline_rollout_dir is not None else None,
            str(args.neo_rollout_dir) if args.neo_rollout_dir is not None else None,
            str(args.baseline_checkpoint_dir) if args.baseline_checkpoint_dir is not None else None,
            str(args.neo_checkpoint_dir) if args.neo_checkpoint_dir is not None else None,
            args.expected_steps,
            args.tensor_rtol,
            args.tensor_atol,
        )
        for node in nodes
    ]
    results = ray.get(refs)

    rollout_count = 0
    if args.baseline_rollout_dir is not None:
        rollout_count = _compare_loaded_rollouts(
            _exactly_one_node_value(results, "baseline_rollouts"),
            _exactly_one_node_value(results, "neo_rollouts"),
        )
    snapshot_tensor_count = 0
    if args.baseline_dump_dir is not None:
        snapshot_tensor_count = _compare_loaded_torch_files(
            _exactly_one_node_value(results, "baseline_snapshots"),
            _exactly_one_node_value(results, "neo_snapshots"),
            rtol=args.tensor_rtol,
            atol=args.tensor_atol,
        )
    checkpoint_tensor_count = 0
    if args.baseline_checkpoint_dir is not None:
        shard_files = [path for result in results for path in result["checkpoint_shard_files"]]
        _validate_checkpoint_shard_coverage(shard_files)
        checkpoint_tensor_count = sum(result["checkpoint_values_compared"] for result in results)
        checkpoint_tensor_count += _compare_loaded_torch_files(
            _exactly_one_node_value(results, "baseline_checkpoint_globals"),
            _exactly_one_node_value(results, "neo_checkpoint_globals"),
            rtol=args.tensor_rtol,
            atol=args.tensor_atol,
        )
    return rollout_count, snapshot_tensor_count, checkpoint_tensor_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-log", required=True, type=Path)
    parser.add_argument("--neo-log", required=True, type=Path)
    parser.add_argument("--baseline-dump-dir", type=Path)
    parser.add_argument("--neo-dump-dir", type=Path)
    parser.add_argument("--baseline-rollout-dir", type=Path)
    parser.add_argument("--neo-rollout-dir", type=Path)
    parser.add_argument("--baseline-checkpoint-dir", type=Path)
    parser.add_argument("--neo-checkpoint-dir", type=Path)
    parser.add_argument("--expected-steps", type=int, default=2)
    parser.add_argument("--metric-rtol", type=float, default=1e-5)
    parser.add_argument("--metric-atol", type=float, default=1e-6)
    parser.add_argument("--tensor-rtol", type=float, default=0.0)
    parser.add_argument("--tensor-atol", type=float, default=0.0)
    parser.add_argument(
        "--distributed-ray-address",
        help="Compare node-local artifacts across all live GPU nodes (for example: auto)",
    )
    args = parser.parse_args()

    baseline_text, baseline_metrics = _parse_metrics(args.baseline_log, args.expected_steps)
    neo_text, neo_metrics = _parse_metrics(args.neo_log, args.expected_steps)
    _assert_neoproto_active(baseline_text, neo_text, neo_metrics)
    metric_count = _compare_metrics(
        baseline_metrics,
        neo_metrics,
        rtol=args.metric_rtol,
        atol=args.metric_atol,
    )
    artifact_pairs = (
        ("tensor dump", args.baseline_dump_dir, args.neo_dump_dir),
        ("rollout dump", args.baseline_rollout_dir, args.neo_rollout_dir),
        ("checkpoint", args.baseline_checkpoint_dir, args.neo_checkpoint_dir),
    )
    for label, baseline_path, neo_path in artifact_pairs:
        if (baseline_path is None) != (neo_path is None):
            raise AssertionError(f"Both {label} paths must be provided together")

    if args.distributed_ray_address:
        rollout_count, snapshot_tensor_count, checkpoint_tensor_count = _compare_distributed_artifacts(args)
    else:
        rollout_count = 0
        if args.baseline_rollout_dir is not None:
            rollout_count = _compare_rollouts(
                args.baseline_rollout_dir,
                args.neo_rollout_dir,
                args.expected_steps,
            )
        snapshot_tensor_count = 0
        if args.baseline_dump_dir is not None:
            snapshot_tensor_count = _compare_torch_directories(
                args.baseline_dump_dir,
                args.neo_dump_dir,
                expected_steps=args.expected_steps,
                rtol=args.tensor_rtol,
                atol=args.tensor_atol,
                checkpoint=False,
            )
        checkpoint_tensor_count = 0
        if args.baseline_checkpoint_dir is not None:
            checkpoint_tensor_count = _compare_torch_directories(
                args.baseline_checkpoint_dir,
                args.neo_checkpoint_dir,
                expected_steps=args.expected_steps,
                rtol=args.tensor_rtol,
                atol=args.tensor_atol,
                checkpoint=True,
            )

    print(
        json.dumps(
            {
                "status": "PASS",
                "steps": args.expected_steps,
                "semantic_metrics_compared": metric_count,
                "rollout_records_compared": rollout_count,
                "snapshot_values_compared": snapshot_tensor_count,
                "checkpoint_values_compared": checkpoint_tensor_count,
                "metric_rtol": args.metric_rtol,
                "metric_atol": args.metric_atol,
                "tensor_rtol": args.tensor_rtol,
                "tensor_atol": args.tensor_atol,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
