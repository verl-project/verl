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

r"""Validate NeoProto Runtime correctness and performance matrices.

Use one entry point for baseline Ray, Runtime Ray, and Runtime Monarch:

    bash tests/special_e2e/ppo_trainer/run_runtime_matrix_1x8.sh --mode correctness
    bash tests/special_e2e/ppo_trainer/run_runtime_matrix_1x8.sh --mode performance

BASELINE_REPO/BASELINE_COMMIT select the comparison checkout; MODEL_PATH,
TRAIN_FILES, and VAL_FILES select inputs. --mode all runs both modes. --dry-run
prints the same cases and effective rollout settings without starting workers.

Correctness uses two seeded steps, priority scheduling, and checkpoint/rollout
comparisons. It sets max_num_seqs=1 and vLLM optimization level 0 to work around
batch-dependent RMSNorm (https://github.com/vllm-project/vllm/issues/48271), so
its timings do not represent normal performance. It does not disable engine
multiprocessing or override async scheduling and FlashInfer autotuning. Full
determinism in the rollout server already enables batch invariance.
ROLLOUT_ENFORCE_EAGER=True explicitly selects eager execution for both sides
when required. Performance keeps the default engine optimizations.

Metric tolerances default to rtol=1e-5/atol=1e-6 and tensor tolerances to zero.
Different routing or engine configurations can change sampled rollouts; report
those differences instead of weakening comparisons or requiring a specific fix
commit as the baseline. Performance uses two reversed-order repetitions, drops
two warmup steps, and retains the configured throughput threshold.

Both modes use NeoProto only. RUNTIME_MATRIX_TORCHSTORE_STRATEGY selects host or
local_rank volume placement. TORCHSTORE_LOCAL_CACHE_BYTES overrides the Monarch
cache budget when supplied; the backend recipe otherwise determines it.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

PATHS = (
    "baseline-ray-neoproto",
    "feature-ray-neoproto",
    "feature-monarch-neoproto",
)
COMPARISONS = (
    ("baseline-ray-neoproto", "feature-ray-neoproto"),
    ("baseline-ray-neoproto", "feature-monarch-neoproto"),
)
CORRECTNESS_COMPARISONS = COMPARISONS


def _load_index(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _path_key(row: dict[str, str]) -> str:
    return f"{row['variant']}-{row['backend']}-{row['data_plane']}"


def _load_equivalence_checker(repo: Path) -> ModuleType:
    path = repo / "tests/special_e2e/ppo_trainer/check_neoproto_equivalence.py"
    spec = importlib.util.spec_from_file_location("neoproto_equivalence", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load checker from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _check_checkpoint(root: Path, step: int) -> None:
    checkpoint = root / "checkpoint" / f"global_step_{step}"
    files = [str(path.relative_to(checkpoint)) for path in checkpoint.rglob("*.pt")]
    required = ("actor/model_", "actor/optim_", "critic/model_", "critic/optim_")
    missing = [fragment for fragment in required if not any(fragment in name for name in files)]
    if missing:
        raise AssertionError(f"incomplete checkpoint under {checkpoint}: missing={missing}")


def check_correctness(args: argparse.Namespace) -> None:
    checker = _load_equivalence_checker(args.repo)
    index = _load_index(args.index)
    correctness_rows = [row for row in index if row["mode"] == "correctness"]
    if len(correctness_rows) != len(PATHS) or any(row["repetition"] != "1" for row in correctness_rows):
        raise AssertionError("correctness requires exactly one run for each path")
    rows = {_path_key(row): args.root / row["run_name"] for row in correctness_rows}
    if set(rows) != set(PATHS):
        raise AssertionError(f"correctness paths are {sorted(rows)}")

    parsed = {}
    for path, root in rows.items():
        text, metrics = checker._parse_metrics(root / "training.log", args.steps)
        parsed[path] = (text, metrics)
        _check_checkpoint(root, args.steps)

    for path, (_, metrics) in parsed.items():
        for step, values in metrics.items():
            if values.get("timing_s/dataplane/materialize_calls", 0) <= 0:
                raise AssertionError(f"{path} step {step} did not materialize NeoProto refs")
            if "timing_s/dataplane/prefetch_gen" not in values:
                raise AssertionError(f"{path} step {step} is missing the NeoProto prefetch timing")

    results = {f"{left}_vs_{right}": {"status": "PASS"} for left, right in CORRECTNESS_COMPARISONS}
    _compare_strict_runs(checker, args, rows, parsed, results)

    summary = {
        "status": "PASS",
        "runs_per_path": 1,
        "metric_rtol": args.metric_rtol,
        "metric_atol": args.metric_atol,
        "tensor_rtol": args.tensor_rtol,
        "tensor_atol": args.tensor_atol,
        "comparisons": results,
    }
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))


def _compare_strict_runs(
    checker: ModuleType,
    args: argparse.Namespace,
    rows: dict[str, Path],
    parsed: dict[str, tuple[str, dict[int, dict[str, float]]]],
    results: dict[str, dict[str, Any]],
) -> None:
    """Compare every path through the current metrics, rollout and checkpoint outputs."""
    for left, right in CORRECTNESS_COMPARISONS:
        left_metrics = parsed[left][1]
        right_metrics = parsed[right][1]
        left_rollouts = checker._read_jsonl_dir(rows[left] / "rollouts", args.steps)
        right_rollouts = checker._read_jsonl_dir(rows[right] / "rollouts", args.steps)
        steps = {}
        for step in range(1, args.steps + 1):
            steps[str(step)] = {
                "status": "PASS",
                "semantic_metrics_compared": checker._compare_metrics(
                    {step: left_metrics[step]},
                    {step: right_metrics[step]},
                    rtol=args.metric_rtol,
                    atol=args.metric_atol,
                ),
                "rollout_records_compared": checker._compare_loaded_rollouts(
                    {step: left_rollouts[step]}, {step: right_rollouts[step]}
                ),
                "checkpoint": "not saved at this step" if step < args.steps else "pending",
            }
        count = checker._compare_torch_directories(
            rows[left] / "checkpoint" / f"global_step_{args.steps}",
            rows[right] / "checkpoint" / f"global_step_{args.steps}",
            expected_steps=args.steps,
            rtol=args.tensor_rtol,
            atol=args.tensor_atol,
            checkpoint=True,
        )
        steps[str(args.steps)]["checkpoint"] = {"status": "PASS", "values_compared": count}
        results[f"{left}_vs_{right}"]["steps"] = steps


def _load_metrics(path: Path) -> dict[int, dict[str, float]]:
    rows = {}
    for line in path.read_text().splitlines():
        data = json.loads(line).get("data", {})
        if "training/global_step" in data:
            rows[int(data["training/global_step"])] = data
    return rows


def _aggregate(rows: dict[int, dict[str, float]], steps: range, gpus: int) -> dict[str, float | int]:
    selected = [rows[step] for step in steps]
    tokens = sum(int(row["perf/total_num_tokens"]) for row in selected)
    seconds = sum(float(row["timing_s/step"]) for row in selected)
    return {
        "tokens": tokens,
        "seconds": seconds,
        "throughput_tok_s_gpu": tokens / seconds / gpus,
    }


def _throughput_delta(right: dict[str, float], left: dict[str, float]) -> float:
    left_value = left["throughput_tok_s_gpu"]
    right_value = right["throughput_tok_s_gpu"]
    return (right_value - left_value) / max(abs(left_value), abs(right_value))


def check_performance(args: argparse.Namespace) -> None:
    index = _load_index(args.index)
    repetitions = []
    for repetition in range(1, args.repetitions + 1):
        pair_rows = {
            _path_key(row): row
            for row in index
            if row["mode"] == "performance" and row["repetition"] == str(repetition)
        }
        if set(pair_rows) != set(PATHS):
            raise AssertionError(f"repetition {repetition} paths are {sorted(pair_rows)}")
        tokens_by_path = {}
        results = {}
        for path in PATHS:
            rows = _load_metrics(args.root / pair_rows[path]["run_name"] / "metrics.jsonl")
            expected = list(range(1, args.steps + 1))
            if sorted(rows) != expected:
                raise AssertionError(f"{path} repetition {repetition} steps are {sorted(rows)}")
            measured_steps = range(args.warmup_steps + 1, args.steps + 1)
            tokens_by_path[path] = [int(rows[step]["perf/total_num_tokens"]) for step in measured_steps]
            results[path] = _aggregate(rows, measured_steps, args.gpus)
        reference = tokens_by_path[PATHS[0]]
        if any(tokens != reference for tokens in tokens_by_path.values()):
            raise AssertionError(f"repetition {repetition} token mismatch: {tokens_by_path}")
        repetitions.append({"repetition": repetition, "tokens_per_path": sum(reference), "paths": results})

    means = {}
    for path in PATHS:
        values = [repetition["paths"][path]["throughput_tok_s_gpu"] for repetition in repetitions]
        means[path] = {"throughput_tok_s_gpu": sum(values) / len(values)}

    comparisons = {f"{left}_vs_{right}": _throughput_delta(means[right], means[left]) for left, right in COMPARISONS}
    failures = {name: delta for name, delta in comparisons.items() if delta < -args.max_delta}
    summary = {
        "status": "PASS" if not failures else "FAIL",
        "accepted_relative_delta": args.max_delta,
        "repetitions": repetitions,
        "mean": means,
        "throughput_delta": comparisons,
    }
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))
    if failures:
        raise AssertionError(f"mean throughput regressions exceed the limit: {failures}")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    correctness = subparsers.add_parser("correctness")
    correctness.add_argument("--repo", required=True, type=Path)
    correctness.add_argument("--root", required=True, type=Path)
    correctness.add_argument("--index", required=True, type=Path)
    correctness.add_argument("--output", required=True, type=Path)
    correctness.add_argument("--steps", type=int, default=2)
    correctness.add_argument("--metric-rtol", type=float, default=1e-5)
    correctness.add_argument("--metric-atol", type=float, default=1e-6)
    correctness.add_argument("--tensor-rtol", type=float, default=0.0)
    correctness.add_argument("--tensor-atol", type=float, default=0.0)
    correctness.set_defaults(func=check_correctness)

    performance = subparsers.add_parser("performance")
    performance.add_argument("--root", required=True, type=Path)
    performance.add_argument("--index", required=True, type=Path)
    performance.add_argument("--output", required=True, type=Path)
    performance.add_argument("--repetitions", type=int, default=2)
    performance.add_argument("--steps", type=int, default=10)
    performance.add_argument("--warmup-steps", type=int, default=2)
    performance.add_argument("--gpus", type=int, default=8)
    performance.add_argument("--max-delta", type=float, default=0.03)
    performance.set_defaults(func=check_performance)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
