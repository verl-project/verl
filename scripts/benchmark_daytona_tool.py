# Copyright 2025 Individual Contributor: Muhammad Hashmi
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
"""Benchmark the Daytona sandbox tool: setup cost and execution latency."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import statistics
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from verl.tools.schemas import (
    OpenAIFunctionParametersSchema,
    OpenAIFunctionPropertySchema,
    OpenAIFunctionSchema,
    OpenAIFunctionToolSchema,
)

DEFAULT_CONCURRENCY = [1, 4, 8]
DEFAULT_WARMUPS = 3
DEFAULT_ITERATIONS = 20


def build_code_interpreter_schema() -> OpenAIFunctionToolSchema:
    """Return the code interpreter schema used by the benchmarked tool."""
    return OpenAIFunctionToolSchema(
        type="function",
        function=OpenAIFunctionSchema(
            name="code_interpreter",
            description="A tool for executing Python code.",
            parameters=OpenAIFunctionParametersSchema(
                type="object",
                properties={
                    "code": OpenAIFunctionPropertySchema(
                        type="string",
                        description="The Python code to execute.",
                    )
                },
                required=["code"],
            ),
        ),
    )


@dataclass(frozen=True)
class Scenario:
    """Describe one benchmark scenario."""

    name: str
    label: str
    code: str
    expected_substring: str | None
    expects_error: bool = False


SCENARIOS = [
    Scenario(
        name="simple_stdout",
        label="Executing print(2+2) — minimal round-trip latency",
        code="print(2 + 2)",
        expected_substring="4",
    ),
    Scenario(
        name="cpu_bound_stdout",
        label="Executing sum(i*i for 20k iterations) — CPU-bound compute",
        code=("total = 0\nfor i in range(20000):\n    total += i * i\nprint(total)\n"),
        expected_substring="2666466670000",
    ),
    Scenario(
        name="runtime_error",
        label="Executing raise ValueError — error propagation",
        code="raise ValueError('boom from benchmark')",
        expected_substring="ValueError",
        expects_error=True,
    ),
]


def percentile(values: list[float], pct: float) -> float:
    """Return a simple percentile for a non-empty sorted sample."""
    if not values:
        raise ValueError("percentile() requires at least one value")
    ordered = sorted(values)
    index = round((len(ordered) - 1) * pct)
    return ordered[index]


def build_daytona_config(args: argparse.Namespace) -> dict[str, Any]:
    """Build the Daytona tool configuration for benchmarking."""
    config = {
        "type": "native",
        "rate_limit": max(args.concurrency),
        "enable_global_rate_limit": True,
        "create_timeout": args.create_timeout,
        "default_timeout": args.default_timeout,
        "delete_timeout": args.delete_timeout,
        "auto_stop_interval": args.auto_stop_interval,
        "auto_delete_interval": args.auto_delete_interval,
        "name_prefix": "verl-daytona-bench",
        "language": "python",
    }

    for key, value in {
        "snapshot": args.daytona_snapshot,
        "api_url": args.daytona_api_url,
        "target": args.daytona_target,
        "organization_id": args.daytona_organization_id,
    }.items():
        if value is not None:
            config[key] = value

    return config


def ensure_output_dir(output_root: Path) -> Path:
    """Create a timestamped benchmark output directory."""
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = output_root / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def check_daytona_credentials() -> None:
    """Fail fast if Daytona credentials are missing."""
    if not os.environ.get("DAYTONA_API_KEY") and not os.environ.get("DAYTONA_JWT_TOKEN"):
        raise SystemExit("DAYTONA_API_KEY (or DAYTONA_JWT_TOKEN) is not set. Export it before running the benchmark.")


def make_tool(args: argparse.Namespace):
    """Instantiate the Daytona tool backend."""
    from verl.tools.daytona_sandbox_tool import DaytonaSandboxTool

    return DaytonaSandboxTool(build_daytona_config(args), build_code_interpreter_schema())


# -- Phase 1: Sandbox setup --------------------------------------------------


async def _create_one(tool) -> tuple[str, float]:
    """Create a single sandbox and return (instance_id, elapsed_seconds)."""
    t0 = perf_counter()
    instance_id, _ = await tool.create()
    return instance_id, perf_counter() - t0


async def measure_setup(tool, count: int) -> dict[str, Any]:
    """Create `count` sandboxes in parallel, measuring each creation time.

    Returns a dict with per-sandbox timings, total wall time, and the list
    of instance_ids for use in subsequent phases.

    On partial failure, cleans up any successfully created sandboxes before
    re-raising.
    """
    total_start = perf_counter()

    results = await asyncio.gather(
        *[_create_one(tool) for _ in range(count)],
        return_exceptions=True,
    )

    total_wall = perf_counter() - total_start

    # Separate successes from failures.
    succeeded = [(iid, t) for r in results if not isinstance(r, BaseException) for iid, t in [r]]
    failures = [r for r in results if isinstance(r, BaseException)]

    if failures:
        # Clean up whatever was created before propagating the error.
        created_ids = [iid for iid, _ in succeeded]
        if created_ids:
            print(f"\n  Setup failed — cleaning up {len(created_ids)} sandboxes that were created...")
            await _force_cleanup(tool, created_ids)
        raise failures[0]

    instance_ids = [iid for iid, _ in succeeded]
    create_times = [t for _, t in succeeded]

    return {
        "sandbox_count": count,
        "create_times_s": create_times,
        "p50_create_s": percentile(create_times, 0.50),
        "p95_create_s": percentile(create_times, 0.95),
        "max_create_s": max(create_times),
        "mean_create_s": statistics.fmean(create_times),
        "total_wall_s": total_wall,
        "instance_ids": instance_ids,
    }


async def _force_cleanup(tool, instance_ids: list[str]) -> None:
    """Best-effort cleanup of sandboxes. Logs but does not raise on individual failures."""
    results = await asyncio.gather(
        *[_release_one(tool, iid) for iid in instance_ids],
        return_exceptions=True,
    )
    failed = sum(1 for r in results if r.get("error"))
    if failed:
        print(f"  Warning: {failed}/{len(instance_ids)} sandboxes failed to release during cleanup")
    else:
        print(f"  Cleaned up {len(instance_ids)} sandboxes")


# -- Phase 2: Execution latency (sandboxes already running) ------------------


async def run_single_execution(tool, instance_id: str, scenario: Scenario, timeout: int) -> dict[str, Any]:
    """Execute one code snippet on an existing sandbox and measure latency."""
    started_at = perf_counter()

    try:
        response, _, metrics = await tool.execute(instance_id, {"code": scenario.code, "timeout": timeout})
    except Exception as exc:
        ended_at = perf_counter()
        return {
            "latency_s": ended_at - started_at,
            "success": False,
            "error": str(exc),
            "response_text": "",
            "metrics": {},
        }

    ended_at = perf_counter()
    metrics = metrics or {}
    response_text = response.text or ""
    has_error_signal = "had_error" in metrics
    had_error = bool(metrics.get("had_error"))

    if scenario.expects_error:
        success = scenario.expected_substring in response_text
        if has_error_signal:
            success = success and had_error
    else:
        success = scenario.expected_substring in response_text
        if has_error_signal:
            success = success and not had_error

    return {
        "latency_s": ended_at - started_at,
        "success": success,
        "error": None,
        "response_text": response_text,
        "metrics": metrics,
    }


async def measure_execution(
    tool,
    instance_ids: list[str],
    scenario: Scenario,
    concurrency: int,
    warmups: int,
    iterations: int,
    timeout: int,
) -> dict[str, Any]:
    """Measure execution latency on pre-created sandboxes.

    Distributes work round-robin across `concurrency` sandboxes from the pool.
    """
    pool = instance_ids[:concurrency]
    measured_calls = []
    total_measured_time = 0.0

    for phase_name, batch_count in (("warmup", warmups), ("measured", iterations)):
        for _ in range(batch_count):
            batch_start = perf_counter()

            # Round-robin: each concurrent task gets its own sandbox from the pool.
            tasks = [run_single_execution(tool, pool[i % len(pool)], scenario, timeout) for i in range(concurrency)]
            results = await asyncio.gather(*tasks)

            batch_duration = perf_counter() - batch_start

            if phase_name == "measured":
                total_measured_time += batch_duration
                measured_calls.extend(results)

    latencies = [c["latency_s"] for c in measured_calls]
    successes = sum(1 for c in measured_calls if c["success"])

    return {
        "scenario": scenario.name,
        "concurrency": concurrency,
        "measured_call_count": len(measured_calls),
        "success_count": successes,
        "failure_count": len(measured_calls) - successes,
        "p50_latency_s": percentile(latencies, 0.50),
        "p95_latency_s": percentile(latencies, 0.95),
        "max_latency_s": max(latencies),
        "mean_latency_s": statistics.fmean(latencies),
        "throughput_calls_per_s": len(measured_calls) / total_measured_time if total_measured_time else 0.0,
    }


# -- Phase 3: Teardown -------------------------------------------------------


async def _release_one(tool, instance_id: str) -> dict[str, Any]:
    """Release a single sandbox and record the outcome."""
    t0 = perf_counter()
    try:
        await tool.release(instance_id)
    except Exception as exc:
        return {
            "instance_id": instance_id,
            "elapsed_s": perf_counter() - t0,
            "error": str(exc),
        }

    return {
        "instance_id": instance_id,
        "elapsed_s": perf_counter() - t0,
        "error": None,
    }


async def measure_teardown(tool, instance_ids: list[str]) -> dict[str, Any]:
    """Release all sandboxes in parallel and measure total teardown time."""
    total_start = perf_counter()

    release_results = await asyncio.gather(*[_release_one(tool, iid) for iid in instance_ids])

    total_wall = perf_counter() - total_start
    failures = [result for result in release_results if result["error"] is not None]
    if failures:
        failure_summary = ", ".join(f"{item['instance_id']}: {item['error']}" for item in failures[:3])
        raise RuntimeError(f"Failed to release {len(failures)} sandboxes: {failure_summary}")

    release_times = [result["elapsed_s"] for result in release_results]

    return {
        "sandbox_count": len(instance_ids),
        "release_times_s": release_times,
        "mean_release_s": statistics.fmean(release_times) if release_times else 0.0,
        "total_wall_s": total_wall,
    }


# -- Reporting ----------------------------------------------------------------


def build_terminal_summary(summary: dict[str, Any]) -> str:
    """Render a fixed-width aligned terminal summary."""
    lines = []

    headers = ["Scenario", "Conc", "p50 (s)", "p95 (s)", "Thru (c/s)", "OK", "Fail"]
    widths = [16, 5, 8, 8, 10, 6, 4]

    ok_rows = [r for r in summary["results"] if r.get("status") == "ok"]
    if not ok_rows:
        return ""

    def fmt_row(vals: list[str]) -> str:
        return "  ".join(v.rjust(w) for v, w in zip(vals, widths, strict=False))

    lines.append("")
    lines.append(fmt_row(headers))
    lines.append("  ".join("-" * w for w in widths))

    for row in ok_rows:
        lines.append(
            fmt_row(
                [
                    row["scenario"],
                    str(row["concurrency"]),
                    f"{row['p50_latency_s']:.4f}",
                    f"{row['p95_latency_s']:.4f}",
                    f"{row['throughput_calls_per_s']:.2f}",
                    str(row["success_count"]),
                    str(row["failure_count"]),
                ]
            )
        )

    return "\n".join(lines) + "\n"


def write_json(path: Path, payload: Any) -> None:
    """Write JSON with stable indentation."""
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def write_csv(path: Path, result_rows: list[dict[str, Any]]) -> None:
    """Write a flat CSV of benchmark results."""
    import csv

    fieldnames = [
        "scenario",
        "concurrency",
        "p50_s",
        "p95_s",
        "max_s",
        "mean_s",
        "throughput_calls_per_s",
        "success_count",
        "failure_count",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in result_rows:
            if row.get("status") != "ok":
                continue
            writer.writerow(
                {
                    "scenario": row["scenario"],
                    "concurrency": row["concurrency"],
                    "p50_s": f"{row['p50_latency_s']:.4f}",
                    "p95_s": f"{row['p95_latency_s']:.4f}",
                    "max_s": f"{row['max_latency_s']:.4f}",
                    "mean_s": f"{row['mean_latency_s']:.4f}",
                    "throughput_calls_per_s": f"{row['throughput_calls_per_s']:.2f}",
                    "success_count": row["success_count"],
                    "failure_count": row["failure_count"],
                }
            )


# -- Main orchestrator --------------------------------------------------------


async def run_benchmarks(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Run all benchmarks: setup, execution, teardown."""
    max_concurrency = max(args.concurrency)
    result_rows: list[dict[str, Any]] = []
    setup_result: dict[str, Any] | None = None

    tool = make_tool(args)

    # Phase 1: Create sandboxes up front.
    instance_ids: list[str] = []
    print(f"\n{'=' * 60}")
    print(f"  SETUP — Creating {max_concurrency} sandboxes")
    print(f"{'=' * 60}")
    try:
        setup = await measure_setup(tool, max_concurrency)
        instance_ids = setup.pop("instance_ids")
        print(
            f"  {max_concurrency} sandboxes created in {setup['total_wall_s']:.2f}s (p50={setup['p50_create_s']:.3f}s)"
        )

        # Phase 2: Measure execution latency at each concurrency level.
        for scenario in SCENARIOS:
            print(f"\n{'=' * 60}")
            print(f"  {scenario.label}")
            print(f"{'=' * 60}")
            for concurrency in args.concurrency:
                print(f"  concurrency={concurrency}...", end=" ", flush=True)
                result = await measure_execution(
                    tool,
                    instance_ids,
                    scenario,
                    concurrency,
                    args.warmups,
                    args.iterations,
                    args.default_timeout,
                )
                result["status"] = "ok"
                result_rows.append(result)
                ok = result["success_count"]
                fail = result["failure_count"]
                print(
                    f"p50={result['p50_latency_s']:.3f}s  "
                    f"throughput={result['throughput_calls_per_s']:.1f}/s  {ok}/{ok + fail} ok"
                )
    finally:
        # Always clean up whatever sandboxes were created, even on failure.
        if instance_ids:
            print(f"\n{'=' * 60}")
            print(f"  TEARDOWN — Releasing {len(instance_ids)} sandboxes")
            print(f"{'=' * 60}")
            teardown = await measure_teardown(tool, instance_ids)
            print(f"  {len(instance_ids)} sandboxes released in {teardown['total_wall_s']:.2f}s")
            setup_result = {"setup": setup, "teardown": teardown}
        await tool.close()

    return result_rows, setup_result


def parse_args() -> argparse.Namespace:
    """Parse benchmark CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="outputs/daytona")
    parser.add_argument(
        "--concurrency",
        type=int,
        nargs="+",
        default=DEFAULT_CONCURRENCY,
        help="Concurrent tool calls per batch. Sandboxes are pre-created for the max level.",
    )
    parser.add_argument("--warmups", type=int, default=DEFAULT_WARMUPS)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--default-timeout", type=int, default=30)
    parser.add_argument("--create-timeout", type=int, default=60)
    parser.add_argument("--delete-timeout", type=int, default=60)
    parser.add_argument("--auto-stop-interval", type=int, default=15)
    parser.add_argument("--auto-delete-interval", type=int, default=30)
    parser.add_argument("--daytona-api-url", default=None)
    parser.add_argument("--daytona-target", default=None)
    parser.add_argument("--daytona-organization-id", default=None)
    parser.add_argument("--daytona-snapshot", default=None)
    return parser.parse_args()


def main() -> int:
    """Run the benchmark and save the artifact bundle locally."""
    args = parse_args()
    check_daytona_credentials()
    output_dir = ensure_output_dir(Path(args.output_root))

    result_rows, setup_result = asyncio.run(run_benchmarks(args))

    summary = {
        "created_at": datetime.now(UTC).isoformat(),
        "host": platform.platform(),
        "python_version": platform.python_version(),
        "results": result_rows,
        "setup_result": setup_result,
    }

    write_json(output_dir / "summary.json", summary)
    write_csv(output_dir / "results.csv", result_rows)

    print(f"\nSaved benchmark artifacts to {output_dir}")
    print(build_terminal_summary(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
