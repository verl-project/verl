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
"""Prepare real ALFWorld tasks and optionally freeze a measured long-tail workload."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

SYSTEM_PROMPT = (
    "Solve the household task in ALFWorld. Use the alfworld tool with exactly one available action per turn. "
    "Explore locations and inspect receptacles as needed. Do not invent objects or claim success yourself. "
    "The environment determines success. Keep reasoning brief and continue until the task is complete."
)


def episode_records(log_path):
    """Read structured episode records from Ray's prefixed console logs."""
    records = []
    decoder = json.JSONDecoder()
    for line in Path(log_path).read_text().splitlines():
        if "alfworld_episode " in line:
            record, _ = decoder.raw_decode(line.split("alfworld_episode ", 1)[1])
            records.append(record)
    return records


def group_durations(records, responses_per_group=2):
    """Exclude incomplete groups and deduplicate replayed log lines."""
    groups = defaultdict(dict)
    for record in records:
        groups[record["uid"]][record["session"]] = record
    completed = []
    for sessions in groups.values():
        if set(sessions) != set(range(responses_per_group)):
            continue
        values = list(sessions.values())
        if len({record["game_file"] for record in values}) != 1:
            raise ValueError("A pilot group contains different games")
        completed.append((values[0]["game_file"], max(record["duration_s"] for record in values)))
    if not completed:
        raise ValueError("No complete groups in pilot log; use rollout.val_kwargs.n=2")
    return completed


def tail_report(completed):
    """Report measured group latency; the threshold is a workload criterion, not a dataset claim."""
    durations = np.asarray([duration for _, duration in completed])
    p50, p90, p95, p99 = np.percentile(durations, [50, 90, 95, 99])
    return dict(
        groups=len(completed),
        p50_s=float(p50),
        p90_s=float(p90),
        p95_s=float(p95),
        p99_s=float(p99),
        max_s=float(durations.max()),
        p95_over_p50=float(p95 / max(p50, 1e-9)),
    )


def select_long_tail(rows, completed, count, seed, minimum_ratio=5.0):
    """Freeze an 80:20 fast/slow mixture using genuine measured training episodes."""
    by_game = defaultdict(list)
    available = {row["extra_info"]["game_file"]: row for row in rows}
    for game, duration in completed:
        if game in available:
            by_game[game].append(duration)
    medians = {game: float(np.median(values)) for game, values in by_game.items()}
    if len(medians) < 32:
        raise ValueError("Measure at least 32 distinct training games before selecting a tail subset")
    ordered = sorted(medians, key=medians.get)
    fast = ordered[: max(1, len(ordered) // 2)]
    slow = ordered[-max(1, len(ordered) // 10) :]
    rng = np.random.default_rng(seed)
    slow_count = max(1, round(count * 0.2))
    selected = list(rng.choice(fast, count - slow_count)) + list(rng.choice(slow, slow_count))
    rng.shuffle(selected)
    report = tail_report([(game, medians[game]) for game in selected])
    if report["p95_over_p50"] < minimum_ratio:
        raise ValueError(f"Measured mixture is not sufficiently long-tailed: {report}; collect a broader pilot")
    return [available[game] for game in selected], report


def collect_games(root, split):
    """Use the official solvability filter and supported ALFWorld task families."""
    rows = []
    for game in sorted((root / "json_2.1.1" / split).rglob("game.tw-pddl")):
        if "movable" in str(game) or "Sliced" in str(game):
            continue
        if not json.loads(game.read_text()).get("solvable", False):
            continue
        metadata = json.loads(game.with_name("traj_data.json").read_text())
        rows.append(
            dict(
                data_source="alfworld",
                agent_name="alfworld_agent",
                ability="agentic_household",
                prompt=[{"role": "system", "content": SYSTEM_PROMPT}],
                reward_model={"style": "rule", "ground_truth": "environment_won"},
                extra_info={
                    "game_file": str(game.relative_to(root)),
                    "task_type": metadata["task_type"],
                    "split": split,
                    "index": len(rows),
                },
            )
        )
    if not rows:
        raise ValueError(f"No solvable games under {root / 'json_2.1.1' / split}")
    return rows


def main():
    """Create parquet inputs and a source/selection manifest; never generate artificial delays."""
    from datasets import Dataset

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pilot-log", type=Path)
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--minimum-tail-ratio", type=float, default=5.0)
    args = parser.parse_args()
    if args.count < 32:
        parser.error("--count must be at least 32")
    rows = collect_games(args.data_root.resolve(), "train")
    manifest = dict(
        source="https://github.com/alfworld/alfworld",
        seed=args.seed,
        selection="uniform pilot",
        natural_tail_measured=False,
    )
    if args.pilot_log:
        records = episode_records(args.pilot_log)
        completed = group_durations(records)
        rows, selected_report = select_long_tail(rows, completed, args.count, args.seed, args.minimum_tail_ratio)
        manifest.update(
            selection="80 percent fast, 20 percent slow; sampled with replacement",
            natural_tail_measured=True,
            pilot=tail_report(completed),
            selected_pilot_estimate=selected_report,
            pilot_success_rate=float(np.mean([r["won"] for r in records])),
            pilot_cap_rate=float(np.mean([r["capped"] for r in records])),
        )
    else:
        np.random.default_rng(args.seed).shuffle(rows)
        rows = rows[: args.count]
    args.output.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(rows).to_parquet(str(args.output / "train.parquet"))
    # Evaluate the pilot on training tasks; held-out evaluation uses test.parquet.
    Dataset.from_list(rows).to_parquet(str(args.output / "pilot.parquet"))
    Dataset.from_list(collect_games(args.data_root.resolve(), "valid_unseen")[:64]).to_parquet(
        str(args.output / "test.parquet")
    )
    manifest["games"] = [row["extra_info"]["game_file"] for row in rows]
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in manifest.items() if key != "games"}, indent=2))


if __name__ == "__main__":
    main()
