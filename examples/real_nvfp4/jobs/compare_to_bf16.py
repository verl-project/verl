#!/usr/bin/env python3
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

"""Compare a real-NVFP4 run against the BF16 reference at matched global steps.

These curves move a lot between step 20 and step 265, so a comparison is only
meaningful step-for-step.  See ``BF16_REFERENCE_0603.md`` for the reference and
for why a low log-prob difference alone is not evidence that an arm is better.

Read-only.
"""

import argparse
import json
import pathlib

import numpy as np
import wandb

METRICS = [
    "response_length/mean",
    "critic/score/mean",
    "actor/entropy",
    "rollout_corr/kl",
    "rollout_corr/rollout_is_eff_sample_size",
]
DEFAULT_REFERENCE = "/lustre/fsw/general_sa/shuazhang/tmp_inspect_20260901/bf16_ref_0603.json"
BF16_RUN_IDS = [
    "7a6qxjqp", "6bgjqwzb", "7xommzdu", "7fs8133e", "ilz3hbo1", "oe8mnow8",
    "lyt568dw", "izol4n5m", "jpjh7d07", "yv8l3cka", "ipm3jj1o",
]


def build_reference(entity: str, project: str, path: pathlib.Path) -> dict:
    """Stitch the BF16 segments by cumulative row index and cache the result."""
    api = wandb.Api()
    data = {key: [] for key in METRICS}
    for run_id in BF16_RUN_IDS:
        history = api.run(f"{entity}/{project}/{run_id}").history(keys=METRICS, samples=5000, pandas=True)
        if not len(history):
            continue
        history = history.sort_values("_step")
        for key in METRICS:
            data[key].extend(history[key].astype(float).tolist())
    data["steps"] = list(range(1, len(data[METRICS[0]]) + 1))
    path.write_text(json.dumps(data))
    return data


def slope(x: np.ndarray, y: np.ndarray) -> float:
    mask = ~np.isnan(y)
    return float(np.polyfit(x[mask], y[mask], 1)[0]) if mask.sum() > 2 else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="W&B run name/id of the real-NVFP4 arm")
    parser.add_argument("--entity", default="shawnzzz")
    parser.add_argument("--project", default="DAPO-NVFP4-QAT")
    parser.add_argument("--reference", default=DEFAULT_REFERENCE)
    parser.add_argument("--rebuild-reference", action="store_true")
    args = parser.parse_args()

    reference_path = pathlib.Path(args.reference)
    if args.rebuild_reference or not reference_path.exists():
        reference = build_reference(args.entity, args.project, reference_path)
    else:
        reference = json.loads(reference_path.read_text())

    history = wandb.Api().run(f"{args.entity}/{args.project}/{args.run}").history(
        keys=METRICS, samples=5000, pandas=True
    )
    if not len(history):
        print("run has no logged training steps yet")
        return 1
    history = history.sort_values("_step")
    n = len(history)
    steps = np.arange(1, n + 1)
    print(f"{args.run}: {n} training steps\nBF16 reference: {len(reference['steps'])} steps\n")

    header = f"{'metric':<42} {'W4A4 last':>11} {'BF16 @same':>11} {'W4A4 slope':>12} {'BF16 slope':>12}"
    print(header)
    print("-" * len(header))
    for key in METRICS:
        w4 = history[key].to_numpy(dtype=float)
        bf = np.array(reference[key][:n], dtype=float)
        if len(bf) == 0:
            continue
        bf_steps = steps[: len(bf)]
        print(
            f"{key:<42} {w4[-1]:>11.4f} {bf[-1]:>11.4f} "
            f"{slope(steps, w4):>+12.4f} {slope(bf_steps, bf):>+12.4f}"
        )

    length = history["response_length/mean"].to_numpy(dtype=float)
    length_slope = slope(steps, length)
    bf_length = np.array(reference["response_length/mean"][:n], dtype=float)
    print(
        f"\nresponse length is the primary signal: W4A4 {length_slope:+.2f} tok/step "
        f"vs BF16 {slope(steps[: len(bf_length)], bf_length):+.2f} tok/step over the same {n} steps."
    )
    if length_slope < 0:
        print("NEGATIVE length slope - this is the failure signature of a wrong quantization scope or refit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
