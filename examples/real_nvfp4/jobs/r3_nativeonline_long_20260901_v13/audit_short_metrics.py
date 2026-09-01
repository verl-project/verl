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
"""Gate the long chain on the 20-step short arm's *numbers*, not just its markers.

Job 2694835 (v11) passed every structural gate - 32/32 rollout servers, R3 replay,
native reload attestation, a 399 GiB full-Adam checkpoint, Slurm exit 0 - while its
rollout emitted near-uniform garbage.  Structural gates cannot see that; these
thresholds can.  Reference band comes from the only verified-healthy run,
v8 job 2694027.

Read-only.  Writes the pass file only when every threshold holds.
"""

import argparse
import json
import sys

import numpy as np
import wandb

# v8 j2694027, 20 steps: kl 0.0053-0.0082, ESS 0.9858-0.9912, entropy 0.327-0.893,
# clip_ratio 0-0.0059, grad_norm 0.092-1.215, length 933-1168.
THRESHOLDS = {
    "rollout_corr/kl": ("max", 0.02),
    "rollout_corr/rollout_is_eff_sample_size": ("min", 0.97),
    "actor/entropy": ("max", 1.5),
    "response_length/clip_ratio": ("max", 0.10),
    "actor/grad_norm": ("min_gt", 0.0),
}
LENGTH_KEY = "response_length/mean"
MIN_LENGTH_SLOPE = -5.0
MIN_STEPS = 20


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="W&B run name/id of the 20-step short arm")
    parser.add_argument("--entity", default="shawnzzz")
    parser.add_argument("--project", default="DAPO-NVFP4-QAT")
    parser.add_argument("--out", required=True, help="path of the pass file to write")
    args = parser.parse_args()

    run = wandb.Api().run(f"{args.entity}/{args.project}/{args.run}")
    keys = [*THRESHOLDS, LENGTH_KEY]
    history = run.history(keys=keys, samples=500, pandas=True)

    failures = []
    report = {}
    if len(history) < MIN_STEPS:
        failures.append(f"only {len(history)} logged steps, expected >= {MIN_STEPS}")

    for key, (rule, bound) in THRESHOLDS.items():
        if key not in history:
            failures.append(f"{key}: missing from history")
            continue
        values = history[key].to_numpy(dtype=float)
        observed = float(values.max()) if rule == "max" else float(values.min())
        report[key] = observed
        ok = observed <= bound if rule == "max" else (observed > bound if rule == "min_gt" else observed >= bound)
        if not ok:
            failures.append(f"{key}: {rule} is {observed:.6g}, bound {bound:g}")

    values = history[LENGTH_KEY].to_numpy(dtype=float)
    slope = float(np.polyfit(np.arange(len(values)), values, 1)[0])
    report[LENGTH_KEY] = {"slope_per_step": slope, "first": float(values[0]), "last": float(values[-1])}
    if slope < MIN_LENGTH_SLOPE:
        failures.append(f"{LENGTH_KEY}: slope {slope:+.3f} tok/step below {MIN_LENGTH_SLOPE:+.3f}")

    print(json.dumps({"run": args.run, "state": run.state, "metrics": report}, indent=2))
    if failures:
        for failure in failures:
            print(f"REAL_NVFP4_LONG_REFUSED: {failure}", file=sys.stderr)
        return 2

    with open(args.out, "w") as handle:
        json.dump({"run": args.run, "metrics": report, "thresholds": {k: v for k, v in THRESHOLDS.items()}}, handle, indent=2, default=str)
        handle.write("\n")
    print(f"REAL_NVFP4_SHORT_METRICS_PASS run={args.run} out={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
