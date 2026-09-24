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
"""Offline ablation for weighted multi-output agent-loop trajectories.

Reads the per-step ``*.jsonl`` files that ``trainer.rollout_data_dir`` writes
(one row per stored training segment, ``uid = {prompt_uid}_{session}_{index}``,
``score`` broadcast to every segment of a logical trajectory) and recomputes,
from the SAME rollouts, the GRPO advantage each segment would receive under

* ``unweighted`` -- every stored row has weight 1.0 (pre-#7580 behaviour), and
* ``weighted``   -- every row of an N-segment trajectory has weight 1/N.

Because loss_weight only changes how sampled rollouts are weighted, not which
rollouts are sampled, the two objectives can be compared exactly: no second
training run, no GPU, no seed variance.

Reported:

1. Per-trajectory gradient influence as a function of N. Unweighted it grows
   linearly with N; weighted it is constant. That constant is the invariant.
2. The success rate the objective actually optimises against (row-weighted)
   versus the trajectory-level ground truth.
3. Kish effective sample size under 1/N (the intended cost).
4. Cosine similarity of the two per-row advantage vectors (is it a no-op?).
5. Per-step ``mean(1/N)`` over rows -- the factor by which the loss would shrink
   and drift step-to-step if the weights were NOT renormalised to mean 1.0.
   This is the quantity ``normalize_loss_weight_global`` removes.

Usage:
    python3 scripts/analyze_multi_output_loss_weight.py \
        --rollout-data-dir /path/to/rollout_data_dir [--steps 1-100] [--out DIR]

Only ``uid``, ``score`` and whether ``output`` is empty (padding) are used from
each row; prompt/response text is never decoded beyond that.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import statistics
from collections import defaultdict


def _parse_steps(spec: str | None) -> set[int] | None:
    if not spec:
        return None
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-")
            out.update(range(int(lo), int(hi) + 1))
        elif part:
            out.add(int(part))
    return out


def load_trajectories(rollout_dir: str, steps: set[int] | None) -> list[dict]:
    """One record per logical trajectory: step, GRPO group key, score, n_segments."""
    rows: list[dict] = []
    n_padding = 0
    for path in sorted(glob.glob(os.path.join(rollout_dir, "*.jsonl"))):
        m = re.fullmatch(r"(\d+)\.jsonl", os.path.basename(path))
        if not m:
            continue
        step = int(m.group(1))
        if steps is not None and step not in steps:
            continue
        per_traj: dict[tuple[str, str], list[float]] = defaultdict(list)
        with open(path) as handle:
            for line in handle:
                try:
                    rec = json.loads(line)
                    uid = rec["uid"]
                    score = float(rec["score"])
                except (ValueError, KeyError, TypeError):
                    continue
                # Synthetic padding rows (added to reach the required batch
                # multiple) have no response. The trainer zeroes their weight
                # via validate_loss_weights(valid_mask); mirror that here.
                if "output" in rec and not rec["output"]:
                    n_padding += 1
                    continue
                prompt_uid, session, _index = uid.rsplit("_", 2)
                # Length proxy for the partition-preserving weight. The dump stores
                # the decoded response text, not token ids, so use its character
                # count; only the RATIO between segments of one trajectory matters.
                length = len(rec["output"]) if isinstance(rec.get("output"), str) else 1
                per_traj[(prompt_uid, session)].append((score, max(length, 1)))
        for (prompt_uid, _session), items in per_traj.items():
            scores = [sc for sc, _ in items]
            lengths = [ln for _, ln in items]
            rows.append(
                {
                    "step": step,
                    "group": f"{step}:{prompt_uid}",
                    "score": scores[-1],
                    "n_segments": len(scores),
                    "lengths": lengths,
                    "score_consistent": len({round(s, 6) for s in scores}) == 1,
                }
            )
    if n_padding:
        print(f"skipped {n_padding} padding rows (empty response)")
    return rows


def grpo_advantages(scores: list[float], epsilon: float = 1e-6) -> list[float]:
    """Mirror ``compute_grpo_outcome_advantage`` (unbiased std, eps in denominator)."""
    if len(scores) == 1:
        return [0.0]
    mean = sum(scores) / len(scores)
    var = sum((s - mean) ** 2 for s in scores) / (len(scores) - 1)
    std = math.sqrt(var)
    return [(s - mean) / (std + epsilon) for s in scores]


def kish_ess(weights: list[float]) -> float:
    total = sum(weights)
    sq = sum(w * w for w in weights)
    return (total * total / sq) if sq > 0 else 0.0


def analyse(rows: list[dict], success_threshold: float) -> dict:
    n_traj = len(rows)
    n_rows = sum(r["n_segments"] for r in rows)

    by_n: dict[int, dict] = defaultdict(lambda: {"trajectories": 0, "rows": 0, "solved": 0})
    for r in rows:
        b = by_n[r["n_segments"]]
        b["trajectories"] += 1
        b["rows"] += r["n_segments"]
        b["solved"] += 1 if r["score"] > success_threshold else 0

    share = {}
    for n, b in sorted(by_n.items()):
        share[n] = {
            "trajectories": b["trajectories"],
            "unweighted_share": b["rows"] / n_rows,
            "weighted_share": b["trajectories"] / n_traj,
            "solve_rate": b["solved"] / b["trajectories"],
            "per_traj_unweighted": b["rows"] / n_rows / b["trajectories"],
            "per_traj_weighted": 1.0 / n_traj,
        }

    solved = [r for r in rows if r["score"] > success_threshold]
    failed = [r for r in rows if r["score"] <= success_threshold]
    solved_rows = sum(r["n_segments"] for r in solved)

    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r["group"]].append(r)

    adv_w: list[float] = []  # session-equal: w_j = 1/N
    adv_u: list[float] = []  # unweighted:    w_j = 1
    adv_p: list[float] = []  # partition-preserving: w_j = T_j / mean(T) (then 1/N-scaled so the
    #                          trajectory total matches session-equal; see docs/advance/agent_loop.rst)
    weights_w: list[float] = []
    length_ratio_max: list[float] = []
    for members in groups.values():
        advantages = grpo_advantages([m["score"] for m in members])
        for member, adv in zip(members, advantages, strict=True):
            n = member["n_segments"]
            lengths = member["lengths"]
            mean_len = sum(lengths) / n
            adv_w.extend([adv / n] * n)
            adv_u.extend([adv] * n)
            adv_p.extend([adv / n * (ln / mean_len) for ln in lengths])
            weights_w.extend([1.0 / n] * n)
            if n > 1:
                length_ratio_max.append(max(lengths) / max(min(lengths), 1))

    def _cos(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b, strict=True))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return dot / (na * nb) if na > 0 and nb > 0 else float("nan")

    cosine = _cos(adv_w, adv_u)
    cosine_session_vs_partition = _cos(adv_w, adv_p)
    cosine_partition_vs_unweighted = _cos(adv_p, adv_u)

    # Per-step mean(1/N) over rows == n_traj/n_rows for that step. Without
    # renormalisation this is the factor the whole loss is multiplied by.
    per_step: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    for r in rows:
        per_step[r["step"]][0] += 1
        per_step[r["step"]][1] += r["n_segments"]
    step_scale = {k: t / s for k, (t, s) in sorted(per_step.items()) if s}

    return {
        "n_trajectories": n_traj,
        "n_rows": n_rows,
        "n_steps": len(per_step),
        "n_groups": len(groups),
        "score_inconsistent_trajectories": sum(1 for r in rows if not r["score_consistent"]),
        "share_by_n": share,
        "solve_rate_trajectory": len(solved) / n_traj,
        "solve_rate_row": solved_rows / n_rows,
        "mean_segments_solved": statistics.mean(r["n_segments"] for r in solved) if solved else float("nan"),
        "mean_segments_failed": statistics.mean(r["n_segments"] for r in failed) if failed else float("nan"),
        "ess_weighted": kish_ess(weights_w),
        "ess_unweighted": float(n_rows),
        "advantage_cosine": cosine,
        "advantage_cosine_session_vs_partition": cosine_session_vs_partition,
        "advantage_cosine_partition_vs_unweighted": cosine_partition_vs_unweighted,
        "segment_length_ratio": {
            "median_max_over_min": statistics.median(length_ratio_max) if length_ratio_max else float("nan"),
            "p90_max_over_min": (
                sorted(length_ratio_max)[int(0.9 * (len(length_ratio_max) - 1))] if length_ratio_max else float("nan")
            ),
        },
        "loss_scale_if_unnormalised": {
            "overall": n_traj / n_rows,
            "per_step_min": min(step_scale.values()),
            "per_step_max": max(step_scale.values()),
            "per_step_median": statistics.median(step_scale.values()),
        },
    }


def render(stats: dict, source: str) -> str:
    L: list[str] = []
    add = L.append
    add("# Offline ablation: weighted (1/N) vs unweighted multi-output trajectories")
    add("")
    add(f"Source: `{source}`  ")
    add(
        f"**{stats['n_trajectories']} logical trajectories -> {stats['n_rows']} stored rows** "
        f"over {stats['n_steps']} steps / {stats['n_groups']} GRPO groups."
    )
    if stats["score_inconsistent_trajectories"]:
        add(
            f"WARNING: {stats['score_inconsistent_trajectories']} trajectories had "
            "non-identical scores across segments."
        )
    add("")
    add("Both columns are computed from the SAME rollouts; the comparison isolates the")
    add("objective exactly, with no sampling noise.")
    add("")
    add("## 1. Per-trajectory gradient influence vs segment count N")
    add("")
    add(
        "| N | trajectories | solve rate | unweighted share | weighted share | "
        "per-traj unweighted (x1e-4) | per-traj weighted (x1e-4) | ratio vs N=1 |"
    )
    add("|---|---|---|---|---|---|---|---|")
    base = stats["share_by_n"][min(stats["share_by_n"])]["per_traj_unweighted"]
    for n, row in stats["share_by_n"].items():
        ratio = row["per_traj_unweighted"] / base if base > 0 else float("nan")
        add(
            f"| {n} | {row['trajectories']} | {100 * row['solve_rate']:.1f}% | "
            f"{100 * row['unweighted_share']:.1f}% | {100 * row['weighted_share']:.1f}% | "
            f"{1e4 * row['per_traj_unweighted']:.2f} | {1e4 * row['per_traj_weighted']:.2f} | {ratio:.1f}x |"
        )
    add("")
    add("Unweighted, one trajectory's influence is proportional to N. Weighted, the")
    add("column is constant (= 1/n_trajectories): segment count no longer acts as an")
    add("implicit optimisation weight.")
    add("")
    add("## 2. Success rate the objective optimises against")
    add("")
    add("| view | success rate | mean N |")
    add("|---|---|---|")
    add(
        f"| by trajectory (ground truth) | **{100 * stats['solve_rate_trajectory']:.1f}%** | "
        f"solved {stats['mean_segments_solved']:.2f} / failed {stats['mean_segments_failed']:.2f} |"
    )
    add(f"| by stored row (unweighted objective) | **{100 * stats['solve_rate_row']:.1f}%** | |")
    gap = 100 * (stats["solve_rate_trajectory"] - stats["solve_rate_row"])
    add(f"| gap | **{gap:+.1f} pp** | |")
    add("")
    add("The sign and size of the gap depend on how episode length correlates with")
    add("success in this dataset/stage; 1/N restores the trajectory-level rate exactly.")
    add("")
    add("## 3. Effective sample size (Kish)")
    add("")
    add(f"| | ESS | of {stats['n_rows']} rows |")
    add("|---|---|---|")
    add(f"| unweighted | {stats['ess_unweighted']:.0f} | 100.0% |")
    add(f"| weighted 1/N | {stats['ess_weighted']:.0f} | {100 * stats['ess_weighted'] / stats['n_rows']:.1f}% |")
    add("")
    add("## 4. Are the two objectives different?")
    add("")
    add(f"Cosine similarity of per-row advantage vectors: **{stats['advantage_cosine']:.4f}** (1.0 would be a no-op).")
    add("")
    add("## 4b. Session-equal (1/N) vs partition-preserving (T_j / mean T)")
    add("")
    add("`1/N` gives every logical trajectory one vote; `T_j / mean(T)` reproduces the")
    add("unsplit trajectory's `seq-mean-token-mean` loss. They coincide only when the")
    add("segments of a trajectory have similar length. Length here is the character")
    add("count of the stored response text (the dump has no token ids); only the ratio")
    add("between segments of one trajectory enters the weight.")
    add("")
    lr = stats["segment_length_ratio"]
    add("| | value |")
    add("|---|---|")
    add(f"| cosine(session-equal, partition-preserving) | **{stats['advantage_cosine_session_vs_partition']:.4f}** |")
    add(f"| cosine(partition-preserving, unweighted) | {stats['advantage_cosine_partition_vs_unweighted']:.4f} |")
    add(f"| within-trajectory max/min segment length, median | {lr['median_max_over_min']:.2f}x |")
    add(f"| within-trajectory max/min segment length, p90 | {lr['p90_max_over_min']:.2f}x |")
    add("")
    add("## 5. Why the weights must be renormalised to mean 1.0")
    add("")
    s = stats["loss_scale_if_unnormalised"]
    add("Every `loss_agg_mode` divides by an UNWEIGHTED denominator, so raw 1/N weights")
    add("scale the whole loss by `mean(1/N)` over the rows of the batch:")
    add("")
    add("| | mean(1/N) over rows | implied lr multiplier |")
    add("|---|---|---|")
    add(f"| all steps pooled | {s['overall']:.3f} | x{s['overall']:.2f} |")
    add(f"| per-step min | {s['per_step_min']:.3f} | x{s['per_step_min']:.2f} |")
    add(f"| per-step median | {s['per_step_median']:.3f} | x{s['per_step_median']:.2f} |")
    add(f"| per-step max | {s['per_step_max']:.3f} | x{s['per_step_max']:.2f} |")
    add(f"| step-to-step drift | | **{s['per_step_max'] / s['per_step_min']:.2f}x** |")
    add("")
    add("`normalize_loss_weight_global` rescales the weights to mean 1.0 over the global")
    add("batch, preserving every ratio w_i/w_j while removing both the shrink and the drift.")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rollout-data-dir", required=True, help="trainer.rollout_data_dir")
    ap.add_argument("--steps", default=None, help="e.g. '1-100' or '10,20,30'; default all")
    ap.add_argument("--success-threshold", type=float, default=0.0, help="score > threshold counts as solved")
    ap.add_argument("--out", default=None, help="write report.md and stats.json here")
    args = ap.parse_args()

    rows = load_trajectories(args.rollout_data_dir, _parse_steps(args.steps))
    if not rows:
        raise SystemExit(f"no rollout rows found under {args.rollout_data_dir}")
    stats = analyse(rows, args.success_threshold)
    report = render(stats, args.rollout_data_dir)
    print(report)
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "report.md"), "w") as h:
            h.write(report)
        with open(os.path.join(args.out, "stats.json"), "w") as h:
            json.dump(stats, h, indent=2)
        print(f"written to {args.out}/report.md and stats.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
