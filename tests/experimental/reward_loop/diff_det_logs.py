#!/usr/bin/env python3
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
"""Compare [DET] checkpoint logs from two determinism runs to find the first divergence.

Usage:
  python tests/experimental/reward_loop/diff_det_logs.py <run1_det.log> <run2_det.log>

Each line in a det log looks like:
  [DET] step=1 generate_sequences t0=sum:1.2345678901e+02,shape:(8, 64)
  [DET] step=0 after_forward_backward rank=0 loss_sum=1.2345678901e-01

Alignment is by (label, rank) + occurrence index — the Nth time a given
(label, rank) checkpoint fires in run1 is compared to the Nth in run2. This is
robust to (a) different step-numbering schemes between the trainer layer
(TaskRunner actor, global_steps) and engine layer (worker actors, call counter),
and (b) interleaved ordering of rank-0/rank-1 lines.

The "first divergence" is reported by scanning occurrence #0 across all labels
in physical pipeline order (not alphabetical), so an earlier-firing checkpoint
is never misreported as diverging after a later one.
"""

import re
import sys
from collections import defaultdict

LINE_RE = re.compile(r"\[DET\](?:\s+step=(\d+))?\s+(\S+)(?:\s+rank=(\d+))?\s*(.*)")

# Physical pipeline order (earliest first). Used to find the true first divergence.
PIPELINE_ORDER = [
    "priority_batch",
    "route",
    "generate_sequences",
    "after_balance_batch",
    "reward",
    "token_level_rewards",
    "old_log_probs",
    "ref_log_prob",
    "values",
    "advantage",
    "update_critic",
    "input_ids",
    "raw_logits",
    "logits_pre_logp",
    "after_forward",
    "after_forward_backward",
    "pre_clip",
    "post_clip",
    "after_optimizer_step",
    "post_optimizer",
    "update_actor",
    "update_weights_done",
]


def parse(path):
    """Return {(label, rank): [rest, rest, ...]} in occurrence order."""
    by_key = defaultdict(list)
    with open(path) as f:
        for line in f:
            m = LINE_RE.search(line)
            if not m:
                continue
            _step, label, rank, rest = m.group(1), m.group(2), m.group(3), m.group(4)
            rank = int(rank) if rank is not None else None
            by_key[(label, rank)].append(rest)
    return by_key


def pipeline_idx(label):
    return PIPELINE_ORDER.index(label) if label in PIPELINE_ORDER else len(PIPELINE_ORDER)


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    run1, run2 = parse(sys.argv[1]), parse(sys.argv[2])
    all_keys = sorted(set(run1.keys()) | set(run2.keys()), key=lambda k: (pipeline_idx(k[0]), str(k[1])))

    first_div = None  # (label, rank, occ, r1, r2)

    print(f"{'label':<28} {'rank':<6} {'occ':<4} {'status'}")
    print("-" * 80)
    for label, rank in all_keys:
        seq1 = run1.get((label, rank), [])
        seq2 = run2.get((label, rank), [])
        rstr = str(rank) if rank is not None else "-"
        n = max(len(seq1), len(seq2))
        for occ in range(n):
            r1 = seq1[occ] if occ < len(seq1) else "<MISSING>"
            r2 = seq2[occ] if occ < len(seq2) else "<MISSING>"
            if r1 == r2:
                status = "ok"
            else:
                status = "*** DIFF ***"
                # First divergence in physical pipeline order: the first DIFF
                # encountered while scanning keys (pipeline-ordered) and occ
                # (0..n). Note occ!=0 DIFFs can still be the first divergence
                # when occ==0 was ok (e.g. determinism breaks after a few micro-batches).
                if first_div is None:
                    first_div = (label, rank, occ, r1, r2)
            print(f"{label:<28} {rstr:<6} {occ:<4} {status}")
            if status == "*** DIFF ***":
                print(f"        run1: {r1}")
                print(f"        run2: {r2}")

    print("-" * 80)
    if first_div is not None:
        label, rank, occ, r1, r2 = first_div
        rstr = f" rank={rank}" if rank is not None else ""
        print(
            f"\nFIRST DIVERGENCE: {label}{rstr} occurrence #{occ}\n"
            "Look at the stage BEFORE this checkpoint — the divergence originates there."
        )
        idx = pipeline_idx(label)
        if idx > 0:
            print(f"  => likely culprit is between '{PIPELINE_ORDER[idx - 1]}' and '{label}'")
        if label == "input_ids":
            print("  => input checkpoint (BEFORE model forward). If this DIFFs, the micro-batch")
            print("     composition differs (data loading / dynamic_bsz assignment) -> NOT a forward")
            print("     op issue. If input ok but raw_logits DIFFs -> model forward non-deterministic.")
        elif label == "raw_logits":
            print("  => raw logits checkpoint (model forward output, BEFORE any postprocessing).")
            print("     If this DIFFs -> MODEL FORWARD is non-deterministic (root cause in")
            print("     forward matmul/attention/autocast, NOT in logp computation).")
            print("     If ok but after_forward logp_sum DIFFs -> logp/postprocessing is non-deterministic.")
        elif label == "logits_pre_logp":
            print("  => logits checkpoint (model forward output, BEFORE logp computation).")
            print("     If this DIFFs -> model forward is non-deterministic (root cause in")
            print("     forward matmul/attention, not in logp computation).")
            print("     If this is ok but after_forward logp_sum DIFFs -> logp computation")
            print("     (logprobs_from_logits) is non-deterministic.")
        elif label == "after_forward":
            print("  => forward checkpoint (loss + log_probs). If this DIFFs, the inputs")
            print("     (responses from rollout) differ -> rollout is non-deterministic.")
            print("     If this is ok but pre_clip DIFFs -> non-deterministic backward op.")
        elif label in ("after_forward_backward", "pre_clip"):
            print("  => forward/backward checkpoint; if after_forward was ok but")
            print("     pre_clip (local grad, no all-reduce) DIFFs, the BACKWARD pass produced")
            print("     different local gradients -> non-deterministic backward op.")
        elif label in ("post_clip", "after_optimizer_step"):
            print("  => post-clip grad_norm DIFFs; if pre_clip was ok, the all-reduce inside")
            print("     clip_grad_norm_ is non-deterministic.")
    else:
        print("\nAll [DET] checkpoints aligned across both runs.")


if __name__ == "__main__":
    main()
