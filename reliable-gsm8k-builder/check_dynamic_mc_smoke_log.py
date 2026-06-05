#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


CHECKS: tuple[tuple[str, str], ...] = (
    ("dataset hook ran", r"\[GSM8KDynamicMCDataset\].*hook=\d+"),
    ("stage 1 rollouts seen", r"stage1_seen=([1-9]\d*)"),
    ("accepted correct candidates", r"accepted_correct_total=([1-9]\d*)"),
    ("accepted incorrect candidates", r"accepted_incorrect_total=([1-9]\d*)"),
    ("stage 2 rows queued", r"queued_stage2=([1-9]\d*)"),
    ("epoch-end promotion ran", r"epoch_end=\d+ pending_stage2=([1-9]\d*)"),
    ("stage 2 rows inserted", r"inserted_stage2=([1-9]\d*)"),
    ("train dataloader rebuilt", r"Rebuilt train dataloader after dynamic dataset insertion"),
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a VERL-native dynamic MC smoke-run log.")
    parser.add_argument("log_file", help="Path to the log written by bash/20260604_gsm8k_dynamic_mc_verl_native.sh")
    args = parser.parse_args()

    log_path = Path(args.log_file).expanduser().resolve()
    if not log_path.exists():
        print(f"[dynamic-mc-smoke] missing log file: {log_path}", file=sys.stderr)
        return 2

    text = log_path.read_text(encoding="utf-8", errors="replace")
    missing = [name for name, pattern in CHECKS if re.search(pattern, text) is None]

    if missing:
        print(f"[dynamic-mc-smoke] FAIL log={log_path}")
        for name in missing:
            print(f"missing: {name}")
        return 1

    print(f"[dynamic-mc-smoke] PASS log={log_path}")
    print("verified: Stage 1 hook, verified candidate buffering, Stage 2 queue/insert, dataloader rebuild")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
