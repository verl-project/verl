"""Strict, stdlib-only validation of completed training-step metrics."""

import argparse
import json
import math
import re
from pathlib import Path

REQUIRED = (
    "actor/loss",
    "actor/pg_loss",
    "actor/grad_norm",
    "actor/entropy",
    "rollout_corr/kl",
    "rollout_corr/rollout_is_eff_sample_size",
    "rollout_corr/rollout_log_ppl",
    "training/rollout_probs_diff_mean",
    "training/rollout_probs_diff_max",
    "training/global_step",
    "train/actor_optimizer_steps",
)
NUMBER = re.compile(
    r"[+-]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?|nan|inf(?:inity)?)",
    re.IGNORECASE,
)
WRAPPER = re.compile(r"np\.(?:float(?:16|32|64|128)|int(?:8|16|32|64))\((.*)\)")
ANSI = re.compile(r"\x1b\[[0-9;]*m")
STEP = re.compile(r"(?:^|\s)step:(\d+)\s+-\s+")


def finite_scalar(raw):
    value = raw.strip()
    match = WRAPPER.fullmatch(value)
    if match:
        value = match[1].strip()
    if not NUMBER.fullmatch(value):
        raise ValueError("unsupported numeric encoding")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("non-finite metric")
    return result


def validate_lines(lines, first_step, last_step):
    if first_step < 1 or last_step < first_step:
        raise ValueError("invalid step interval")
    rows = {}
    for line in lines:
        line = ANSI.sub("", line).strip()
        match = STEP.search(line)
        if not match:
            continue
        # Evaluation-only rows are not optimizer updates. An incomplete actor
        # row is still parsed and must fail the required-key check below.
        if "actor/" not in line and "training/global_step:" not in line:
            continue
        step = int(match[1])
        if not first_step <= step <= last_step:
            raise ValueError(f"unexpected training step {step}")
        fields = {}
        for field in re.split(r"\s+-\s+", line[match.end() :]):
            key, separator, raw = field.partition(":")
            if not separator or key not in REQUIRED:
                continue
            if key in fields:
                raise ValueError(f"duplicate metric {key} at step {step}")
            try:
                fields[key] = finite_scalar(raw)
            except ValueError as error:
                raise ValueError(f"{key} at step {step}: {error}") from error
        missing = set(REQUIRED) - fields.keys()
        if missing:
            raise ValueError(f"missing metrics at step {step}: {sorted(missing)}")
        if fields["training/global_step"] != step:
            raise ValueError(f"inconsistent global step at step {step}")
        if fields["train/actor_optimizer_steps"] != 1:
            raise ValueError(f"expected one optimizer step at step {step}")
        if fields["rollout_corr/kl"] > 0.5:
            raise ValueError(f"desynchronized rollout at step {step}")
        if fields["rollout_corr/rollout_is_eff_sample_size"] < 0.8:
            raise ValueError(f"collapsed ESS at step {step}")
        if step in rows and rows[step] != fields:
            raise ValueError(f"conflicting duplicate step {step}")
        rows[step] = fields
    expected = list(range(first_step, last_step + 1))
    if sorted(rows) != expected:
        raise ValueError(
            f"missing training steps: expected {expected}, found {sorted(rows)}"
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--first-step", type=int, required=True)
    parser.add_argument("--last-step", type=int, required=True)
    args = parser.parse_args()
    with args.log.open() as stream:
        rows = validate_lines(stream, args.first_step, args.last_step)
    print(
        "STRICT_NUMERIC_HISTORY_PASS",
        json.dumps(
            {
                "log": str(args.log),
                "first_step": args.first_step,
                "last_step": args.last_step,
                "validated_steps": len(rows),
                "max_kl": max(row["rollout_corr/kl"] for row in rows.values()),
                "min_ess": min(
                    row["rollout_corr/rollout_is_eff_sample_size"]
                    for row in rows.values()
                ),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
