"""Replay execution journals into fit_step windows and a cumulative summary."""
import argparse
import hashlib
import json
import math
from pathlib import Path

FLOP_FIELDS = ("actor_flops", "reference_flops", "prefill_flops", "decode_flops")
FIELDS = FLOP_FIELDS + ("actor_tokens", "reference_tokens", "prefill_tokens", "decode_tokens")


def evaluate(directory, run_finished=True):
    root = Path(directory)
    manifest = json.loads((root / "manifest.json").read_text())
    # Older manifests used trainer_npus; accept it for offline replay.
    train_npus = manifest.get("train_npus", manifest.get("trainer_npus"))
    rollout_npus = manifest.get("rollout_npus")
    for name, count in (("train_npus", train_npus), ("rollout_npus", rollout_npus)):
        if type(count) is not int or count <= 0:
            raise ValueError(f"{name} must be a positive integer")
    total_npus = train_npus + rollout_npus
    if not isinstance(manifest["warmup_steps"], int) or manifest["warmup_steps"] < 0:
        raise ValueError("warmup_steps must be a nonnegative integer")
    records, hashes = [], {}
    for path in sorted(root.glob("events-*.jsonl")):
        raw = path.read_bytes()
        hashes[path.name] = hashlib.sha256(raw).hexdigest()
        records.extend(json.loads(line) for line in raw.splitlines() if line.strip())
    errors = []
    seen = set()
    for r in records:
        key = (r["producer"], r["sequence"])
        if key in seen:
            errors.append("duplicate journal record")
        seen.add(key)
    producers = {r["producer"]: r for r in records if r["kind"] == "producer"}
    for producer in producers:
        sequence = sorted(r["sequence"] for r in records if r["producer"] == producer)
        if sequence != list(range(1, len(sequence) + 1)):
            errors.append(f"missing journal records: {producer}")
    windows = sorted((r for r in records if r["kind"] == "window"), key=lambda r: r["start"])
    if len({r["step"] for r in windows}) != len(windows):
        errors.append("duplicate fit_step (restart or overlapping run)")
    for index, window in enumerate(windows):
        if (not math.isfinite(window["end"] - window["start"])
                or window["end"] <= window["start"]
                or (index and window["start"] < windows[index - 1]["end"])):
            errors.append("invalid or overlapping windows")
    rows = windows[manifest["warmup_steps"]:]
    if not rows:
        errors.append("no post-warmup fit_step windows")
    executions = [r for r in records if r["kind"] == "execution"]
    ids = [r["event_id"] for r in executions]
    if len(set(ids)) != len(ids):
        errors.append("duplicate execution")
    complete = set(ids)
    for r in records:
        if r["kind"] == "begin" and r["event_id"] not in complete:
            if any(r["timestamp"] < w["end"] for w in rows):
                errors.append(f"unfinished execution: {r['event_id']}")
    rollout_producers = {r["producer"] for r in executions if r["role"] == "rollout"}
    actor_producers = {r["producer"] for r in executions if r["role"] == "actor"}
    reference_producers = {r["producer"] for r in executions if r["role"] == "reference"}
    for producer in rollout_producers | actor_producers | reference_producers:
        watermark = max((r["timestamp"] for r in records if r["producer"] == producer
                         and r["kind"] in ("watermark", "execution")), default=0)
        if rows and watermark < rows[-1]["end"]:
            errors.append(f"producer has not reached final window watermark: {producer}")
    if len(rollout_producers) != manifest["expected_rollout_engines"]:
        errors.append(f"expected {manifest['expected_rollout_engines']} rollout engines, observed {len(rollout_producers)}")
    if len(actor_producers) != 1:
        errors.append("expected one Actor global-rank-zero producer")
    if manifest.get("reference_enabled") and len(reference_producers) != 1:
        errors.append("expected one Reference global-rank-zero producer")
    for r in executions:
        if not math.isfinite(r["end"] - r["start"]) or r["end"] <= r["start"]:
            raise ValueError("invalid execution interval")
        for key in FIELDS:
            if not math.isfinite(r.get(key, 0)) or r.get(key, 0) < 0:
                raise ValueError("invalid FLOPs/token value")
    output = []
    for w in rows:
        totals = dict.fromkeys(FIELDS, 0.0)
        boundary_flops = 0.0
        for r in executions:
            overlap = max(0, min(w["end"], r["end"]) - max(w["start"], r["start"]))
            fraction = overlap / (r["end"] - r["start"])
            for key in FIELDS:
                totals[key] += fraction * r.get(key, 0)
            if 0 < fraction < 1:
                boundary_flops += fraction * sum(r.get(k, 0) for k in FLOP_FIELDS)
        seconds = w["end"] - w["start"]
        output.append(dict(step=w["step"], start=w["start"], end=w["end"], wall_seconds=seconds,
                           boundary_allocated_flops=boundary_flops, **totals))
    totals = {key: sum(r[key] for r in output) for key in FIELDS}
    wall = sum(r["wall_seconds"] for r in output)
    peak = manifest["device_peak_flops"]
    if not math.isfinite(peak) or peak <= 0:
        raise ValueError("requires finite positive peak")
    flops = sum(totals[k] for k in FLOP_FIELDS)
    uncertainty = max((r["uncertainty_s"] for r in producers.values()), default=0)
    clock_checks = [r for r in records if r["kind"] == "clock_check"]
    drift = max((abs(r["offset_change_s"]) + r["uncertainty_s"] for r in clock_checks), default=0)
    if max(uncertainty, drift) > manifest.get("clock_tolerance_s", 0.1):
        errors.append("clock calibration uncertainty/drift exceeds tolerance")
    # Long-run oscillator drift is unmeasured; never present this as verified hardware MFU.
    verified_peak = bool(manifest.get("peak_source"))
    if not run_finished:
        errors.append("training did not finish successfully")
    summary = dict(estimated_mfu_percent=100 * flops / (wall * total_npus * peak) if wall and not errors else None,
                   diagnostic_estimated_mfu_percent=100 * flops / (wall * total_npus * peak) if wall else None,
                   train_npus=train_npus, rollout_npus=rollout_npus, total_npus=total_npus,
                   status="complete_estimate" if not errors and verified_peak else "incomplete" if errors else "assumed_peak",
                   errors=errors, fit_steps=len(output), wall_seconds=wall, estimated_flops=flops,
                   formula_version=manifest.get("formula_version"),
                   reference_included=bool(reference_producers),
                   clock_initial_uncertainty_s=uncertainty, clock_drift_verified=False,
                   clock_observed_drift_bound_s=drift,
                   boundary_allocated_flops=sum(r["boundary_allocated_flops"] for r in output),
                   device_names=sorted({r["device_name"] for r in executions if "device_name" in r}),
                   event_sha256=hashes, **totals)
    (root / "windows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in output))
    (root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("--failed", action="store_true")
    args = parser.parse_args()
    print(json.dumps(evaluate(args.directory, not args.failed), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
