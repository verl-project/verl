"""Preserve sd-search's inputs and split/order while adding AgentLoop metadata."""

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq

from .prepare_data import prepare


def fingerprint(path):
    """Hash original payload fields, including exact prompt bytes and row order."""
    digest, count = hashlib.sha256(), 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=1024):
        for row in batch.to_pylist():
            info = row["extra_info"]
            payload = [
                row["data_source"],
                row["prompt"],
                row["reward_model"],
                row["ability"],
                info["split"],
                info["index"],
            ]
            digest.update(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode())
            digest.update(b"\n")
            count += 1
    return {"rows": count, "payload_sha256": digest.hexdigest()}


def prepare_all(sd_root, destination, report_dir):
    report_dir.mkdir(parents=True, exist_ok=True)
    reports = {}
    for source in sorted((sd_root / "data/processed").glob("*.parquet")):
        # User explicitly deferred reward alignment: preserve EM for every split.
        metric = "em"
        target = destination / source.name
        if not target.exists():
            prepare(source, target, preserve_prompt=True, reward_metric=metric)
        original, adapted = fingerprint(source), fingerprint(target)
        if original != adapted:
            raise ValueError(f"Source payload or sample order changed: {source.name}")
        for batch in pq.ParquetFile(target).iter_batches(batch_size=1024, columns=["extra_info"]):
            if any(row["extra_info"].get("reward_metric") != metric for row in batch.to_pylist()):
                raise ValueError(f"Incorrect reward routing: {target}")
        reports[source.name] = {
            "source": str(source.resolve()),
            "destination": str(target.resolve()),
            "reward_metric": metric,
            "exact_payload_and_order_equal": True,
            **original,
        }
        print(source.name, original["rows"], metric, "EXACT_PAYLOAD_ORDER_OK", flush=True)
    if not {"train.parquet", "monitor.parquet", "eval_all.parquet"} <= reports.keys():
        raise ValueError("Required sd-search splits are missing")
    (report_dir / "datasets.json").write_text(json.dumps(reports, indent=2) + "\n")
    return reports


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser()
    parser.add_argument("--sd-root", type=Path, default=root.parent / "sd-search")
    parser.add_argument("--destination", type=Path, default=root / "data/search_r1_sd_search")
    parser.add_argument("--report-dir", type=Path, default=root / "artifacts/sd-search-alignment")
    args = parser.parse_args()
    prepare_all(args.sd_root, args.destination, args.report_dir)
