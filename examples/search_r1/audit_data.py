"""Compare local questions and answers to pinned author parquet, offline."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from .prepare_data import adapt_row


def summarize(paths, normalize_question_suffix=False):
    records, questions, counts = Counter(), set(), Counter()
    ordered = hashlib.sha256()
    for path in paths:
        parquet = pq.ParquetFile(path)
        columns = [
            name
            for name in ["data_source", "question", "prompt", "reward_model", "golden_answers", "extra_info"]
            if name in parquet.schema_arrow.names
        ]
        for batch in parquet.iter_batches(batch_size=1024, columns=columns):
            for i, raw in enumerate(batch.to_pylist()):
                row = adapt_row(raw, i)
                question = row["prompt"][1]["content"]
                if normalize_question_suffix and not question.endswith("?"):
                    question += "?"
                answers = sorted(row["reward_model"]["ground_truth"]["target"])
                key = json.dumps([row["data_source"], question, answers], ensure_ascii=False)
                signature = hashlib.sha256(key.encode()).hexdigest()
                records[signature] += 1
                ordered.update(bytes.fromhex(signature))
                questions.add((row["data_source"], question))
                counts[row["data_source"]] += 1
    return {"records": records, "questions": questions, "counts": dict(counts), "ordered_sha256": ordered.hexdigest()}


def compare(official_paths, candidate_paths):
    result = {}
    for normalized in [False, True]:
        reference = summarize(official_paths, normalized)
        candidate = summarize(candidate_paths, normalized)
        missing = reference["records"] - candidate["records"]
        extra = candidate["records"] - reference["records"]
        result["question_suffix_normalized" if normalized else "exact_question"] = {
            "official_rows": sum(reference["counts"].values()),
            "candidate_rows": sum(candidate["counts"].values()),
            "official_counts": reference["counts"],
            "candidate_counts": candidate["counts"],
            "question_and_answer_multisets_equal": not missing and not extra,
            "missing_rows": sum(missing.values()),
            "extra_rows": sum(extra.values()),
            "order_equal": reference["ordered_sha256"] == candidate["ordered_sha256"],
            "official_ordered_sha256": reference["ordered_sha256"],
            "candidate_ordered_sha256": candidate["ordered_sha256"],
        }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--official", nargs="+", required=True)
    parser.add_argument("--candidate", nargs="+", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    report = compare(args.official, args.candidate)
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
