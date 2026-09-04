#!/usr/bin/env python3
"""
eval_by_dataset.py — Per-dataset breakdown of verl eval results.

Reads eval JSON files (which contain results[].extra_info.dataset_name if you
used the updated searchr1_dataset_tool.py), or falls back to heuristic
classification from question text.

Usage:
  # Analyse a single file
  python evaluate/eval_by_dataset.py --files eval_checker_no_triage.json

  # Compare multiple files
  python eval_by_dataset.py \
      --files \
        eval_clean.json \
        eval_search_only_no_triage.json \
        eval_checker_no_triage.json \
        eval_checker_triage_after_fix.json \
        eval_no_triage_explicitcheck_after_fix_v2.json \
        eval_triage_explicitcheck_after_fix_v2.json \
      --output per_dataset_results.csv

  # If eval files live in a directory
  python eval_by_dataset.py --dir ./eval_results/ --output per_dataset_results.csv
"""

import argparse
import json
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional


# =============================================================================
# Dataset classifier (fallback when extra_info.dataset_name not present)
# =============================================================================

def classify_by_question(q: str) -> str:
    """
    Heuristic fallback.  Accuracy ~95% on the current 87-sample test set.
    Replace with extra_info.dataset_name once you regenerate parquet.
    """
    q_lower = q.lower().strip()
    # CSIRO MedRedQA: Reddit AskDocs personal narratives
    if re.search(
        r"\b(i\'m|im |i am |i have |i\'ve|ive |hey |hi,|hi i|update:|about me:|"
        r"patient:|age:\s*\d|sex:\s*[mf])\b",
        q_lower,
    ):
        return "csiro"
    if re.search(r"\b\d+[mf]\b", q_lower):     return "csiro"
    if "r/askdocs" in q_lower:                  return "csiro"
    if len(q.split()) > 50:                     return "csiro"
    # PubMedQA: yes/no/maybe from abstracts
    if re.match(r"^(does|do|is there|are there|can|was|were)\b", q_lower) and len(q.split()) < 25:
        return "pubmedqa"
    # Default: LiveQA / MedQuAD
    return "liveqa_medquad"


# =============================================================================
# Load one eval JSON file
# =============================================================================

def load_eval_file(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def get_dataset_name(result: Dict[str, Any]) -> str:
    """
    Return dataset_name from extra_info if present (updated parquet),
    else fall back to question heuristic.
    """
    ei = result.get("extra_info") or {}
    if isinstance(ei, dict):
        dn = ei.get("dataset_name", "")
        if dn and dn != "unknown":
            return dn
    return classify_by_question(result.get("question", ""))


# =============================================================================
# Aggregate metrics per config × dataset
# =============================================================================

def aggregate(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Returns {dataset_name -> {metric -> value}}"""
    groups: Dict[str, List] = defaultdict(list)
    for r in results:
        groups[get_dataset_name(r)].append(r)

    out = {}
    for ds, rlist in sorted(groups.items()):
        n = len(rlist)
        def mean(key):
            vals = [r.get(key, 0) or 0 for r in rlist]
            return sum(vals) / n if n else 0.0

        # handle num_explicit_checks too
        avg_check = mean("num_checks") + mean("num_explicit_checks") + mean("num_auto_checks")

        out[ds] = {
            "n":              n,
            "f1_mean":        round(mean("f1"), 4),
            "em_mean":        round(mean("em"), 4),
            "fuzzy_acc":      round(sum(1 for r in rlist if r.get("fuzzy_correct")) / n, 4),
            "avg_search":     round(mean("num_searches"), 3),
            "avg_check":      round(avg_check, 3),
            "avg_tools":      round(mean("num_tools"), 3),
            "avg_turns":      round(mean("num_turns"), 3),
            "ans_tag_rate":   round(sum(1 for r in rlist if r.get("has_answer_tag")) / n, 4),
        }
    return out


# =============================================================================
# Pretty print
# =============================================================================

def print_table(label: str, agg: Dict[str, Dict[str, Any]]) -> None:
    print(f"\n{'─'*90}")
    print(f"  {label}")
    print(f"{'─'*90}")
    print(f"  {'Dataset':<25} {'N':>4} {'F1':>7} {'FuzzyAcc':>9} "
          f"{'AvgSearch':>10} {'AvgCheck':>9} {'AnsTag%':>8}")
    print(f"  {'─'*80}")
    for ds, m in agg.items():
        print(f"  {ds:<25} {m['n']:>4} {m['f1_mean']:>7.4f} {m['fuzzy_acc']:>9.4f} "
              f"{m['avg_search']:>10.3f} {m['avg_check']:>9.3f} {m['ans_tag_rate']:>8.1%}")


# =============================================================================
# CSV export
# =============================================================================

def export_csv(rows: List[Dict[str, Any]], path: str) -> None:
    import csv
    if not rows: return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved: {path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Per-dataset eval breakdown")
    parser.add_argument("--files", nargs="+", default=[],
                        help="One or more eval JSON files to analyse")
    parser.add_argument("--dir",   default=None,
                        help="Directory of eval JSON files (alternative to --files)")
    parser.add_argument("--output", default=None,
                        help="Optional: save results as CSV")
    args = parser.parse_args()

    paths: List[str] = list(args.files)
    if args.dir:
        paths += sorted(
            os.path.join(args.dir, f)
            for f in os.listdir(args.dir)
            if f.endswith(".json")
        )
    if not paths:
        parser.print_help()
        return

    all_rows: List[Dict[str, Any]] = []

    for path in paths:
        if not os.path.isfile(path):
            print(f"[skip] not found: {path}")
            continue
        d       = load_eval_file(path)
        results = d.get("results", [])
        if not results:
            print(f"[skip] no results in: {path}")
            continue

        # Config label from prometheus served_model_name or filename
        cfg_str = json.dumps(d.get("config", {}))
        m = re.search(r"served_model_name[^:]+:\s*\"([^\"]+)\"", cfg_str)
        label = m.group(1) if m else os.path.splitext(os.path.basename(path))[0]
        # Shorten label
        label = re.sub(r"^merged_qwen2\.5_7b_combined_", "", label)
        label = re.sub(r"_step_\d+$", "", label)

        agg = aggregate(results)
        print_table(label, agg)

        # Check whether dataset_name came from extra_info or heuristic
        has_field = any(
            isinstance(r.get("extra_info"), dict) and r["extra_info"].get("dataset_name")
            for r in results[:5]
        )
        if not has_field:
            print("  [note] dataset_name from heuristic (regenerate parquet for exact values)")

        # Overall metrics for reference
        n  = len(results)
        f1 = sum(r.get("f1",0) for r in results) / n
        print(f"\n  Overall: n={n}  F1={f1:.4f}")

        # Collect CSV rows
        for ds, m in agg.items():
            all_rows.append({"config": label, "dataset": ds, **m})

    if args.output and all_rows:
        export_csv(all_rows, args.output)

    # Cross-file F1 gap summary
    if len(paths) > 1:
        print(f"\n{'='*90}")
        print("  F1 gap summary: LiveQA/MedQuAD − CSIRO  (positive = factual questions easier)")
        print(f"  {'Config':<45} {'Gap (pp)':>10}")
        print(f"  {'─'*58}")
        by_config: Dict[str, Dict] = defaultdict(dict)
        for row in all_rows:
            by_config[row["config"]][row["dataset"]] = row["f1_mean"]
        for cfg, ds_map in by_config.items():
            csiro = ds_map.get("csiro", None)
            lq    = ds_map.get("liveqa_medquad", None)
            if csiro is not None and lq is not None:
                gap = (lq - csiro) * 100
                print(f"  {cfg:<45} {gap:>+9.1f}pp")


if __name__ == "__main__":
    main()
