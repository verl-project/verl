#!/usr/bin/env python3
#  python3 inference_outputs/analyze_generation_parquet.py \
# /ocean/projects/med230010p/yji3/BrowseCamp/verl/inference_outputs/checker_guarded__test__0324_133104.parquet
#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE)
ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)


def normalize_response_cell(cell: Any) -> list[str]:
    if isinstance(cell, list):
        return [str(x) for x in cell]
    if pd.isna(cell):
        return []
    return [str(cell)]


def extract_tool_names(text: str) -> list[str]:
    names = []
    for match in TOOL_CALL_RE.findall(text or ""):
        blob = match.strip()
        try:
            payload = json.loads(blob)
            name = payload.get("name")
            if isinstance(name, str):
                names.append(name)
        except Exception:
            # keep going; malformed tool calls still count elsewhere
            pass
    return names


def extract_answer(text: str) -> str:
    m = ANSWER_RE.search(text or "")
    return m.group(1).strip() if m else ""


def get_ground_truth(cell: Any) -> str:
    if isinstance(cell, dict):
        gt = cell.get("ground_truth")
        if isinstance(gt, list):
            return " || ".join(str(x) for x in gt)
        if gt is None:
            return ""
        return str(gt)
    if isinstance(cell, list):
        return " || ".join(str(x) for x in cell)
    if pd.isna(cell):
        return ""
    return str(cell)


def normalize_for_eval(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def exact_match_score(pred: str, gold: str) -> float:
    return float(normalize_for_eval(pred) == normalize_for_eval(gold))


def token_f1_score(pred: str, gold: str) -> float:
    pred_tokens = normalize_for_eval(pred).split()
    gold_tokens = normalize_for_eval(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = {}
    for tok in pred_tokens:
        common[tok] = common.get(tok, 0) + 1
    overlap = 0
    for tok in gold_tokens:
        if common.get(tok, 0) > 0:
            overlap += 1
            common[tok] -= 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def summarize_file(path: Path, sample_rows: int) -> dict[str, Any]:
    df = pd.read_parquet(path)

    if "responses" not in df.columns:
        raise ValueError(f"{path} has no 'responses' column. Columns: {list(df.columns)}")

    normalized = df["responses"].apply(normalize_response_cell)
    first_response = normalized.apply(lambda xs: xs[0] if xs else "")

    tool_call_counts = first_response.str.count(r"<tool_call>")
    answer_tag_counts = first_response.str.count(r"<answer>")
    extracted_answers = first_response.apply(extract_answer)
    tool_names = first_response.apply(extract_tool_names)

    search_calls = tool_names.apply(lambda xs: sum(x == "search" for x in xs))
    check_calls = tool_names.apply(lambda xs: sum(x == "check" for x in xs))

    gold_series = None
    if "reward_model" in df.columns:
        gold_series = df["reward_model"].apply(get_ground_truth)
    elif "gold_answer" in df.columns:
        gold_series = df["gold_answer"].apply(get_ground_truth)
    elif "answer" in df.columns:
        gold_series = df["answer"].apply(get_ground_truth)

    em = None
    f1 = None
    if gold_series is not None:
        em = [exact_match_score(p, g) for p, g in zip(extracted_answers, gold_series, strict=False)]
        f1 = [token_f1_score(p, g) for p, g in zip(extracted_answers, gold_series, strict=False)]

    summary = {
        "path": str(path),
        "rows": len(df),
        "columns": list(df.columns),
        "avg_num_candidates": normalized.apply(len).mean(),
        "empty_first_response_rate": float((first_response.str.strip() == "").mean()),
        "has_answer_tag_rate": float((answer_tag_counts > 0).mean()),
        "has_any_tool_call_rate": float((tool_call_counts > 0).mean()),
        "avg_tool_calls_first_response": float(tool_call_counts.mean()),
        "avg_search_calls_first_response": float(search_calls.mean()),
        "avg_check_calls_first_response": float(check_calls.mean()),
        "avg_first_response_chars": float(first_response.str.len().mean()),
        "avg_extracted_answer_chars": float(extracted_answers.str.len().mean()),
        "has_ground_truth": gold_series is not None,
    }
    if em is not None and f1 is not None:
        summary["exact_match"] = float(sum(em) / len(em)) if em else 0.0
        summary["token_f1"] = float(sum(f1) / len(f1)) if f1 else 0.0

    preview_cols = [c for c in ["data_source", "question", "prompt", "responses", "reward_model", "gold_answer", "answer"] if c in df.columns]
    preview = df[preview_cols].head(sample_rows).copy()
    if "responses" in preview.columns:
        preview["responses"] = preview["responses"].apply(
            lambda xs: normalize_response_cell(xs)[0][:1200] if normalize_response_cell(xs) else ""
        )
    if "reward_model" in preview.columns:
        preview["ground_truth"] = preview["reward_model"].apply(get_ground_truth)
    elif "gold_answer" in preview.columns:
        preview["ground_truth"] = preview["gold_answer"].apply(get_ground_truth)
    elif "answer" in preview.columns:
        preview["ground_truth"] = preview["answer"].apply(get_ground_truth)

    return {"summary": summary, "preview": preview}


def print_summary(result: dict[str, Any]) -> None:
    summary = result["summary"]
    print("=" * 80)
    print(summary["path"])
    print("=" * 80)
    for key, value in summary.items():
        if key in {"path", "columns"}:
            continue
        print(f"{key}: {value}")
    print(f"columns: {summary['columns']}")
    print("\nSample rows:")
    print(result["preview"].to_string(index=False))
    print()


def compare_results(results: list[dict[str, Any]]) -> None:
    rows = []
    for item in results:
        s = item["summary"]
        rows.append(
            {
                "file": Path(s["path"]).name,
                "rows": s["rows"],
                "empty_rate": s["empty_first_response_rate"],
                "answer_tag_rate": s["has_answer_tag_rate"],
                "tool_call_rate": s["has_any_tool_call_rate"],
                "avg_tools": s["avg_tool_calls_first_response"],
                "avg_search": s["avg_search_calls_first_response"],
                "avg_check": s["avg_check_calls_first_response"],
                "avg_resp_chars": s["avg_first_response_chars"],
                "avg_answer_chars": s["avg_extracted_answer_chars"],
            }
        )
    cmp_df = pd.DataFrame(rows)
    print("=" * 80)
    print("Comparison")
    print("=" * 80)
    print(cmp_df.to_string(index=False))
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze verl generation parquet outputs")
    parser.add_argument("paths", nargs="+", help="One or more parquet files")
    parser.add_argument("--sample-rows", type=int, default=3)
    args = parser.parse_args()

    results = [summarize_file(Path(p), args.sample_rows) for p in args.paths]
    for result in results:
        print_summary(result)
    if len(results) > 1:
        compare_results(results)


if __name__ == "__main__":
    main()
