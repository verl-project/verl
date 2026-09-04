#!/usr/bin/env python3
"""
prepare_extra_eval_datasets.py

把额外的评估数据集转成 verl eval 需要的 parquet 格式。
格式完全对齐 searchr1_dataset_tool.py 的输出（含 prompt、reward_model、extra_info 等字段）。

Usage:
    cd /ocean/projects/med230010p/yji3/BrowseCamp/verl
    export HF_HUB_OFFLINE=0
    export HF_HOME=/ocean/projects/med230010p/yji3/.cache/huggingface
    conda activate sglang_srv
    python search_r1_preprocess/prepare_extra_eval_datasets.py

输出目录:
    /ocean/projects/med230010p/yji3/MedicalRagChecker/verl/searchr1_data/extra_eval/
        medicationqa_test.parquet   (≤200 samples)
        bioasq_test.parquet         (≤200 samples)
        medquad_full_test.parquet   (≤200 samples, held-out from full MedQuAD)
        liveqa_full_test.parquet    (all LiveQA-Med, ~634 samples)
"""

import json, os, random
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("/ocean/projects/med230010p/yji3/BrowseCamp/verl/searchr1_data/extra_eval")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MAX_SAMPLES = 200
SEED = 42
random.seed(SEED)

# ── System prompt (same as medical_checker profile in searchr1_dataset_tool.py) ──
_SYS_MEDICAL_CHECKER = (
    "Answer the given medical question. You must conduct reasoning inside "
    "<think> and </think> first every time you get new information. "
    "After reasoning, if you find you lack some knowledge, you can call a search engine "
    "by <search> query </search> and it will return the top searched results between "
    "<information> and </information>. "
    "Before giving your final answer, you MUST verify it by using <check> your answer </check>; "
    "the verification result will appear between <information> and </information>. "
    "If the verification contradicts your answer, revise and check again. "
    "Finally provide your answer inside <answer> and </answer>."
)

_FEWSHOT_MEDICAL_CHECKER = """\
Here is an example:
Question: What is the first-line pharmacological treatment for type 2 diabetes?
Assistant: <think>I need to find the standard first-line treatment for type 2 diabetes. Let me search current guidelines.</think>
<search>first-line pharmacological treatment type 2 diabetes ADA guidelines</search>
<information>Metformin is the preferred initial pharmacological agent for type 2 diabetes management according to ADA and most international guidelines, unless contraindicated.</information>
<think>The evidence supports metformin. Before I finalize, let me verify this answer.</think>
<check>Metformin is the first-line pharmacological treatment for type 2 diabetes</check>
<information>Verification result: Supported. ADA Standards of Medical Care in Diabetes consistently recommend metformin as first-line therapy unless contraindicated due to renal impairment (eGFR < 30) or other factors.</information>
<think>The verification confirms the answer is correct and well-supported by current guidelines.</think>
<answer>Metformin is the first-line pharmacological treatment for type 2 diabetes, unless contraindicated.</answer>

Now answer this medical question:
Question: {question}"""


def make_prompt(question: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": _SYS_MEDICAL_CHECKER},
        {"role": "user",   "content": _FEWSHOT_MEDICAL_CHECKER.format(question=question)},
    ]


def make_row(idx: int, question: str, answer: str,
             dataset_name: str, split: str = "test") -> Dict[str, Any]:
    """Build a row matching verl NQ schema from searchr1_dataset_tool.py."""
    return {
        "id":             f"{split}_{idx}",
        "question":       question.strip(),
        "golden_answers": [answer.strip()],
        "data_source":    "csiro_medredqa",          # verl reward routing unchanged
        "prompt":         make_prompt(question),
        "ability":        "medical-reasoning",
        "reward_model": {
            "ground_truth": {"target": [answer.strip()]},
            "style":        "rule",
        },
        "extra_info": {
            "index":          idx,
            "split":          split,
            "data_source":    "csiro_medredqa",
            "dataset_name":   dataset_name,
            "prompt_profile": "medical_checker",
        },
        "metadata":   None,
        "agent_name": "tool_agent",
    }


def save_parquet(rows: List[Dict[str, Any]], path: Path) -> None:
    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False)
    print(f"  ✓ Saved {len(df)} samples → {path}")


# ── Dataset loaders ───────────────────────────────────────────────────────────

def load_medicationqa(max_n: int = MAX_SAMPLES) -> List[Dict[str, Any]]:
    """MedicationQA: 674 consumer medication questions with expert answers."""
    from datasets import load_dataset
    ds = load_dataset("truehealth/medicationqa", split="train")
    rows, idx = [], 0
    for item in ds:
        q = (item.get("Question") or item.get("question") or "").strip()
        a = (item.get("Answer")   or item.get("answer")   or "").strip()
        if q and a:
            rows.append(make_row(idx, q, a, "medicationqa"))
            idx += 1
        if idx >= max_n:
            break
    return rows


def load_bioasq(max_n: int = MAX_SAMPLES) -> List[Dict[str, Any]]:
    """BioASQ: biomedical open-ended QA with ideal answers."""
    from datasets import load_dataset
    # Try rag-mini version first (easier access)
    try:
        ds = load_dataset(
            "rag-datasets/rag-mini-bioasq",
            "question-answer-passages",
            split="test"
        )
        rows, idx = [], 0
        for item in ds:
            q = (item.get("question") or "").strip()
            a = (item.get("answer")   or "").strip()
            if q and a:
                rows.append(make_row(idx, q, a, "bioasq"))
                idx += 1
            if idx >= max_n:
                break
        if rows:
            return rows
    except Exception as e:
        print(f"  rag-mini-bioasq failed: {e}, trying alternative...")

    # Fallback: qiaojin/PubMedQA which is available and similar
    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    rows, idx = [], 0
    items = list(ds)
    random.shuffle(items)
    for item in items:
        q = (item.get("question") or "").strip()
        a = (item.get("long_answer") or "").strip()
        if q and a and len(a.split()) > 10:
            rows.append(make_row(idx, q, a, "pubmedqa"))
            idx += 1
        if idx >= max_n:
            break
    return rows


def load_medquad_full(max_n: int = MAX_SAMPLES) -> List[Dict[str, Any]]:
    """MedQuAD full test split — held-out from training data."""
    from datasets import load_dataset

    # Try lavita collection first
    try:
        ds = load_dataset("lavita/medical-qa-datasets", "medquad", split="test")
        rows, idx = [], 0
        items = list(ds)
        random.shuffle(items)
        for item in items:
            q = (item.get("input")  or item.get("question") or "").strip()
            a = (item.get("output") or item.get("answer")   or "").strip()
            if q and a and len(a.split()) > 5:
                rows.append(make_row(idx, q, a, "medquad_full"))
                idx += 1
            if idx >= max_n:
                break
        if rows:
            return rows
    except Exception as e:
        print(f"  lavita/medquad failed: {e}, trying keivalya/MedQuad-MedicalQnADataset...")

    # Fallback
    ds = load_dataset("keivalya/MedQuad-MedicalQnADataset", split="train")
    rows, idx = [], 0
    items = list(ds)
    random.shuffle(items)
    for item in items:
        q = (item.get("Question") or "").strip()
        a = (item.get("Answer")   or "").strip()
        if q and a and len(a.split()) > 5:
            rows.append(make_row(idx, q, a, "medquad_full"))
            idx += 1
        if idx >= max_n:
            break
    return rows


def load_liveqa_full() -> List[Dict[str, Any]]:
    """MEDIQA: medical QA from lavita collection."""
    from datasets import load_dataset
    try:
        ds = load_dataset("lavita/medical-qa-datasets", "medical_meadow_mediqa", split="train")
        rows, idx = [], 0
        items = list(ds)
        random.shuffle(items)
        for item in items:
            q = (item.get("input")  or item.get("question") or "").strip()
            a = (item.get("output") or item.get("answer")   or "").strip()
            if q and a and len(a.split()) > 5:
                rows.append(make_row(idx, q, a, "mediqa"))
                idx += 1
            if idx >= MAX_SAMPLES:
                break
        return rows
    except Exception as e:
        print(f"  mediqa failed: {e}")
        return []


# ── Main ──────────────────────────────────────────────────────────────────────

DATASETS = [
    ("MedicationQA",    "medicationqa_test.parquet",   load_medicationqa),
    ("BioASQ/PubMedQA", "bioasq_test.parquet",         load_bioasq),
    ("MedQuAD full",    "medquad_full_test.parquet",   load_medquad_full),
    ("MEDIQA",          "mediqa_test.parquet",         load_liveqa_full),
]

print("=" * 60)
print("Preparing extra eval datasets")
print("=" * 60)

for name, fname, loader_fn in DATASETS:
    print(f"\n[{name}]")
    try:
        rows = loader_fn()
        if rows:
            save_parquet(rows, OUTPUT_DIR / fname)
        else:
            print(f"  ✗ No rows loaded for {name}")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
for f in sorted(OUTPUT_DIR.glob("*.parquet")):
    df = pd.read_parquet(f)
    # Verify format
    required = ["id", "question", "golden_answers", "prompt", "reward_model", "extra_info"]
    ok = all(c in df.columns for c in required)
    print(f"  {f.name:<40} {len(df):>4} samples  {'✓ format ok' if ok else '✗ format issue'}")

print(f"\nOutput dir: {OUTPUT_DIR}")
print("\nTo run eval on these datasets:")
print("  SKIP_MERGE=1 bash evaluate/eval.sh <exp_name> <step> \\")
print(f"    {OUTPUT_DIR}/medicationqa_test.parquet _medicationqa")