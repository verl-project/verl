#!/usr/bin/env python3
"""
Search-R1 / VERL dataset utility: convert + compare.

Changes vs original:
  - load_rag_jsonl: adds "dataset_name" (real inferred source) alongside "data_source"
  - convert_to_nq_rows: propagates "dataset_name" into extra_info
  - data_source field is UNCHANGED ("csiro_medredqa") — verl reward routing unaffected

Usage:
  # Combined dataset (recommended)
  python searchr1_dataset_tool.py convert \
    --all_file \
    /ocean/projects/med230010p/yji3/MedicalRagChecker/medical_data/runs_csiro_train/rag_generation_outputs_csiro_train.jsonl \
    /ocean/projects/med230010p/yji3/MedicalRagChecker/medical_data/runs_liveqa_full/rag_generation_outputs_liveqa_test.jsonl \
    /ocean/projects/med230010p/yji3/MedicalRagChecker/medical_data/runs_pubmedqa_pqa_artificial/rag_generation_outputs_pubmedqa_train.jsonl \
    --test_sample_n 30 \
    --data_source csiro_medredqa \
    --output_dir ./searchr1_data/combined \
    --prompt_profile medical_checker \
    --append_prompt_to_output_dir

  # Verify dataset_name in output
  python searchr1_dataset_tool.py show_distribution \
    --parquet ./searchr1_data/combined__medical_checker/train.parquet
"""

import argparse
import json
import os
import random
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


# =============================================================================
# Prompt Templates
# =============================================================================

_SYS_SEARCHR1 = (
    "Answer the given question. You must conduct reasoning inside "
    "<think> and </think> first every time you get new information. "
    "After reasoning, if you find you lack some knowledge, you can call a search engine "
    "by <search> query </search> and it will return the top searched results between "
    "<information> and </information>. You can search as many times as you want. "
    "If you find no further external knowledge needed, you can directly provide the answer "
    "inside <answer> and </answer>, without detailed illustrations. "
    "For example, <answer> Beijing </answer>."
)

_SYS_MEDICAL = (
    "Answer the given medical question. You must conduct reasoning inside "
    "<think> and </think> first every time you get new information. "
    "After reasoning, if you find you lack some knowledge, you can call a search engine "
    "by <search> query </search> and it will return the top searched results between "
    "<information> and </information>. You can search as many times as you want. "
    "If you find no further external knowledge needed, you can directly provide the answer "
    "inside <answer> and </answer>, without detailed illustrations. "
    "For example, <answer> Aspirin is used to reduce fever and relieve pain </answer>."
)

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

SYSTEM_INSTRUCTIONS: Dict[str, str] = {
    "searchr1":        _SYS_SEARCHR1,
    "medical":         _SYS_MEDICAL,
    "medical_checker": _SYS_MEDICAL_CHECKER,
}

_FEWSHOT_SEARCHR1 = """\
Here is an example:
Question: What is the population of Tokyo?
Assistant: <think>I need to find the current population of Tokyo. Let me search for this information.</think>
<search>Tokyo population 2024</search>
<information>Tokyo is the capital of Japan with a population of approximately 13.96 million in the city proper, and about 37-38 million in the greater metropolitan area.</information>
<think>I found the answer. Tokyo's population is about 13.96 million in the city and 37-38 million in the greater area.</think>
<answer>approximately 13.96 million (city proper) or 37-38 million (greater metropolitan area)</answer>

Now answer this question:
Question: {question}"""

_FEWSHOT_MEDICAL = """\
Here is an example:
Question: What is the first-line pharmacological treatment for type 2 diabetes?
Assistant: <think>I need to recall the standard first-line pharmacological treatment for type 2 diabetes. Let me search to confirm current recommendations.</think>
<search>first-line pharmacological treatment type 2 diabetes guidelines 2024</search>
<information>According to the American Diabetes Association (ADA) and most international guidelines, metformin remains the preferred initial pharmacological agent for type 2 diabetes management when not contraindicated.</information>
<think>The search confirms that metformin is the first-line pharmacological treatment for type 2 diabetes unless contraindicated (e.g., severe renal impairment, eGFR < 30).</think>
<answer>Metformin is the first-line pharmacological treatment for type 2 diabetes, unless contraindicated.</answer>

Now answer this medical question:
Question: {question}"""

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

FEWSHOT_TEMPLATES: Dict[str, str] = {
    "searchr1":        _FEWSHOT_SEARCHR1,
    "medical":         _FEWSHOT_MEDICAL,
    "medical_checker": _FEWSHOT_MEDICAL_CHECKER,
}


def make_prompt(question: str, prompt_profile: str) -> List[Dict[str, str]]:
    if prompt_profile not in SYSTEM_INSTRUCTIONS:
        raise ValueError(
            f"Unknown prompt_profile: '{prompt_profile}'. "
            f"Choose from: {list(SYSTEM_INSTRUCTIONS.keys())}"
        )
    return [
        {"role": "system", "content": SYSTEM_INSTRUCTIONS[prompt_profile]},
        {"role": "user",   "content": FEWSHOT_TEMPLATES[prompt_profile].format(question=question)},
    ]


# =============================================================================
# Helpers
# =============================================================================

def _to_jsonable(x: Any) -> Any:
    try:
        import numpy as np
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic):  return x.item()
    except Exception:
        pass
    if isinstance(x, dict):             return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):    return [_to_jsonable(v) for v in x]
    if isinstance(x, (bytes, bytearray)): return x.decode("utf-8", errors="replace")
    return x


def truncate(val: Any, max_len: int = 140) -> str:
    try:
        s = json.dumps(_to_jsonable(val), ensure_ascii=False)
    except Exception:
        s = str(val)
    return (s[:max_len - 3] + "...") if len(s) > max_len else s


def _is_listlike(x: Any) -> bool:
    try:
        import numpy as np
        return isinstance(x, (list, tuple, np.ndarray))
    except Exception:
        return isinstance(x, (list, tuple))


def _data_source_from_path(path: str) -> str:
    """
    Infer dataset name from filename.
    rag_generation_outputs_csiro_train.jsonl    -> csiro
    rag_generation_outputs_liveqa_full.jsonl    -> liveqa_full
    rag_generation_outputs_pubmedqa_train.jsonl -> pubmedqa_pqa_artificial
    rag_generation_outputs_medquad_full.jsonl   -> medquad
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    stem = re.sub(r"^rag_generation_outputs_", "", stem)
    stem = re.sub(r"_(train|test|val|dev)$", "", stem)
    return stem or os.path.splitext(os.path.basename(path))[0]


# =============================================================================
# Data loading
# =============================================================================

@dataclass
class QAExample:
    question: str
    answer:   str
    extra:    Dict[str, Any]


def load_rag_jsonl(path: str) -> List[QAExample]:
    """
    Load JSONL and attach both:
      data_source  = "csiro_medredqa"  (unchanged, verl uses this for reward routing)
      dataset_name = real source inferred from filename  (new, for analysis)
    """
    real_source = _data_source_from_path(path)   # e.g. "csiro", "liveqa_full", "pubmedqa_pqa_artificial"
    auto_source = "csiro_medredqa"               # keep unchanged for verl

    out: List[QAExample] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue
            q = (item.get("query") or item.get("question") or "").strip()
            a = (item.get("gt_answer") or item.get("answer") or item.get("final_answer") or "").strip()
            if not q or not a:
                continue
            extra = {
                "source_path":      path,
                "data_source":      auto_source,   # unchanged
                "dataset_name":     real_source,   # NEW: real per-file source
                "line":             i,
                "query_id":         item.get("query_id"),
                "retrieved_context": item.get("retrieved_context", []),
            }
            out.append(QAExample(question=q, answer=a, extra=extra))
    return out


def load_raw_json_chunks(path: str) -> List[QAExample]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and any(str(k).isdigit() for k in data.keys()):
        data = list(data.values())
    if not isinstance(data, list):
        data = [data]
    out: List[QAExample] = []
    for item in data:
        meta = item.get("meta", {}) if isinstance(item, dict) else {}
        q = (meta.get("question") or item.get("question") or "").strip()
        a = (meta.get("answer") or item.get("answer") or item.get("gt_answer") or "").strip()
        if not q or not a:
            continue
        out.append(QAExample(question=q, answer=a, extra={
            "source_path":  path,
            "dataset_name": _data_source_from_path(path),
            "data_source":  "csiro_medredqa",
        }))
    return out


def load_loader_output(path: str) -> List[QAExample]:
    txt = open(path, "r", encoding="utf-8").read().strip()
    items: List[Dict[str, Any]] = []
    try:
        obj = json.loads(txt)
        if isinstance(obj, list):   items = obj
        elif isinstance(obj, dict): items = list(obj.values()) if "question" not in obj else [obj]
    except Exception:
        for line in txt.splitlines():
            line = line.strip()
            if not line: continue
            try: items.append(json.loads(line))
            except Exception: continue
    out: List[QAExample] = []
    for it in items:
        if not isinstance(it, dict): continue
        q = (it.get("question") or it.get("query") or it.get("prompt") or "").strip()
        a = (it.get("answer") or it.get("final_answer") or it.get("gt_answer") or it.get("long_answer") or "").strip()
        if not q or not a: continue
        out.append(QAExample(question=q, answer=a, extra={
            "source_path":  path,
            "dataset_name": _data_source_from_path(path),
            "data_source":  "csiro_medredqa",
        }))
    return out


def detect_and_load(path: str, mode: str) -> List[QAExample]:
    if mode == "jsonl":    return load_rag_jsonl(path)
    if mode == "raw_json": return load_raw_json_chunks(path)
    if mode == "loader":   return load_loader_output(path)
    if path.endswith(".jsonl"): return load_rag_jsonl(path)
    return load_loader_output(path)


# =============================================================================
# Difficulty heuristics
# =============================================================================

def estimate_difficulty(question: str) -> str:
    words   = question.split()
    q_len   = len(words)
    q_lower = question.lower()
    length_score = 0 if q_len < 25 else 1 if q_len < 60 else 2 if q_len < 120 else 3
    reasoning_keywords = [
        "most likely", "best explanation", "most appropriate", "next step",
        "differential diagnosis", "which of the following", "mechanism",
        "pathophysiology", "contraindicated", "first-line", "gold standard",
        "why", "how does", "relationship", "explain",
    ]
    keyword_score  = sum(1 for k in reasoning_keywords if k in q_lower)
    vignette_score = 1 if ("\n" in question or q_len >= 80) else 0
    total = length_score + keyword_score + vignette_score
    return "easy" if total <= 1 else "medium" if total <= 4 else "hard"


# =============================================================================
# Conversion to NQ schema
# =============================================================================

def convert_to_nq_rows(
    examples:       List[QAExample],
    split:          str,
    data_source:    str,
    prompt_profile: str,
    ability:        str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for idx, ex in enumerate(examples):
        per_ex_source = ex.extra.get("data_source") or data_source
        per_ex_source = "csiro_medredqa"                          # unchanged
        dataset_name  = ex.extra.get("dataset_name", "unknown")  # NEW

        rows.append({
            "id":             f"{split}_{idx}",
            "question":       ex.question,
            "golden_answers": [ex.answer],
            "data_source":    per_ex_source,                      # unchanged
            "prompt":         make_prompt(ex.question, prompt_profile),
            "ability":        ability,
            "reward_model": {
                "ground_truth": {"target": [ex.answer]},
                "style":        "rule",
            },
            "extra_info": {
                "index":         idx,
                "split":         split,
                "data_source":   per_ex_source,                   # unchanged
                "dataset_name":  dataset_name,                    # NEW
                "prompt_profile": prompt_profile,
            },
            "metadata":   None,
            "agent_name": "tool_agent",
        })
    return rows


# =============================================================================
# Save helpers
# =============================================================================

def save_parquet(rows: List[Dict[str, Any]], out_path: str) -> None:
    import pandas as pd
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    pd.DataFrame(rows).to_parquet(out_path, engine="pyarrow", index=False)


def save_jsonl(rows: List[Dict[str, Any]], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(_to_jsonable(r), ensure_ascii=False) + "\n")


def build_corpus_from_rag_jsonl(jsonl_files: List[str], out_path: str) -> None:
    seen: set = set()
    passages: List[Dict[str, str]] = []
    for fpath in jsonl_files:
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try: item = json.loads(line)
                except Exception: continue
                for ctx in item.get("retrieved_context", []) or []:
                    text = (ctx.get("text") or "").strip()
                    if not text or text in seen: continue
                    seen.add(text)
                    title = (ctx.get("title") or "").strip()
                    doc_id = (ctx.get("doc_id") or "").strip()
                    contents = f"{title}\n{text}".strip() if title else text
                    passages.append({"id": doc_id or str(len(passages)), "contents": contents})
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for p in passages:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")


# =============================================================================
# NQ reference comparison
# =============================================================================

def download_nq_train_parquet(cache_dir: Optional[str] = None) -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(
        repo_id="PeterJinGo/nq_hotpotqa_train", filename="train.parquet",
        repo_type="dataset", cache_dir=cache_dir,
    )


def read_parquet_samples(path: str, n: int = 5) -> Tuple[List[Dict[str, Any]], int]:
    import pandas as pd
    import pyarrow.parquet as pq
    df = pd.read_parquet(path)
    samples = [df.iloc[i].to_dict() for i in range(min(n, len(df)))]
    return samples, int(pq.ParquetFile(path).metadata.num_rows)


def read_jsonl_samples(path: str, n: int = 5) -> Tuple[List[Dict[str, Any]], int]:
    samples, total = [], 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            total += 1
            if len(samples) < n:
                try: samples.append(json.loads(line))
                except Exception: pass
    return samples, total


def parquet_schema_str(path: str) -> str:
    import pyarrow.parquet as pq
    return str(pq.read_schema(path))


def compare_against_nq(
    my_samples: List[Dict[str, Any]], my_total: int, my_label: str,
    nq_samples: List[Dict[str, Any]], nq_total: int,
    my_parquet_path: Optional[str] = None, nq_parquet_path: Optional[str] = None,
) -> None:
    print(f"\n{'='*80}\n1) Basic stats\n{'='*80}")
    print(f"{'':30s} {'NQ (native)':>16s}   {my_label:>16s}")
    print(f"{'Total rows':30s} {nq_total:>16d}   {my_total:>16d}")

    print(f"\n{'='*80}\n2) Column names\n{'='*80}")
    nq_cols = set(nq_samples[0].keys()) if nq_samples else set()
    my_cols = set(my_samples[0].keys()) if my_samples else set()
    for c in sorted(nq_cols | my_cols):
        print(f"{c:30s} NQ: {'✅' if c in nq_cols else '❌'}   YOU: {'✅' if c in my_cols else '❌'}")

    if nq_parquet_path and my_parquet_path:
        print(f"\n{'='*80}\n3) Schema\n{'='*80}")
        print("[NQ]\n" + parquet_schema_str(nq_parquet_path))
        print("[You]\n" + parquet_schema_str(my_parquet_path))

    if not (nq_samples and my_samples): return

    print(f"\n{'='*80}\n4) First row\n{'='*80}")
    for k in sorted(nq_samples[0].keys()):
        print(f"\n- {k}:\n  NQ : {truncate(nq_samples[0].get(k),220)}\n  YOU: {truncate(my_samples[0].get(k),220)}")

    print(f"\n{'='*80}\n5) Prompt + dataset_name check\n{'='*80}")
    my0    = my_samples[0]
    issues = []
    for c in ["id","question","golden_answers","data_source","prompt","ability","reward_model","extra_info","metadata"]:
        if c not in my0: issues.append(f"Missing column: {c}")

    if "extra_info" in my0:
        ei = my0["extra_info"]
        if isinstance(ei, dict):
            if "dataset_name" not in ei:
                issues.append("extra_info.dataset_name MISSING — run the updated convert script")
            else:
                print(f"✅ extra_info.dataset_name present: {ei['dataset_name']}")

    if "prompt" in my0:
        p = my0["prompt"]
        if _is_listlike(p) and len(p) > 0 and isinstance(p[0], dict):
            all_content = " ".join(str(m.get("content","")) for m in p if isinstance(m,dict))
            for tag in ["<think>","</think>","<answer>","</answer>"]:
                if tag not in all_content:
                    issues.append(f"Missing few-shot marker: {tag}")
            if "<search>" not in all_content:
                issues.append("Missing <search> in few-shot")

    if issues:
        print(f"⚠️  {len(issues)} issue(s):")
        for i, msg in enumerate(issues,1): print(f"  {i}. {msg}")
    else:
        print("✅ All checks passed.")


# =============================================================================
# Dataset name distribution helper
# =============================================================================

def show_distribution(parquet_path: str) -> None:
    """Print dataset_name distribution from a parquet file."""
    import pandas as pd
    df = pd.read_parquet(parquet_path)
    print(f"\nFile: {parquet_path}  ({len(df)} rows)")

    # extra_info is stored as dict — extract dataset_name
    if "extra_info" in df.columns:
        def get_dn(ei):
            if isinstance(ei, dict): return ei.get("dataset_name", "unknown")
            try:
                d = json.loads(str(ei))
                return d.get("dataset_name", "unknown")
            except Exception:
                return "unknown"
        df["_dataset_name"] = df["extra_info"].apply(get_dn)
        counts = df["_dataset_name"].value_counts()
        print("\ndataset_name distribution:")
        for name, cnt in counts.items():
            print(f"  {name:<40} {cnt:>5}  ({cnt/len(df)*100:.1f}%)")
    else:
        print("No extra_info column found.")

    if "data_source" in df.columns:
        print("\ndata_source distribution (for reference):")
        for name, cnt in df["data_source"].value_counts().items():
            print(f"  {name:<40} {cnt:>5}")


# =============================================================================
# CLI commands
# =============================================================================

def cmd_convert(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    prompt_profile = args.prompt_profile

    if args.append_prompt_to_output_dir:
        suffix = re.sub(r"[^a-zA-Z0-9_\-]+", "_", prompt_profile)
        args.output_dir = f"{args.output_dir.rstrip('/')}__{suffix}"

    os.makedirs(args.output_dir, exist_ok=True)
    difficulty_filter: Optional[set] = {d.strip() for d in args.difficulty.split(",") if d.strip()} if args.difficulty else None

    def filter_and_dedup(examples: List[QAExample], label: str, exclude_keys: Optional[set] = None) -> List[QAExample]:
        before = len(examples)
        if args.min_answer_len > 0:
            examples = [e for e in examples if len(e.answer.split()) >= args.min_answer_len]
        for e in examples: e.extra["difficulty"] = estimate_difficulty(e.question)
        if difficulty_filter:
            examples = [e for e in examples if e.extra.get("difficulty") in difficulty_filter]
        seen: set = set()
        out: List[QAExample] = []
        for e in examples:
            key = e.question.strip()[:200]
            if key in seen or (exclude_keys and key in exclude_keys): continue
            seen.add(key); out.append(e)
        print(f"  {label}: {before} -> {len(out)} after filters/dedup")
        return out

    all_files = args.all_file
    if all_files:
        n = args.test_sample_n
        test_ex: List[QAExample] = []
        train_ex: List[QAExample] = []
        for f in all_files:
            ex  = detect_and_load(f, args.mode)
            src = _data_source_from_path(f)
            if n > 0 and len(ex) > n:
                indices = list(range(len(ex)))
                random.shuffle(indices)
                test_ex.extend(ex[i] for i in indices[:n])
                train_ex.extend(ex[i] for i in indices[n:])
                print(f"[{src}] {os.path.basename(f)}: total={len(ex)} -> test={n}, train={len(ex)-n}")
            else:
                train_ex.extend(ex)
                print(f"[{src}] {os.path.basename(f)}: total={len(ex)} -> test=0, train={len(ex)}")
        test_ex  = filter_and_dedup(test_ex, "test")
        test_keys = {e.question.strip()[:200] for e in test_ex}
        train_ex = filter_and_dedup(train_ex, "train", exclude_keys=test_keys)
    else:
        train_ex, test_ex = [], []
        for f in args.train_file:
            ex = detect_and_load(f, args.mode); train_ex.extend(ex)
            print(f"Loaded train: {os.path.basename(f)} -> {len(ex)}")
        for f in args.val_file + args.test_file:
            ex = detect_and_load(f, args.mode); test_ex.extend(ex)
            print(f"Loaded test:  {os.path.basename(f)} -> {len(ex)}")
        test_ex  = filter_and_dedup(test_ex, "test")
        test_keys = {e.question.strip()[:200] for e in test_ex}
        train_ex = filter_and_dedup(train_ex, "train", exclude_keys=test_keys)

    # Print distribution
    print(f"\n{'='*50}")
    print(f"TRAIN: {len(train_ex)}  |  TEST: {len(test_ex)}")
    for split_name, exs in [("train", train_ex), ("test", test_ex)]:
        if exs:
            dist = Counter(e.extra.get("dataset_name", "unknown") for e in exs)
            print(f"  {split_name} dataset_name distribution: {dict(dist)}")
    print(f"{'='*50}\n")

    train_rows = convert_to_nq_rows(train_ex, "train", args.data_source, prompt_profile, args.ability)
    test_rows  = convert_to_nq_rows(test_ex,  "test",  args.data_source, prompt_profile, args.ability)

    out_train = os.path.join(args.output_dir, "train.parquet")
    out_test  = os.path.join(args.output_dir, "test.parquet")
    try:
        save_parquet(train_rows, out_train); save_parquet(test_rows, out_test)
        print(f"Saved: {out_train} ({len(train_rows)} rows)")
        print(f"Saved: {out_test}  ({len(test_rows)} rows)")
    except Exception as e:
        print(f"Parquet save failed ({e}); falling back to JSONL.")
        save_jsonl(train_rows, os.path.join(args.output_dir, "train.jsonl"))
        save_jsonl(test_rows,  os.path.join(args.output_dir, "test.jsonl"))

    if args.build_corpus:
        jsonl_inputs = [p for p in (all_files or args.train_file + args.val_file + args.test_file) if p.endswith(".jsonl")]
        if jsonl_inputs:
            corpus_out = args.corpus_output or os.path.join(args.output_dir, "corpus.jsonl")
            build_corpus_from_rag_jsonl(jsonl_inputs, corpus_out)
            print(f"Saved corpus: {corpus_out}")


def cmd_compare(args: argparse.Namespace) -> None:
    nq_path = args.nq_parquet or download_nq_train_parquet(cache_dir=args.hf_cache_dir)
    nq_samples, nq_total = read_parquet_samples(nq_path, n=args.n_samples)
    if args.my_parquet:
        my_samples, my_total = read_parquet_samples(args.my_parquet, n=args.n_samples)
        compare_against_nq(my_samples, my_total, "YOUR parquet", nq_samples, nq_total,
                           my_parquet_path=args.my_parquet, nq_parquet_path=nq_path)
    else:
        my_samples, my_total = read_jsonl_samples(args.my_jsonl, n=args.n_samples)
        compare_against_nq(my_samples, my_total, "YOUR JSONL", nq_samples, nq_total)


def cmd_convert_and_compare(args: argparse.Namespace) -> None:
    cmd_convert(args)
    my_train = os.path.join(args.output_dir, "train.parquet")
    if not os.path.exists(my_train): return
    compare_args = argparse.Namespace(
        nq_parquet=args.nq_parquet, hf_cache_dir=args.hf_cache_dir,
        my_parquet=my_train, my_jsonl=None, n_samples=args.n_samples,
    )
    cmd_compare(compare_args)


def cmd_show_prompt(args: argparse.Namespace) -> None:
    question = args.question or "What are the main causes of hypertension?"
    prompt = make_prompt(question, args.prompt_profile)
    print(f"\n=== Rendered prompt (profile={args.prompt_profile}) ===\n")
    for msg in prompt:
        print(f"[{msg['role'].upper()}]\n{msg['content']}\n")


def cmd_show_distribution(args: argparse.Namespace) -> None:
    show_distribution(args.parquet)


# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Search-R1 dataset tool")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_convert_flags(p: argparse.ArgumentParser) -> None:
        p.add_argument("--mode", choices=["jsonl","raw_json","loader","auto"], default="auto")
        p.add_argument("--all_file",    nargs="+", default=[])
        p.add_argument("--train_file",  nargs="+", default=[])
        p.add_argument("--val_file",    nargs="+", default=[])
        p.add_argument("--test_file",   nargs="+", default=[])
        p.add_argument("--output_dir",  default="./searchr1_data")
        p.add_argument("--data_source", default="csiro_medredqa")
        p.add_argument("--difficulty",  default=None)
        p.add_argument("--min_answer_len", type=int, default=1)
        p.add_argument("--ability",     default="medical-reasoning")
        p.add_argument("--seed",        type=int, default=42)
        p.add_argument("--build_corpus", action="store_true")
        p.add_argument("--corpus_output", default=None)
        p.add_argument("--prompt_profile",
                       choices=["searchr1","medical","medical_checker"], default="medical")
        p.add_argument("--append_prompt_to_output_dir", action="store_true")
        p.add_argument("--nq_parquet",   default=None)
        p.add_argument("--hf_cache_dir", default=None)
        p.add_argument("--n_samples",    type=int, default=5)
        p.add_argument("--test_sample_n",type=int, default=30)

    p_c = sub.add_parser("convert");             add_convert_flags(p_c)
    p_cac = sub.add_parser("convert_and_compare"); add_convert_flags(p_cac)

    p_cmp = sub.add_parser("compare")
    p_cmp.add_argument("--nq_parquet", default=None)
    p_cmp.add_argument("--hf_cache_dir", default=None)
    p_cmp.add_argument("--my_parquet", default=None)
    p_cmp.add_argument("--my_jsonl",   default=None)
    p_cmp.add_argument("--n_samples",  type=int, default=5)

    p_sp = sub.add_parser("show_prompt")
    p_sp.add_argument("--prompt_profile",
                      choices=["searchr1","medical","medical_checker"], default="medical")
    p_sp.add_argument("--question", default=None)

    p_sd = sub.add_parser("show_distribution")
    p_sd.add_argument("--parquet", required=True, help="Path to parquet file to inspect")

    return parser


def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()
    dispatch = {
        "convert":              cmd_convert,
        "compare":              cmd_compare,
        "convert_and_compare":  cmd_convert_and_compare,
        "show_prompt":          cmd_show_prompt,
        "show_distribution":    cmd_show_distribution,
    }
    if args.cmd not in dispatch:
        raise SystemExit(f"Unknown command: {args.cmd}")
    if args.cmd == "compare" and not args.my_parquet and not getattr(args, "my_jsonl", None):
        raise SystemExit("Please provide --my_parquet or --my_jsonl.")
    dispatch[args.cmd](args)


if __name__ == "__main__":
    main()
