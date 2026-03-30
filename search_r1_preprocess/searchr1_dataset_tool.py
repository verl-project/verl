# #!/usr/bin/env python3
# """
# Search-R1 / VERL dataset utility: convert + compare in one place.

# This script consolidates three workflows:
# 1) Convert your medical/RAG JSONL (or other formats) into a Search-R1 (NQ-compatible) parquet.
# 2) Optionally build a corpus.jsonl from retrieved contexts (if present in the JSONL).
# 3) Compare your parquet (or JSONL) against the Search-R1 native NQ dataset schema and samples.

# Key design choice:
# - Avoid `datasets.load_dataset()` for the NQ parquet to prevent schema-casting failures across splits.
#   We download `train.parquet` directly from HuggingFace and read it with pandas/pyarrow.
# python search_r1_preprocess/searchr1_dataset_tool.py convert_and_compare \
#   --mode jsonl \
#   --train_file  /ocean/projects/med230010p/yji3/MedicalRagChecker/medical_data/runs_csiro_train/rag_generation_outputs_csiro_train.jsonl \
#   --test_file   /ocean/projects/med230010p/yji3/MedicalRagChecker/medical_data/runs_csiro_test/rag_generation_outputs_csiro_test.jsonl \
#   --output_dir ./searchr1_data/medredqa \
#   --data_source csiro_medredqa

# python search_r1_preprocess/searchr1_dataset_tool.py compare \
#  --my_parquet ./searchr1_data/medredqa/train.parquet


# For double check

# python - << 'PY'
# import pyarrow.parquet as pq
# print(pq.read_schema("./searchr1_data/medredqa/train.parquet"))
# PY


"""
Search-R1 / VERL dataset utility: convert + compare in one place.

This script consolidates three workflows:
1) Convert your medical/RAG JSONL (or other formats) into a Search-R1 (NQ-compatible) parquet.
2) Optionally build a corpus.jsonl from retrieved contexts (if present in the JSONL).
3) Compare your parquet (or JSONL) against the Search-R1 native NQ dataset schema and samples.


防止 os error
export XDG_CACHE_HOME=/ocean/projects/med230010p/yji3/.cache
export HF_HOME=/ocean/projects/med230010p/yji3/.cache/huggingface
export HF_DATASETS_CACHE=/ocean/projects/med230010p/yji3/.cache/huggingface/datasets
export TRANSFORMERS_CACHE=/ocean/projects/med230010p/yji3/.cache/huggingface/transformers
export HF_HUB_CACHE=/ocean/projects/med230010p/yji3/.cache/huggingface/hub



Key design choice:

- Avoid `datasets.load_dataset()` for the NQ parquet to prevent schema-casting failures across splits.
  We download `train.parquet` directly from HuggingFace and read it with pandas/pyarrow.
只有 csiro_medredqa的 
python search_r1_preprocess/searchr1_dataset_tool.py convert_and_compare \
  --mode jsonl \
  --train_file  /ocean/projects/med230010p/yji3/MedicalRagChecker/medical_data/runs_csiro_train/rag_generation_outputs_csiro_train.jsonl \
  --test_file   /ocean/projects/med230010p/yji3/MedicalRagChecker/medical_data/runs_csiro_test/rag_generation_outputs_csiro_test.jsonl \
  --output_dir ./searchr1_data/medredqa \
  --data_source csiro_medredqa \
    --prompt_profile medical_checker \
  --append_prompt_to_output_dir 

python search_r1_preprocess/searchr1_dataset_tool.py convert \
  --all_file \    /ocean/projects/med230010p/yji3/MedicalRagChecker/medical_data/runs_csiro_train/rag_generation_outputs_csiro_train.jsonl \    /ocean/projects/med230010p/yji3/MedicalRagChecker/medical_data/runs_liveqa_full/rag_generation_outputs_liveqa_test.jsonl \    /ocean/projects/med230010p/yji3/MedicalRagChecker/medical_data/runs_pubmedqa_pqa_artificial/rag_generation_outputs_pubmedqa_train.jsonl \
  --test_sample_n 30 \
  --data_source csiro_medredqa \
  --output_dir ./searchr1_data/combined \
  --prompt_profile medical_checker \
  --append_prompt_to_output_dir


只有 search的 case 


python search_r1_preprocess/searchr1_dataset_tool.py convert \
  --mode jsonl \
  --all_file \
    /ocean/projects/med230010p/yji3/MedicalRagChecker/medical_data/runs_csiro_train/rag_generation_outputs_csiro_train.jsonl \
    /ocean/projects/med230010p/yji3/MedicalRagChecker/medical_data/runs_liveqa_full/rag_generation_outputs_liveqa_test.jsonl \
    /ocean/projects/med230010p/yji3/MedicalRagChecker/medical_data/runs_pubmedqa_pqa_artificial/rag_generation_outputs_pubmedqa_train.jsonl \
  --test_sample_n 30 \
  --data_source csiro_medredqa \
  --output_dir ./searchr1_data/combined \
  --prompt_profile medical \
  --append_prompt_to_output_dir

prompt_profile参数说明
searchr1 <think> + <search> + <answer>通用问答
medical <think> + <search> + <answer>医疗问答（换了示例问题）
medical_checker<think> + <search> + <check> + <answer>医疗问答 + 验证步骤

  
append_prompt_to_output_dir
# 不加这个参数
--output_dir ./searchr1_data/medredqa
# 输出到：./searchr1_data/medredqa/

# 加了这个参数 + profile=medical
--output_dir ./searchr1_data/medredqa --append_prompt_to_output_dir
# 输出到：./searchr1_data/medredqa__medical/

# 加了这个参数 + profile=medical_checker
--output_dir ./searchr1_data/medredqa --append_prompt_to_output_dir
# 输出到：./searchr1_data/medredqa__medical_checker/
```

这样你如果想**对比不同 prompt 策略的训练效果**，可以同时保留两份数据不互相覆盖：
```
searchr1_data/
├── medredqa__medical/
│   ├── train.parquet   ← 用 think+search+answer
│   └── test.parquet
└── medredqa__medical_checker/
    ├── train.parquet   ← 用 think+search+check+answer
    └── test.parquet
"""

"""
Search-R1 / VERL dataset utility: convert + compare in one place.

This script consolidates three workflows:
1) Convert your medical/RAG JSONL (or other formats) into a Search-R1 (NQ-compatible) parquet.
2) Optionally build a corpus.jsonl from retrieved contexts (if present in the JSONL).
3) Compare your parquet (or JSONL) against the Search-R1 native NQ dataset schema and samples.

Key design choice:
- Avoid `datasets.load_dataset()` for the NQ parquet to prevent schema-casting failures across splits.
  We download `train.parquet` directly from HuggingFace and read it with pandas/pyarrow.

Usage examples:

python searchr1_dataset_tool.py convert_and_compare \
  --mode jsonl \
  --train_file  /path/to/train.jsonl \
  --test_file   /path/to/test.jsonl \
  --output_dir ./searchr1_data/medredqa \
  --data_source csiro_medredqa

python searchr1_dataset_tool.py compare \
 --my_parquet ./searchr1_data/medredqa/train.parquet
"""



import argparse
import json
import os
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


# =============================================================================
# Prompt Templates  ← 核心修改区域
# =============================================================================
#
# 关键设计：必须在 user 消息里内嵌 few-shot example，
# 让模型看到 <think>/<search>/<answer> 的示范格式后才会模仿。
# 没有 few-shot example → 模型不知道要生成这些 tag。
#
# 对应原始 test_fewshot.parquet 的格式：
#   system: 规则说明
#   user:   示例 + "Now answer this question:\nQuestion: {question}"
# =============================================================================

# ---------- System prompts (规则说明，不含示例) ----------

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

# ---------- Few-shot user templates ----------
# {question} は実際の質問に置換される

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
Assistant: <think>I need to recall the standard first-line pharmacological treatment for type 2 diabetes. This is a well-established clinical guideline. Let me search to confirm current recommendations.</think>
<search>first-line pharmacological treatment type 2 diabetes guidelines 2024</search>
<information>According to the American Diabetes Association (ADA) and most international guidelines, metformin remains the preferred initial pharmacological agent for type 2 diabetes management when not contraindicated, due to its efficacy, safety profile, low cost, and potential cardiovascular benefits.</information>
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


# =============================================================================
# make_prompt  ← 核心修改：user 消息内嵌 few-shot example
# =============================================================================

def make_prompt(question: str, prompt_profile: str) -> List[Dict[str, str]]:
    """
    构造 Search-R1 兼容的 prompt（system + user 两条消息）。

    user 消息 = few-shot 示例 + 真正的问题
    → 模型看到示例后会模仿 <think>/<search>/[<check>]/<answer> 格式

    对应原始 test_fewshot.parquet 的结构（已验证）：
      [{'role': 'system', 'content': ...规则说明...},
       {'role': 'user',   'content': ...few-shot示例...\\nNow answer:\\nQuestion: ...}]
    """
    if prompt_profile not in SYSTEM_INSTRUCTIONS:
        raise ValueError(
            f"Unknown prompt_profile: '{prompt_profile}'. "
            f"Choose from: {list(SYSTEM_INSTRUCTIONS.keys())}"
        )

    system_content = SYSTEM_INSTRUCTIONS[prompt_profile]
    user_content   = FEWSHOT_TEMPLATES[prompt_profile].format(question=question)

    return [
        {"role": "system", "content": system_content},
        {"role": "user",   "content": user_content},
    ]


# =============================================================================
# 以下は元のコードをそのまま保持
# =============================================================================

# -------------------------
# JSON-safe pretty printing
# -------------------------

def _to_jsonable(x: Any) -> Any:
    try:
        import numpy as np
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
    except Exception:
        pass
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    if isinstance(x, (bytes, bytearray)):
        return x.decode("utf-8", errors="replace")
    return x


def truncate(val: Any, max_len: int = 140) -> str:
    try:
        s = json.dumps(_to_jsonable(val), ensure_ascii=False)
    except Exception:
        s = str(val)
    return (s[: max_len - 3] + "...") if len(s) > max_len else s


def _is_listlike(x: Any) -> bool:
    try:
        import numpy as np
        return isinstance(x, (list, tuple, np.ndarray))
    except Exception:
        return isinstance(x, (list, tuple))


# -------------------------
# Input loading / adapters
# -------------------------

@dataclass
class QAExample:
    question: str
    answer: str
    extra: Dict[str, Any]


def _data_source_from_path(path: str) -> str:
    """
    自动从文件名推断 data_source 标签。
    例：rag_generation_outputs_liveqa_test.jsonl  → liveqa
        rag_generation_outputs_pubmedqa_train.jsonl → pubmedqa
        rag_generation_outputs_csiro_train.jsonl    → csiro
    逻辑：去掉 rag_generation_outputs_ 前缀和 _train/_test/_val 后缀，
    保留中间部分。如果无法解析就用完整 stem。
    """
    stem = os.path.splitext(os.path.basename(path))[0]  # 去掉 .jsonl
    stem = re.sub(r"^rag_generation_outputs_", "", stem)  # 去掉前缀
    stem = re.sub(r"_(train|test|val|dev)$", "", stem)    # 去掉 split 后缀
    return stem or os.path.splitext(os.path.basename(path))[0]


def load_rag_jsonl(path: str) -> List[QAExample]:
    # auto_source = _data_source_from_path(path)
    auto_source = "csiro_medredqa"
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
                "source_path": path,
                "data_source": auto_source,   # ← 每条样本记录自己的来源
                "line": i,
                "query_id": item.get("query_id"),
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
        out.append(QAExample(question=q, answer=a, extra={"source_path": path}))
    return out


def load_loader_output(path: str) -> List[QAExample]:
    txt = open(path, "r", encoding="utf-8").read().strip()
    items: List[Dict[str, Any]] = []
    try:
        obj = json.loads(txt)
        if isinstance(obj, list):
            items = obj
        elif isinstance(obj, dict):
            items = list(obj.values()) if "question" not in obj else [obj]
    except Exception:
        for line in txt.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                continue
    out: List[QAExample] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        q = (it.get("question") or it.get("query") or it.get("prompt") or "").strip()
        a = (it.get("answer") or it.get("final_answer") or it.get("gt_answer") or it.get("long_answer") or "").strip()
        if not q or not a:
            continue
        out.append(QAExample(question=q, answer=a, extra={"source_path": path}))
    return out


def detect_and_load(path: str, mode: str) -> List[QAExample]:
    if mode == "jsonl":
        return load_rag_jsonl(path)
    if mode == "raw_json":
        return load_raw_json_chunks(path)
    if mode == "loader":
        return load_loader_output(path)
    if path.endswith(".jsonl"):
        return load_rag_jsonl(path)
    return load_loader_output(path)


# -------------------------
# Difficulty heuristics
# -------------------------

def estimate_difficulty(question: str) -> str:
    words = question.split()
    q_len = len(words)
    q_lower = question.lower()
    if q_len < 25:
        length_score = 0
    elif q_len < 60:
        length_score = 1
    elif q_len < 120:
        length_score = 2
    else:
        length_score = 3
    reasoning_keywords = [
        "most likely", "best explanation", "most appropriate",
        "next step", "differential diagnosis", "which of the following",
        "mechanism", "pathophysiology", "contraindicated",
        "first-line", "gold standard", "why", "how does",
        "relationship", "explain",
    ]
    keyword_score = sum(1 for k in reasoning_keywords if k in q_lower)
    vignette_score = 1 if ("\n" in question or q_len >= 80) else 0
    total = length_score + keyword_score + vignette_score
    if total <= 1:
        return "easy"
    if total <= 4:
        return "medium"
    return "hard"


# -------------------------
# Conversion to NQ schema
# -------------------------

def convert_to_nq_rows(
    examples: List[QAExample],
    split: str,
    data_source: str,
    prompt_profile: str,
    ability: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for idx, ex in enumerate(examples):
        # 优先用每条样本自带的 data_source（来自文件名推断），否则用全局默认值
        per_ex_source = ex.extra.get("data_source") or data_source
        per_ex_source = "csiro_medredqa"
        rows.append(
            {
                "id": f"{split}_{idx}",
                "question": ex.question,
                "golden_answers": [ex.answer],
                "data_source": per_ex_source,
                "prompt": make_prompt(ex.question, prompt_profile),
                "ability": ability,
                "reward_model": {
                    "ground_truth": {"target": [ex.answer]},
                    "style": "rule",
                },
                "extra_info": {
                    "index": idx,
                    "split": split,
                    "data_source": per_ex_source,
                    "prompt_profile": prompt_profile,
                },
                "metadata": None,
                "agent_name": "tool_agent",
            }
        )
    return rows


# -------------------------
# Saving and corpus building
# -------------------------

def save_parquet(rows: List[Dict[str, Any]], out_path: str) -> None:
    import pandas as pd
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(out_path, engine="pyarrow", index=False)


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
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except Exception:
                    continue
                for ctx in item.get("retrieved_context", []) or []:
                    text = (ctx.get("text") or "").strip()
                    if not text or text in seen:
                        continue
                    seen.add(text)
                    title = (ctx.get("title") or "").strip()
                    doc_id = (ctx.get("doc_id") or "").strip()
                    contents = f"{title}\n{text}".strip() if title else text
                    passages.append({"id": doc_id or str(len(passages)), "contents": contents})
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for p in passages:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")


# -------------------------
# NQ reference + comparison
# -------------------------

def download_nq_train_parquet(cache_dir: Optional[str] = None) -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(
        repo_id="PeterJinGo/nq_hotpotqa_train",
        filename="train.parquet",
        repo_type="dataset",
        cache_dir=cache_dir,
    )


def read_parquet_samples(path: str, n: int = 5) -> Tuple[List[Dict[str, Any]], int]:
    import pandas as pd
    import pyarrow.parquet as pq
    df = pd.read_parquet(path)
    samples = [df.iloc[i].to_dict() for i in range(min(n, len(df)))]
    total = pq.ParquetFile(path).metadata.num_rows
    return samples, int(total)


def read_jsonl_samples(path: str, n: int = 5) -> Tuple[List[Dict[str, Any]], int]:
    samples: List[Dict[str, Any]] = []
    total = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            total += 1
            if len(samples) < n:
                try:
                    samples.append(json.loads(line))
                except Exception:
                    pass
    return samples, total


def parquet_schema_str(path: str) -> str:
    import pyarrow.parquet as pq
    return str(pq.read_schema(path))


def compare_against_nq(
    my_samples: List[Dict[str, Any]],
    my_total: int,
    my_label: str,
    nq_samples: List[Dict[str, Any]],
    nq_total: int,
    my_parquet_path: Optional[str] = None,
    nq_parquet_path: Optional[str] = None,
) -> None:
    print("\n" + "=" * 80)
    print("1) Basic stats")
    print("=" * 80)
    print(f"{'':30s} {'NQ (native)':>16s}   {my_label:>16s}")
    print(f"{'Total rows':30s} {nq_total:>16d}   {my_total:>16d}")
    print(f"{'Num columns (sample)':30s} {len(nq_samples[0]) if nq_samples else 0:>16d}   {len(my_samples[0]) if my_samples else 0:>16d}")

    print("\n" + "=" * 80)
    print("2) Column names")
    print("=" * 80)
    nq_cols = set(nq_samples[0].keys()) if nq_samples else set()
    my_cols = set(my_samples[0].keys()) if my_samples else set()
    all_cols = sorted(nq_cols | my_cols)
    for c in all_cols:
        a = "✅" if c in nq_cols else "❌"
        b = "✅" if c in my_cols else "❌"
        print(f"{c:30s} NQ: {a}   YOU: {b}")
    if nq_cols == my_cols:
        print("\n✅ Column names match.")

    if nq_parquet_path and my_parquet_path:
        print("\n" + "=" * 80)
        print("3) PyArrow schema")
        print("=" * 80)
        print("[NQ schema]")
        print(parquet_schema_str(nq_parquet_path))
        print("\n[Your schema]")
        print(parquet_schema_str(my_parquet_path))

    if not nq_samples or not my_samples:
        return

    print("\n" + "=" * 80)
    print("4) First row (side-by-side)")
    print("=" * 80)
    nq0 = nq_samples[0]
    my0 = my_samples[0]
    for k in sorted(nq0.keys()):
        print(f"\n- {k}:")
        print(f"  NQ : {truncate(nq0.get(k), 220)}")
        print(f"  YOU: {truncate(my0.get(k), 220)}")

    print("\n" + "=" * 80)
    print("5) Prompt structure check  ← KEY: verify few-shot example is present")
    print("=" * 80)

    issues: List[str] = []
    required_cols = [
        "id", "question", "golden_answers", "data_source",
        "prompt", "ability", "reward_model", "extra_info", "metadata",
    ]
    for c in required_cols:
        if c not in my0:
            issues.append(f"Missing column: {c}")

    if "prompt" in my0:
        p = my0["prompt"]
        if not _is_listlike(p) or len(p) == 0 or not isinstance(p[0], dict):
            issues.append("prompt should be list-like of dicts")
        else:
            all_content = " ".join(str(m.get("content", "")) for m in p if isinstance(m, dict))
            # Check few-shot markers
            for tag in ["<think>", "</think>", "<answer>", "</answer>"]:
                if tag not in all_content:
                    issues.append(f"❌ prompt content missing few-shot marker: {tag} — model won't know to generate it!")
            for kw in ["Here is an example", "Now answer"]:
                if kw not in all_content:
                    issues.append(f"❌ few-shot phrase missing: '{kw}' — add few-shot example to user message!")
            if "<search>" not in all_content:
                issues.append("❌ <search> tag missing from few-shot example — model won't trigger search!")
            roles = [m.get("role") for m in p if isinstance(m, dict)]
            if "system" not in roles:
                issues.append("prompt missing 'system' role")
            if "user" not in roles:
                issues.append("prompt missing 'user' role")

    if "reward_model" in my0:
        rm = my0["reward_model"]
        if isinstance(rm, dict):
            gt = rm.get("ground_truth", {})
            tgt = gt.get("target") if isinstance(gt, dict) else None
            if not _is_listlike(tgt):
                issues.append("reward_model.ground_truth.target should be list-like")

    if issues:
        print(f"⚠️  Found {len(issues)} issue(s):")
        for i, msg in enumerate(issues, 1):
            print(f"  {i}. {msg}")
    else:
        print("✅ All checks passed — few-shot markers present, model will generate <think>/<search>/<answer>.")


# -------------------------
# CLI entry points
# -------------------------

def cmd_convert(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    prompt_profile = args.prompt_profile

    if args.append_prompt_to_output_dir:
        base = args.output_dir.rstrip("/")
        suffix = re.sub(r"[^a-zA-Z0-9_\-]+", "_", prompt_profile)
        args.output_dir = f"{base}__{suffix}"

    os.makedirs(args.output_dir, exist_ok=True)

    difficulty_filter: Optional[set] = None
    if args.difficulty:
        difficulty_filter = {d.strip() for d in args.difficulty.split(",") if d.strip()}

    def filter_and_dedup(examples: List[QAExample], label: str,
                         exclude_keys: Optional[set] = None) -> List[QAExample]:
        """过滤 + 去重 + 排除已在 test 里的问题"""
        before = len(examples)
        if args.min_answer_len > 0:
            examples = [e for e in examples if len(e.answer.split()) >= args.min_answer_len]
        for e in examples:
            e.extra["difficulty"] = estimate_difficulty(e.question)
        if difficulty_filter:
            examples = [e for e in examples if e.extra.get("difficulty") in difficulty_filter]
        seen: set = set()
        deduped: List[QAExample] = []
        for e in examples:
            key = e.question.strip()[:200]
            if key in seen:
                continue
            if exclude_keys and key in exclude_keys:
                continue  # 这条在 test 里，不能放进 train
            seen.add(key)
            deduped.append(e)
        print(f"{label}: {before} -> {len(deduped)} after filters/dedup/exclusion")
        return deduped

    # =========================================================
    # 模式选择
    #
    # --all_file 模式（新）：所有文件共用一个大池子
    #   1. 每个文件各抽 test_sample_n 条 → test
    #   2. 每个文件剩余的 → train
    #   3. train 里绝对没有 test 里出现过的问题
    #
    # 兼容旧模式：--train_file / --test_file 分开指定
    # =========================================================
    from collections import Counter

    all_files = args.all_file  # 新参数：所有文件放一起处理

    if all_files:
        # ── 新模式：pool-split ─────────────────────────────────────────
        n = args.test_sample_n  # 每个文件抽多少条作为 test（0=不抽，全部做 train）
        test_ex:  List[QAExample] = []
        train_ex: List[QAExample] = []

        for f in all_files:
            ex = detect_and_load(f, args.mode)
            src = _data_source_from_path(f)

            if n > 0 and len(ex) > n:
                # 随机打散后，前 n 条做 test，其余做 train
                indices = list(range(len(ex)))
                random.shuffle(indices)
                test_idx  = set(indices[:n])
                test_part  = [ex[i] for i in indices[:n]]
                train_part = [ex[i] for i in indices[n:]]
                print(f"[{src}] {f}")
                print(f"  total={len(ex)}  →  test={len(test_part)}, train={len(train_part)}")
            else:
                # 数量不够就全部做 train，不抽 test（或 n=0）
                test_part  = []
                train_part = ex
                print(f"[{src}] {f}")
                print(f"  total={len(ex)}  →  test=0 (too few to sample), train={len(train_part)}")

            test_ex.extend(test_part)
            train_ex.extend(train_part)

        # 先处理 test（去重），记录 test 问题的 key 集合
        test_ex = filter_and_dedup(test_ex, "test")
        test_keys = {e.question.strip()[:200] for e in test_ex}

        # 再处理 train，排除所有 test 里见过的问题
        train_ex = filter_and_dedup(train_ex, "train", exclude_keys=test_keys)

    else:
        # ── 旧模式：--train_file / --test_file 分开指定（向后兼容）────
        train_ex = []
        for f in args.train_file:
            ex = detect_and_load(f, args.mode)
            src = _data_source_from_path(f)
            print(f"Loaded train: {f} ({src}) -> {len(ex)} examples")
            train_ex.extend(ex)

        test_ex = []
        for f in (args.val_file + args.test_file):
            ex = detect_and_load(f, args.mode)
            src = _data_source_from_path(f)
            print(f"Loaded test : {f} ({src}) -> {len(ex)} examples")
            test_ex.extend(ex)

        test_ex  = filter_and_dedup(test_ex, "test")
        test_keys = {e.question.strip()[:200] for e in test_ex}
        train_ex = filter_and_dedup(train_ex, "train", exclude_keys=test_keys)

    # 打印最终分布
    print(f"\n{'='*50}")
    print(f"Final  TRAIN: {len(train_ex)} examples")
    print(f"Final  TEST : {len(test_ex)} examples")
    if test_ex:
        src_counts = Counter(e.extra.get("data_source", "unknown") for e in test_ex)
        print(f"Test source distribution : {dict(src_counts)}")
    if train_ex:
        src_counts = Counter(e.extra.get("data_source", "unknown") for e in train_ex)
        print(f"Train source distribution: {dict(src_counts)}")
    print(f"{'='*50}\n")

    # 转换 & 保存
    train_rows = convert_to_nq_rows(train_ex, "train", args.data_source, prompt_profile, args.ability)
    test_rows  = convert_to_nq_rows(test_ex,  "test",  args.data_source, prompt_profile, args.ability)

    out_train = os.path.join(args.output_dir, "train.parquet")
    out_test  = os.path.join(args.output_dir, "test.parquet")

    try:
        save_parquet(train_rows, out_train)
        save_parquet(test_rows,  out_test)
        print(f"Saved: {out_train} ({len(train_rows)} rows)")
        print(f"Saved: {out_test}  ({len(test_rows)} rows)")
    except Exception as e:
        print(f"Parquet save failed ({e}); falling back to JSONL.")
        save_jsonl(train_rows, os.path.join(args.output_dir, "train.jsonl"))
        save_jsonl(test_rows,  os.path.join(args.output_dir, "test.jsonl"))

    if args.build_corpus:
        all_inputs = all_files if all_files else (args.train_file + args.val_file + args.test_file)
        jsonl_inputs = [p for p in all_inputs if p.endswith(".jsonl")]
        if jsonl_inputs:
            corpus_out = args.corpus_output or os.path.join(args.output_dir, "corpus.jsonl")
            build_corpus_from_rag_jsonl(jsonl_inputs, corpus_out)
            print(f"Saved corpus: {corpus_out}")
        else:
            print("No JSONL inputs detected; corpus build skipped.")


def cmd_compare(args: argparse.Namespace) -> None:
    if args.nq_parquet:
        nq_path = args.nq_parquet
    else:
        nq_path = download_nq_train_parquet(cache_dir=args.hf_cache_dir)

    nq_samples, nq_total = read_parquet_samples(nq_path, n=args.n_samples)

    if args.my_parquet:
        my_samples, my_total = read_parquet_samples(args.my_parquet, n=args.n_samples)
        my_label = "YOUR parquet"
        my_parquet_path = args.my_parquet
    else:
        my_samples, my_total = read_jsonl_samples(args.my_jsonl, n=args.n_samples)
        my_label = "YOUR JSONL"
        my_parquet_path = None

    compare_against_nq(
        my_samples=my_samples,
        my_total=my_total,
        my_label=my_label,
        nq_samples=nq_samples,
        nq_total=nq_total,
        my_parquet_path=my_parquet_path,
        nq_parquet_path=nq_path,
    )


def cmd_convert_and_compare(args: argparse.Namespace) -> None:
    cmd_convert(args)
    my_train = os.path.join(args.output_dir, "train.parquet")
    if not os.path.exists(my_train):
        print("Conversion did not produce train.parquet; cannot compare.")
        return
    compare_args = argparse.Namespace(
        nq_parquet=args.nq_parquet,
        hf_cache_dir=args.hf_cache_dir,
        my_parquet=my_train,
        my_jsonl=None,
        n_samples=args.n_samples,
    )
    cmd_compare(compare_args)


# -------------------------
# Quick sanity check helper
# -------------------------

def cmd_show_prompt(args: argparse.Namespace) -> None:
    """Print the rendered prompt for a sample question — useful for spot-checking."""
    question = args.question or "What are the main causes of hypertension?"
    prompt = make_prompt(question, args.prompt_profile)
    print(f"\n=== Rendered prompt (profile={args.prompt_profile}) ===\n")
    for msg in prompt:
        print(f"[{msg['role'].upper()}]")
        print(msg["content"])
        print()


# -------------------------
# CLI
# -------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Search-R1 dataset tool: convert / compare / show-prompt",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_common_convert_flags(p: argparse.ArgumentParser) -> None:
        p.add_argument("--mode", choices=["jsonl", "raw_json", "loader", "auto"], default="auto")
        # ── 新模式：所有文件统一 pool-split ──────────────────────────────
        p.add_argument(
            "--all_file", nargs="+", default=[],
            help=(
                "【推荐】所有数据文件（不区分 train/test）。\n"
                "从每个文件抽 --test_sample_n 条做 test，剩余做 train，两者无重叠。\n"
                "例：--all_file csiro.jsonl liveqa.jsonl pubmedqa.jsonl"
            ),
        )
        # ── 旧模式（向后兼容）────────────────────────────────────────────
        p.add_argument("--train_file", nargs="+", default=[],
                       help="(旧模式) 仅做 train 的文件。")
        p.add_argument("--val_file",   nargs="+", default=[],
                       help="(旧模式) 仅做 test/val 的文件。")
        p.add_argument("--test_file",  nargs="+", default=[],
                       help="(旧模式) 仅做 test 的文件。")
        p.add_argument("--output_dir", default="./searchr1_data")
        p.add_argument("--data_source", default="medical_ragchecker",
                       help="全局默认 data_source（能从文件名解析时会被自动覆盖）。")
        p.add_argument("--difficulty", default=None)
        p.add_argument("--min_answer_len", type=int, default=1)
        p.add_argument("--ability", default="medical-reasoning")
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--build_corpus", action="store_true")
        p.add_argument("--corpus_output", default=None)
        p.add_argument(
            "--prompt_profile",
            choices=["searchr1", "medical", "medical_checker"],
            default="medical",
            help=(
                "searchr1        — general NQ-style few-shot (think+search+answer)\n"
                "medical         — medical few-shot (think+search+answer)\n"
                "medical_checker — medical few-shot + verify step (think+search+check+answer)"
            ),
        )
        p.add_argument("--append_prompt_to_output_dir", action="store_true")
        p.add_argument("--nq_parquet", default=None)
        p.add_argument("--hf_cache_dir", default=None)
        p.add_argument("--n_samples", type=int, default=5)
        p.add_argument(
            "--test_sample_n", type=int, default=30,
            help=(
                "【all_file 模式】每个文件各抽多少条做 test。\n"
                "0 = 全部做 train，不生成 test。默认 30。"
            ),
        )

    p_convert = sub.add_parser("convert", help="Convert to Search-R1 parquet with few-shot prompts.")
    add_common_convert_flags(p_convert)

    p_compare = sub.add_parser("compare", help="Compare your dataset against NQ reference.")
    p_compare.add_argument("--nq_parquet", default=None)
    p_compare.add_argument("--hf_cache_dir", default=None)
    p_compare.add_argument("--my_parquet", default=None)
    p_compare.add_argument("--my_jsonl", default=None)
    p_compare.add_argument("--n_samples", type=int, default=5)

    p_cac = sub.add_parser("convert_and_compare", help="Convert then compare.")
    add_common_convert_flags(p_cac)

    # New: show_prompt — spot-check rendered prompt without converting a dataset
    p_show = sub.add_parser("show_prompt", help="Print rendered prompt for a sample question.")
    p_show.add_argument(
        "--prompt_profile",
        choices=["searchr1", "medical", "medical_checker"],
        default="medical",
    )
    p_show.add_argument("--question", default=None, help="Question text to render (optional).")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.cmd == "convert":
        cmd_convert(args)
    elif args.cmd == "compare":
        if not args.my_parquet and not args.my_jsonl:
            raise SystemExit("Please provide --my_parquet or --my_jsonl.")
        cmd_compare(args)
    elif args.cmd == "convert_and_compare":
        cmd_convert_and_compare(args)
    elif args.cmd == "show_prompt":
        cmd_show_prompt(args)
    else:
        raise SystemExit(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()

# import argparse
# import json
# import os
# import random
# import re
# from dataclasses import dataclass
# from typing import Any, Dict, Iterable, List, Optional, Tuple


# # =============================================================================
# # Prompt Templates  ← 核心修改区域
# # =============================================================================
# #
# # 关键设计：必须在 user 消息里内嵌 few-shot example，
# # 让模型看到 <think>/<search>/<answer> 的示范格式后才会模仿。
# # 没有 few-shot example → 模型不知道要生成这些 tag。
# #
# # 对应原始 test_fewshot.parquet 的格式：
# #   system: 规则说明
# #   user:   示例 + "Now answer this question:\nQuestion: {question}"
# # =============================================================================

# # ---------- System prompts (规则说明，不含示例) ----------

# _SYS_SEARCHR1 = (
#     "Answer the given question. You must conduct reasoning inside "
#     "<think> and </think> first every time you get new information. "
#     "After reasoning, if you find you lack some knowledge, you can call a search engine "
#     "by <search> query </search> and it will return the top searched results between "
#     "<information> and </information>. You can search as many times as you want. "
#     "If you find no further external knowledge needed, you can directly provide the answer "
#     "inside <answer> and </answer>, without detailed illustrations. "
#     "For example, <answer> Beijing </answer>."
# )

# _SYS_MEDICAL = (
#     "Answer the given medical question. You must conduct reasoning inside "
#     "<think> and </think> first every time you get new information. "
#     "After reasoning, if you find you lack some knowledge, you can call a search engine "
#     "by <search> query </search> and it will return the top searched results between "
#     "<information> and </information>. You can search as many times as you want. "
#     "If you find no further external knowledge needed, you can directly provide the answer "
#     "inside <answer> and </answer>, without detailed illustrations. "
#     "For example, <answer> Aspirin is used to reduce fever and relieve pain </answer>."
# )

# _SYS_MEDICAL_CHECKER = (
#     "Answer the given medical question. You must conduct reasoning inside "
#     "<think> and </think> first every time you get new information. "
#     "After reasoning, if you find you lack some knowledge, you can call a search engine "
#     "by <search> query </search> and it will return the top searched results between "
#     "<information> and </information>. "
#     "Before giving your final answer, you MUST verify it by using <check> your answer </check>; "
#     "the verification result will appear between <information> and </information>. "
#     "If the verification contradicts your answer, revise and check again. "
#     "Finally provide your answer inside <answer> and </answer>."
# )

# SYSTEM_INSTRUCTIONS: Dict[str, str] = {
#     "searchr1":        _SYS_SEARCHR1,
#     "medical":         _SYS_MEDICAL,
#     "medical_checker": _SYS_MEDICAL_CHECKER,
# }

# # ---------- Few-shot user templates ----------
# # {question} は実際の質問に置換される

# _FEWSHOT_SEARCHR1 = """\
# Here is an example:
# Question: What is the population of Tokyo?
# Assistant: <think>I need to find the current population of Tokyo. Let me search for this information.</think>
# <search>Tokyo population 2024</search>
# <information>Tokyo is the capital of Japan with a population of approximately 13.96 million in the city proper, and about 37-38 million in the greater metropolitan area.</information>
# <think>I found the answer. Tokyo's population is about 13.96 million in the city and 37-38 million in the greater area.</think>
# <answer>approximately 13.96 million (city proper) or 37-38 million (greater metropolitan area)</answer>

# Now answer this question:
# Question: {question}"""

# _FEWSHOT_MEDICAL = """\
# Here is an example:
# Question: What is the first-line pharmacological treatment for type 2 diabetes?
# Assistant: <think>I need to recall the standard first-line pharmacological treatment for type 2 diabetes. This is a well-established clinical guideline. Let me search to confirm current recommendations.</think>
# <search>first-line pharmacological treatment type 2 diabetes guidelines 2024</search>
# <information>According to the American Diabetes Association (ADA) and most international guidelines, metformin remains the preferred initial pharmacological agent for type 2 diabetes management when not contraindicated, due to its efficacy, safety profile, low cost, and potential cardiovascular benefits.</information>
# <think>The search confirms that metformin is the first-line pharmacological treatment for type 2 diabetes unless contraindicated (e.g., severe renal impairment, eGFR < 30).</think>
# <answer>Metformin is the first-line pharmacological treatment for type 2 diabetes, unless contraindicated.</answer>

# Now answer this medical question:
# Question: {question}"""

# _FEWSHOT_MEDICAL_CHECKER = """\
# Here is an example:
# Question: What is the first-line pharmacological treatment for type 2 diabetes?
# Assistant: <think>I need to find the standard first-line treatment for type 2 diabetes. Let me search current guidelines.</think>
# <search>first-line pharmacological treatment type 2 diabetes ADA guidelines</search>
# <information>Metformin is the preferred initial pharmacological agent for type 2 diabetes management according to ADA and most international guidelines, unless contraindicated.</information>
# <think>The evidence supports metformin. Before I finalize, let me verify this answer.</think>
# <check>Metformin is the first-line pharmacological treatment for type 2 diabetes</check>
# <information>Verification result: Supported. ADA Standards of Medical Care in Diabetes consistently recommend metformin as first-line therapy unless contraindicated due to renal impairment (eGFR < 30) or other factors.</information>
# <think>The verification confirms the answer is correct and well-supported by current guidelines.</think>
# <answer>Metformin is the first-line pharmacological treatment for type 2 diabetes, unless contraindicated.</answer>

# Now answer this medical question:
# Question: {question}"""

# FEWSHOT_TEMPLATES: Dict[str, str] = {
#     "searchr1":        _FEWSHOT_SEARCHR1,
#     "medical":         _FEWSHOT_MEDICAL,
#     "medical_checker": _FEWSHOT_MEDICAL_CHECKER,
# }


# # =============================================================================
# # make_prompt  ← 核心修改：user 消息内嵌 few-shot example
# # =============================================================================

# def make_prompt(question: str, prompt_profile: str) -> List[Dict[str, str]]:
#     """
#     构造 Search-R1 兼容的 prompt（system + user 两条消息）。

#     user 消息 = few-shot 示例 + 真正的问题
#     → 模型看到示例后会模仿 <think>/<search>/[<check>]/<answer> 格式

#     对应原始 test_fewshot.parquet 的结构（已验证）：
#       [{'role': 'system', 'content': ...规则说明...},
#        {'role': 'user',   'content': ...few-shot示例...\\nNow answer:\\nQuestion: ...}]
#     """
#     if prompt_profile not in SYSTEM_INSTRUCTIONS:
#         raise ValueError(
#             f"Unknown prompt_profile: '{prompt_profile}'. "
#             f"Choose from: {list(SYSTEM_INSTRUCTIONS.keys())}"
#         )

#     system_content = SYSTEM_INSTRUCTIONS[prompt_profile]
#     user_content   = FEWSHOT_TEMPLATES[prompt_profile].format(question=question)

#     return [
#         {"role": "system", "content": system_content},
#         {"role": "user",   "content": user_content},
#     ]


# # =============================================================================
# # 以下は元のコードをそのまま保持
# # =============================================================================

# # -------------------------
# # JSON-safe pretty printing
# # -------------------------

# def _to_jsonable(x: Any) -> Any:
#     try:
#         import numpy as np
#         if isinstance(x, np.ndarray):
#             return x.tolist()
#         if isinstance(x, np.generic):
#             return x.item()
#     except Exception:
#         pass
#     if isinstance(x, dict):
#         return {str(k): _to_jsonable(v) for k, v in x.items()}
#     if isinstance(x, (list, tuple)):
#         return [_to_jsonable(v) for v in x]
#     if isinstance(x, (bytes, bytearray)):
#         return x.decode("utf-8", errors="replace")
#     return x


# def truncate(val: Any, max_len: int = 140) -> str:
#     try:
#         s = json.dumps(_to_jsonable(val), ensure_ascii=False)
#     except Exception:
#         s = str(val)
#     return (s[: max_len - 3] + "...") if len(s) > max_len else s


# def _is_listlike(x: Any) -> bool:
#     try:
#         import numpy as np
#         return isinstance(x, (list, tuple, np.ndarray))
#     except Exception:
#         return isinstance(x, (list, tuple))


# # -------------------------
# # Input loading / adapters
# # -------------------------

# @dataclass
# class QAExample:
#     question: str
#     answer: str
#     extra: Dict[str, Any]


# def load_rag_jsonl(path: str) -> List[QAExample]:
#     out: List[QAExample] = []
#     with open(path, "r", encoding="utf-8") as f:
#         for i, line in enumerate(f):
#             line = line.strip()
#             if not line:
#                 continue
#             try:
#                 item = json.loads(line)
#             except Exception:
#                 continue
#             q = (item.get("query") or item.get("question") or "").strip()
#             a = (item.get("gt_answer") or item.get("answer") or item.get("final_answer") or "").strip()
#             if not q or not a:
#                 continue
#             extra = {
#                 "source_path": path,
#                 "line": i,
#                 "query_id": item.get("query_id"),
#                 "retrieved_context": item.get("retrieved_context", []),
#             }
#             out.append(QAExample(question=q, answer=a, extra=extra))
#     return out


# def load_raw_json_chunks(path: str) -> List[QAExample]:
#     with open(path, "r", encoding="utf-8") as f:
#         data = json.load(f)
#     if isinstance(data, dict) and any(str(k).isdigit() for k in data.keys()):
#         data = list(data.values())
#     if not isinstance(data, list):
#         data = [data]
#     out: List[QAExample] = []
#     for item in data:
#         meta = item.get("meta", {}) if isinstance(item, dict) else {}
#         q = (meta.get("question") or item.get("question") or "").strip()
#         a = (meta.get("answer") or item.get("answer") or item.get("gt_answer") or "").strip()
#         if not q or not a:
#             continue
#         out.append(QAExample(question=q, answer=a, extra={"source_path": path}))
#     return out


# def load_loader_output(path: str) -> List[QAExample]:
#     txt = open(path, "r", encoding="utf-8").read().strip()
#     items: List[Dict[str, Any]] = []
#     try:
#         obj = json.loads(txt)
#         if isinstance(obj, list):
#             items = obj
#         elif isinstance(obj, dict):
#             items = list(obj.values()) if "question" not in obj else [obj]
#     except Exception:
#         for line in txt.splitlines():
#             line = line.strip()
#             if not line:
#                 continue
#             try:
#                 items.append(json.loads(line))
#             except Exception:
#                 continue
#     out: List[QAExample] = []
#     for it in items:
#         if not isinstance(it, dict):
#             continue
#         q = (it.get("question") or it.get("query") or it.get("prompt") or "").strip()
#         a = (it.get("answer") or it.get("final_answer") or it.get("gt_answer") or it.get("long_answer") or "").strip()
#         if not q or not a:
#             continue
#         out.append(QAExample(question=q, answer=a, extra={"source_path": path}))
#     return out


# def detect_and_load(path: str, mode: str) -> List[QAExample]:
#     if mode == "jsonl":
#         return load_rag_jsonl(path)
#     if mode == "raw_json":
#         return load_raw_json_chunks(path)
#     if mode == "loader":
#         return load_loader_output(path)
#     if path.endswith(".jsonl"):
#         return load_rag_jsonl(path)
#     return load_loader_output(path)


# # -------------------------
# # Difficulty heuristics
# # -------------------------

# def estimate_difficulty(question: str) -> str:
#     words = question.split()
#     q_len = len(words)
#     q_lower = question.lower()
#     if q_len < 25:
#         length_score = 0
#     elif q_len < 60:
#         length_score = 1
#     elif q_len < 120:
#         length_score = 2
#     else:
#         length_score = 3
#     reasoning_keywords = [
#         "most likely", "best explanation", "most appropriate",
#         "next step", "differential diagnosis", "which of the following",
#         "mechanism", "pathophysiology", "contraindicated",
#         "first-line", "gold standard", "why", "how does",
#         "relationship", "explain",
#     ]
#     keyword_score = sum(1 for k in reasoning_keywords if k in q_lower)
#     vignette_score = 1 if ("\n" in question or q_len >= 80) else 0
#     total = length_score + keyword_score + vignette_score
#     if total <= 1:
#         return "easy"
#     if total <= 4:
#         return "medium"
#     return "hard"


# # -------------------------
# # Conversion to NQ schema
# # -------------------------

# def convert_to_nq_rows(
#     examples: List[QAExample],
#     split: str,
#     data_source: str,
#     prompt_profile: str,
#     ability: str,
# ) -> List[Dict[str, Any]]:
#     rows: List[Dict[str, Any]] = []
#     for idx, ex in enumerate(examples):
#         rows.append(
#             {
#                 "id": f"{split}_{idx}",
#                 "question": ex.question,
#                 "golden_answers": [ex.answer],
#                 "data_source": data_source,
#                 "prompt": make_prompt(ex.question, prompt_profile),
#                 "ability": ability,
#                 "reward_model": {
#                     "ground_truth": {"target": [ex.answer]},
#                     "style": "rule",
#                 },
#                 "extra_info": {
#                     "index": idx,
#                     "split": split,
#                     "data_source": data_source,
#                     "prompt_profile": prompt_profile,
#                 },
#                 "metadata": None,
#                 "agent_name": "tool_agent",
#             }
#         )
#     return rows


# # -------------------------
# # Saving and corpus building
# # -------------------------

# def save_parquet(rows: List[Dict[str, Any]], out_path: str) -> None:
#     import pandas as pd
#     os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
#     df = pd.DataFrame(rows)
#     df.to_parquet(out_path, engine="pyarrow", index=False)


# def save_jsonl(rows: List[Dict[str, Any]], out_path: str) -> None:
#     os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
#     with open(out_path, "w", encoding="utf-8") as f:
#         for r in rows:
#             f.write(json.dumps(_to_jsonable(r), ensure_ascii=False) + "\n")


# def build_corpus_from_rag_jsonl(jsonl_files: List[str], out_path: str) -> None:
#     seen: set = set()
#     passages: List[Dict[str, str]] = []
#     for fpath in jsonl_files:
#         with open(fpath, "r", encoding="utf-8") as f:
#             for line in f:
#                 line = line.strip()
#                 if not line:
#                     continue
#                 try:
#                     item = json.loads(line)
#                 except Exception:
#                     continue
#                 for ctx in item.get("retrieved_context", []) or []:
#                     text = (ctx.get("text") or "").strip()
#                     if not text or text in seen:
#                         continue
#                     seen.add(text)
#                     title = (ctx.get("title") or "").strip()
#                     doc_id = (ctx.get("doc_id") or "").strip()
#                     contents = f"{title}\n{text}".strip() if title else text
#                     passages.append({"id": doc_id or str(len(passages)), "contents": contents})
#     os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
#     with open(out_path, "w", encoding="utf-8") as f:
#         for p in passages:
#             f.write(json.dumps(p, ensure_ascii=False) + "\n")


# # -------------------------
# # NQ reference + comparison
# # -------------------------

# def download_nq_train_parquet(cache_dir: Optional[str] = None) -> str:
#     from huggingface_hub import hf_hub_download
#     return hf_hub_download(
#         repo_id="PeterJinGo/nq_hotpotqa_train",
#         filename="train.parquet",
#         repo_type="dataset",
#         cache_dir=cache_dir,
#     )


# def read_parquet_samples(path: str, n: int = 5) -> Tuple[List[Dict[str, Any]], int]:
#     import pandas as pd
#     import pyarrow.parquet as pq
#     df = pd.read_parquet(path)
#     samples = [df.iloc[i].to_dict() for i in range(min(n, len(df)))]
#     total = pq.ParquetFile(path).metadata.num_rows
#     return samples, int(total)


# def read_jsonl_samples(path: str, n: int = 5) -> Tuple[List[Dict[str, Any]], int]:
#     samples: List[Dict[str, Any]] = []
#     total = 0
#     with open(path, "r", encoding="utf-8") as f:
#         for line in f:
#             total += 1
#             if len(samples) < n:
#                 try:
#                     samples.append(json.loads(line))
#                 except Exception:
#                     pass
#     return samples, total


# def parquet_schema_str(path: str) -> str:
#     import pyarrow.parquet as pq
#     return str(pq.read_schema(path))


# def compare_against_nq(
#     my_samples: List[Dict[str, Any]],
#     my_total: int,
#     my_label: str,
#     nq_samples: List[Dict[str, Any]],
#     nq_total: int,
#     my_parquet_path: Optional[str] = None,
#     nq_parquet_path: Optional[str] = None,
# ) -> None:
#     print("\n" + "=" * 80)
#     print("1) Basic stats")
#     print("=" * 80)
#     print(f"{'':30s} {'NQ (native)':>16s}   {my_label:>16s}")
#     print(f"{'Total rows':30s} {nq_total:>16d}   {my_total:>16d}")
#     print(f"{'Num columns (sample)':30s} {len(nq_samples[0]) if nq_samples else 0:>16d}   {len(my_samples[0]) if my_samples else 0:>16d}")

#     print("\n" + "=" * 80)
#     print("2) Column names")
#     print("=" * 80)
#     nq_cols = set(nq_samples[0].keys()) if nq_samples else set()
#     my_cols = set(my_samples[0].keys()) if my_samples else set()
#     all_cols = sorted(nq_cols | my_cols)
#     for c in all_cols:
#         a = "✅" if c in nq_cols else "❌"
#         b = "✅" if c in my_cols else "❌"
#         print(f"{c:30s} NQ: {a}   YOU: {b}")
#     if nq_cols == my_cols:
#         print("\n✅ Column names match.")

#     if nq_parquet_path and my_parquet_path:
#         print("\n" + "=" * 80)
#         print("3) PyArrow schema")
#         print("=" * 80)
#         print("[NQ schema]")
#         print(parquet_schema_str(nq_parquet_path))
#         print("\n[Your schema]")
#         print(parquet_schema_str(my_parquet_path))

#     if not nq_samples or not my_samples:
#         return

#     print("\n" + "=" * 80)
#     print("4) First row (side-by-side)")
#     print("=" * 80)
#     nq0 = nq_samples[0]
#     my0 = my_samples[0]
#     for k in sorted(nq0.keys()):
#         print(f"\n- {k}:")
#         print(f"  NQ : {truncate(nq0.get(k), 220)}")
#         print(f"  YOU: {truncate(my0.get(k), 220)}")

#     print("\n" + "=" * 80)
#     print("5) Prompt structure check  ← KEY: verify few-shot example is present")
#     print("=" * 80)

#     issues: List[str] = []
#     required_cols = [
#         "id", "question", "golden_answers", "data_source",
#         "prompt", "ability", "reward_model", "extra_info", "metadata",
#     ]
#     for c in required_cols:
#         if c not in my0:
#             issues.append(f"Missing column: {c}")

#     if "prompt" in my0:
#         p = my0["prompt"]
#         if not _is_listlike(p) or len(p) == 0 or not isinstance(p[0], dict):
#             issues.append("prompt should be list-like of dicts")
#         else:
#             all_content = " ".join(str(m.get("content", "")) for m in p if isinstance(m, dict))
#             # Check few-shot markers
#             for tag in ["<think>", "</think>", "<answer>", "</answer>"]:
#                 if tag not in all_content:
#                     issues.append(f"❌ prompt content missing few-shot marker: {tag} — model won't know to generate it!")
#             for kw in ["Here is an example", "Now answer"]:
#                 if kw not in all_content:
#                     issues.append(f"❌ few-shot phrase missing: '{kw}' — add few-shot example to user message!")
#             if "<search>" not in all_content:
#                 issues.append("❌ <search> tag missing from few-shot example — model won't trigger search!")
#             roles = [m.get("role") for m in p if isinstance(m, dict)]
#             if "system" not in roles:
#                 issues.append("prompt missing 'system' role")
#             if "user" not in roles:
#                 issues.append("prompt missing 'user' role")

#     if "reward_model" in my0:
#         rm = my0["reward_model"]
#         if isinstance(rm, dict):
#             gt = rm.get("ground_truth", {})
#             tgt = gt.get("target") if isinstance(gt, dict) else None
#             if not _is_listlike(tgt):
#                 issues.append("reward_model.ground_truth.target should be list-like")

#     if issues:
#         print(f"⚠️  Found {len(issues)} issue(s):")
#         for i, msg in enumerate(issues, 1):
#             print(f"  {i}. {msg}")
#     else:
#         print("✅ All checks passed — few-shot markers present, model will generate <think>/<search>/<answer>.")


# # -------------------------
# # CLI entry points
# # -------------------------

# def cmd_convert(args: argparse.Namespace) -> None:
#     random.seed(args.seed)
#     prompt_profile = args.prompt_profile

#     if args.append_prompt_to_output_dir:
#         base = args.output_dir.rstrip("/")
#         suffix = re.sub(r"[^a-zA-Z0-9_\-]+", "_", prompt_profile)
#         args.output_dir = f"{base}__{suffix}"

#     os.makedirs(args.output_dir, exist_ok=True)

#     difficulty_filter: Optional[set] = None
#     if args.difficulty:
#         difficulty_filter = {d.strip() for d in args.difficulty.split(",") if d.strip()}

#     train_ex: List[QAExample] = []
#     test_ex: List[QAExample] = []

#     for f in args.train_file:
#         ex = detect_and_load(f, args.mode)
#         print(f"Loaded train: {f} -> {len(ex)} examples")
#         train_ex.extend(ex)

#     for f in (args.val_file + args.test_file):
#         ex = detect_and_load(f, args.mode)
#         print(f"Loaded test : {f} -> {len(ex)} examples")
#         test_ex.extend(ex)

#     def filter_and_dedup(examples: List[QAExample], label: str) -> List[QAExample]:
#         before = len(examples)
#         if args.min_answer_len > 0:
#             examples = [e for e in examples if len(e.answer.split()) >= args.min_answer_len]
#         for e in examples:
#             e.extra["difficulty"] = estimate_difficulty(e.question)
#         if difficulty_filter:
#             examples = [e for e in examples if e.extra.get("difficulty") in difficulty_filter]
#         seen: set = set()
#         deduped: List[QAExample] = []
#         for e in examples:
#             key = e.question.strip()[:200]
#             if key in seen:
#                 continue
#             seen.add(key)
#             deduped.append(e)
#         print(f"{label}: {before} -> {len(deduped)} after filters/dedup")
#         return deduped

#     train_ex = filter_and_dedup(train_ex, "train")
#     test_ex  = filter_and_dedup(test_ex, "test")

#     train_rows = convert_to_nq_rows(train_ex, "train", args.data_source, prompt_profile, args.ability)
#     test_rows  = convert_to_nq_rows(test_ex,  "test",  args.data_source, prompt_profile, args.ability)

#     out_train = os.path.join(args.output_dir, "train.parquet")
#     out_test  = os.path.join(args.output_dir, "test.parquet")

#     try:
#         save_parquet(train_rows, out_train)
#         save_parquet(test_rows,  out_test)
#         print(f"Saved: {out_train} ({len(train_rows)} rows)")
#         print(f"Saved: {out_test}  ({len(test_rows)} rows)")
#     except Exception as e:
#         print(f"Parquet save failed ({e}); falling back to JSONL.")
#         save_jsonl(train_rows, os.path.join(args.output_dir, "train.jsonl"))
#         save_jsonl(test_rows,  os.path.join(args.output_dir, "test.jsonl"))

#     if args.build_corpus:
#         jsonl_inputs = [p for p in (args.train_file + args.val_file + args.test_file) if p.endswith(".jsonl")]
#         if jsonl_inputs:
#             corpus_out = args.corpus_output or os.path.join(args.output_dir, "corpus.jsonl")
#             build_corpus_from_rag_jsonl(jsonl_inputs, corpus_out)
#             print(f"Saved corpus: {corpus_out}")
#         else:
#             print("No JSONL inputs detected; corpus build skipped.")


# def cmd_compare(args: argparse.Namespace) -> None:
#     if args.nq_parquet:
#         nq_path = args.nq_parquet
#     else:
#         nq_path = download_nq_train_parquet(cache_dir=args.hf_cache_dir)

#     nq_samples, nq_total = read_parquet_samples(nq_path, n=args.n_samples)

#     if args.my_parquet:
#         my_samples, my_total = read_parquet_samples(args.my_parquet, n=args.n_samples)
#         my_label = "YOUR parquet"
#         my_parquet_path = args.my_parquet
#     else:
#         my_samples, my_total = read_jsonl_samples(args.my_jsonl, n=args.n_samples)
#         my_label = "YOUR JSONL"
#         my_parquet_path = None

#     compare_against_nq(
#         my_samples=my_samples,
#         my_total=my_total,
#         my_label=my_label,
#         nq_samples=nq_samples,
#         nq_total=nq_total,
#         my_parquet_path=my_parquet_path,
#         nq_parquet_path=nq_path,
#     )


# def cmd_convert_and_compare(args: argparse.Namespace) -> None:
#     cmd_convert(args)
#     my_train = os.path.join(args.output_dir, "train.parquet")
#     if not os.path.exists(my_train):
#         print("Conversion did not produce train.parquet; cannot compare.")
#         return
#     compare_args = argparse.Namespace(
#         nq_parquet=args.nq_parquet,
#         hf_cache_dir=args.hf_cache_dir,
#         my_parquet=my_train,
#         my_jsonl=None,
#         n_samples=args.n_samples,
#     )
#     cmd_compare(compare_args)


# # -------------------------
# # Quick sanity check helper
# # -------------------------

# def cmd_show_prompt(args: argparse.Namespace) -> None:
#     """Print the rendered prompt for a sample question — useful for spot-checking."""
#     question = args.question or "What are the main causes of hypertension?"
#     prompt = make_prompt(question, args.prompt_profile)
#     print(f"\n=== Rendered prompt (profile={args.prompt_profile}) ===\n")
#     for msg in prompt:
#         print(f"[{msg['role'].upper()}]")
#         print(msg["content"])
#         print()


# # -------------------------
# # CLI
# # -------------------------

# def build_parser() -> argparse.ArgumentParser:
#     parser = argparse.ArgumentParser(
#         description="Search-R1 dataset tool: convert / compare / show-prompt",
#         formatter_class=argparse.RawDescriptionHelpFormatter,
#     )
#     sub = parser.add_subparsers(dest="cmd", required=True)

#     def add_common_convert_flags(p: argparse.ArgumentParser) -> None:
#         p.add_argument("--mode", choices=["jsonl", "raw_json", "loader", "auto"], default="auto")
#         p.add_argument("--train_file", nargs="+", default=[])
#         p.add_argument("--val_file",   nargs="+", default=[])
#         p.add_argument("--test_file",  nargs="+", default=[])
#         p.add_argument("--output_dir", default="./searchr1_data")
#         p.add_argument("--data_source", default="medical_ragchecker")
#         p.add_argument("--difficulty", default=None)
#         p.add_argument("--min_answer_len", type=int, default=1)
#         p.add_argument("--ability", default="medical-reasoning")
#         p.add_argument("--seed", type=int, default=42)
#         p.add_argument("--build_corpus", action="store_true")
#         p.add_argument("--corpus_output", default=None)
#         p.add_argument(
#             "--prompt_profile",
#             choices=["searchr1", "medical", "medical_checker"],
#             default="medical",
#             help=(
#                 "searchr1        — general NQ-style few-shot (think+search+answer)\n"
#                 "medical         — medical few-shot (think+search+answer)\n"
#                 "medical_checker — medical few-shot with verification step (think+search+check+answer)"
#             ),
#         )
#         p.add_argument("--append_prompt_to_output_dir", action="store_true")
#         p.add_argument("--nq_parquet", default=None)
#         p.add_argument("--hf_cache_dir", default=None)
#         p.add_argument("--n_samples", type=int, default=5)

#     p_convert = sub.add_parser("convert", help="Convert to Search-R1 parquet with few-shot prompts.")
#     add_common_convert_flags(p_convert)

#     p_compare = sub.add_parser("compare", help="Compare your dataset against NQ reference.")
#     p_compare.add_argument("--nq_parquet", default=None)
#     p_compare.add_argument("--hf_cache_dir", default=None)
#     p_compare.add_argument("--my_parquet", default=None)
#     p_compare.add_argument("--my_jsonl", default=None)
#     p_compare.add_argument("--n_samples", type=int, default=5)

#     p_cac = sub.add_parser("convert_and_compare", help="Convert then compare.")
#     add_common_convert_flags(p_cac)

#     # New: show_prompt — spot-check rendered prompt without converting a dataset
#     p_show = sub.add_parser("show_prompt", help="Print rendered prompt for a sample question.")
#     p_show.add_argument(
#         "--prompt_profile",
#         choices=["searchr1", "medical", "medical_checker"],
#         default="medical",
#     )
#     p_show.add_argument("--question", default=None, help="Question text to render (optional).")

#     return parser


# def main() -> None:
#     parser = build_parser()
#     args = parser.parse_args()

#     if args.cmd == "convert":
#         cmd_convert(args)
#     elif args.cmd == "compare":
#         if not args.my_parquet and not args.my_jsonl:
#             raise SystemExit("Please provide --my_parquet or --my_jsonl.")
#         cmd_compare(args)
#     elif args.cmd == "convert_and_compare":
#         cmd_convert_and_compare(args)
#     elif args.cmd == "show_prompt":
#         cmd_show_prompt(args)
#     else:
#         raise SystemExit(f"Unknown command: {args.cmd}")


# if __name__ == "__main__":
#     main()
