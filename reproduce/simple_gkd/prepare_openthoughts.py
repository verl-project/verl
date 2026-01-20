import json
import gzip
from datasets import load_dataset
from tqdm import tqdm
import re
import pandas as pd
from transformers import AutoTokenizer

def save_math_examples(output_path="math_examples.jsonl.gz"):
    dataset = load_dataset(
        "open-thoughts/OpenThoughts3-1.2M",
        split="train",
        streaming=True,
    )
    
    math_count = 0
    total_count = 0
    
    with gzip.open(output_path, 'wt', encoding='utf-8') as f:
        for example in tqdm(dataset, desc="Processing examples"):
            total_count += 1
            
            if example.get('domain') == 'math':
                json.dump(example, f, ensure_ascii=False)
                f.write('\n')
                math_count += 1
                
                if math_count % 10000 == 0:
                    print(f"Saved {math_count} math examples (total processed: {total_count})")
    return math_count, total_count

def load_math_examples(file_path="math_examples.jsonl.gz"):
    examples = []
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


# ----------------------------
# Regexes
# ----------------------------
# Note: keep as simple as possible; this does NOT handle nested braces inside \boxed{...}.
BOXED_RE = re.compile(r"\\boxed\s*\{([^}]*)\}", flags=re.DOTALL)

# We do NOT use THINK_RE for splitting, because we want to:
# - detect incomplete CoT (<think> without </think>)
# - only keep samples with a complete <think>...</think> block
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


# ----------------------------
# IO
# ----------------------------
def iter_gz_jsonl(path: str):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


# ----------------------------
# Parsing helpers
# ----------------------------
def get_user_and_assistant(example: dict):
    """
    OpenThoughts3 math sample structure:
    example["conversations"] = [{"from":"human","value":...}, {"from":"gpt","value":...}, ...]
    """
    conv = example.get("conversations")
    if not isinstance(conv, list) or not conv:
        return None, None

    user_text = None
    assistant_text = None
    for m in conv:
        if not isinstance(m, dict):
            continue
        frm = m.get("from")
        val = m.get("value")
        if frm == "human" and user_text is None:
            user_text = val
        elif frm == "gpt":
            assistant_text = val  # last assistant turn if multi-turn

    return user_text, assistant_text


def has_complete_think(assistant_text: str) -> bool:
    """
    Keep only examples that have a complete <think>...</think>.
    - If only <think> but no </think>: incomplete CoT -> discard from df, only count.
    - If no <think> at all: discard (per your requirement: df only contains complete think).
    - If only </think> or ordering weird: discard.
    """
    if not assistant_text:
        return False
    i = assistant_text.find(THINK_OPEN)
    j = assistant_text.find(THINK_CLOSE)
    return (i != -1) and (j != -1) and (j > i)


def split_think_answer_complete(assistant_text: str):
    """
    Only call this after has_complete_think() is True.
    - think_text: the content inside <think>...</think>
    - answer_text: everything after the closing </think>
    """
    i = assistant_text.find(THINK_OPEN)
    j = assistant_text.find(THINK_CLOSE)
    think_text = assistant_text[i + len(THINK_OPEN): j].strip()
    answer_text = assistant_text[j + len(THINK_CLOSE):].strip()
    return think_text, answer_text


def extract_boxed_int(text: str):
    """
    Extract the LAST \boxed{...} from `text` and try to parse it as an integer.
    Return (boxed_raw, boxed_int_or_none).
    """
    if not text:
        return None, None

    boxed_all = BOXED_RE.findall(text)
    if not boxed_all:
        return None, None

    boxed = boxed_all[-1].strip()

    # normalize for integer parse
    s = boxed.replace(",", "")
    s = re.sub(r"\s+", "", s)
    s = s.replace("$", "")
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)

    if re.fullmatch(r"[+-]?\d+", s):
        return boxed, int(s)
    return boxed, None


def stats(
    path="math_examples.jsonl.gz",
    tokenizer_name="Qwen/Qwen3-0.6B",
    max_rows=None,
):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    # Counters (all rows, including discarded)
    stats = {
        "total_seen": 0,
        "kept_complete_think": 0,
        "discard_no_think_or_weird": 0,
        "discard_incomplete_think": 0,
    }

    rows = []

    it = iter_gz_jsonl(path)
    for ex in tqdm(it, desc="EDA", total=max_rows if max_rows is not None else None):
        if max_rows is not None and stats["total_seen"] >= max_rows:
            break
        stats["total_seen"] += 1

        user_text, assistant_text = get_user_and_assistant(ex)
        assistant_text = assistant_text or ""

        # classify + discard policy
        has_open = (THINK_OPEN in assistant_text)
        has_close = (THINK_CLOSE in assistant_text)

        if not has_complete_think(assistant_text):
            if has_open and not has_close:
                stats["discard_incomplete_think"] += 1
            else:
                stats["discard_no_think_or_weird"] += 1
            continue  # ✅ discard from final df

        stats["kept_complete_think"] += 1

        think_text, answer_text = split_think_answer_complete(assistant_text)

        think_tokens = len(tokenizer.encode(think_text, add_special_tokens=False)) if think_text else 0
        answer_tokens = len(tokenizer.encode(answer_text, add_special_tokens=False)) if answer_text else 0

        # Boxed might appear in think OR answer; for RL-verifiable you likely want anywhere.
        boxed_raw, boxed_int = extract_boxed_int(assistant_text)

        rl_verifiable = boxed_raw is not None
        is_integer = boxed_int is not None
        integer_and_verifiable = rl_verifiable and is_integer

        rows.append({
            "think_tokens": think_tokens,
            "answer_tokens": answer_tokens,
            "boxed_raw": boxed_raw,
            "boxed_int": boxed_int,
            "rl_verifiable": rl_verifiable,
            "is_integer": is_integer,
            "integer_and_verifiable": integer_and_verifiable,
            "user": user_text,
            "source": ex.get("source"),
            "domain": ex.get("domain"),
        })

    df = pd.DataFrame(rows)

    # ---- Reporting ----
    print("\n=== Data quality ===")
    total = stats["total_seen"]
    kept = stats["kept_complete_think"]
    print(f"total_seen:              {total}")
    print(f"kept_complete_think:     {kept} ({kept/max(total,1):.2%})")
    print(f"discard_incomplete_think:{stats['discard_incomplete_think']} ({stats['discard_incomplete_think']/max(total,1):.2%})")
    print(f"discard_no_think_or_weird:{stats['discard_no_think_or_weird']} ({stats['discard_no_think_or_weird']/max(total,1):.2%})")

    if len(df) == 0:
        print("\n[WARN] df is empty (no complete <think>...</think> examples found in the scanned range).")
        return df

    print("\n=== Summary (on kept df only) ===")
    n = len(df)
    print(f"df_rows: {n}")
    print(f"has_boxed (rl_verifiable): {df['rl_verifiable'].mean():.2%}  ({df['rl_verifiable'].sum()}/{n})")
    print(f"boxed_is_integer:          {df['is_integer'].mean():.2%}  ({df['is_integer'].sum()}/{n})")
    print(f"integer_and_verifiable:   {df['integer_and_verifiable'].mean():.2%}  ({df['integer_and_verifiable'].sum()}/{n})")

    print("\n=== Token stats (on kept df only) ===")
    print(df[["think_tokens", "answer_tokens"]].describe(percentiles=[0.5, 0.9, 0.95, 0.99]))

    return df




# 保存math examples
path = "math_examples.jsonl.gz"
# save_math_examples(path)
df = stats(path, tokenizer_name="Qwen/Qwen3-0.6B", max_rows=None)