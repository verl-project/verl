"""
Preprocess the multi-hop QA data generated with QA-gym into RL records.
Prompts are built by randomly wrapping the question and context in different styles.
The raw question is kept separately for reward scoring.
"""

import json
import os
import random

import pyarrow.parquet as pq

IN_QAS = "/capstor/store/cscs/swissai/infra01/datasets/qa_gym_data/eval10_hybrid_multihop_qas.jsonl"
OUT_RL = "/capstor/store/cscs/swissai/infra01/reasoning/data/RL-prod/qa_gym/eval10_hybrid_multihop_rl_pairs.jsonl"
SEED = 42

# 1 for a single sample, None for all
LIMIT = None


def _wrap(text, kind, rng):
    if text is None:
        text = ""
    nl_n = rng.choice([1, 2, 3])
    nl = "\n" * nl_n

    if kind == "question":
        style = rng.choice(["plain", "label_inline", "label_block", "xml_tag", "md_header"])
        if style == "plain":
            return text
        if style == "label_inline":
            prefix = rng.choice(["Question: ", "Q: ", "query: "])
            return prefix + text
        if style == "label_block":
            prefix = rng.choice(["Question", "Q", "Query"])
            return prefix + nl + text
        if style == "xml_tag":
            tag = rng.choice(["question", "query", "prompt"])
            inner_nl = "\n" * max(0, nl_n - 1)
            return "<%s>%s</%s>" % (tag, inner_nl + text + inner_nl, tag)
        if style == "md_header":
            header = rng.choice(["### Question", "## Question", "# Question", "### Q"])
            return header + nl + text
        return text

    style = rng.choice(["plain", "label_block", "xml_tag", "fence3", "fence4", "md_header"])
    if style == "plain":
        return text
    if style == "label_block":
        prefix = rng.choice(["Context", "Document", "Passage"])
        return prefix + nl + text
    if style == "xml_tag":
        tag = rng.choice(["context", "document", "passage"])
        inner_nl = "\n" * max(0, nl_n - 1)
        return "<%s>%s</%s>" % (tag, inner_nl + text + inner_nl, tag)
    if style == "fence3":
        fence = "```"
        return fence + nl + text + nl + fence
    if style == "fence4":
        fence = "````"
        return fence + nl + text + nl + fence
    if style == "md_header":
        header = rng.choice(["### Context", "## Context", "# Context", "### Document", "## Document", "# Document"])
        return header + nl + text
    return text


def build_prompt(question, context, rng):
    q = _wrap(question, "question", rng)
    c = _wrap(context, "context", rng)
    sep = "\n" * rng.choice([1, 2, 2, 3, 4])
    p_qc = 0.75
    return (q + sep + c) if (rng.random() < p_qc) else (c + sep + q)


def load_context_text(parquet_path, source_row, page_start, page_end):
    """
    Reconstruct the exact QA context by slicing metadata.text_by_page_gen and
    joining non-empty pages with '\\n\\n' (this matches context.char_count).
    """
    pf = pq.ParquetFile(parquet_path)
    row_index = 0
    for batch in pf.iter_batches(batch_size=16, columns=["metadata"], use_threads=False):
        if source_row < row_index + batch.num_rows:
            j = source_row - row_index
            meta = batch.column(0)
            pages = meta.field("text_by_page_gen")[j].as_py()
            sliced = pages[page_start - 1 : page_end]
            sliced = [p for p in sliced if p and p.strip()]
            return "\n\n".join(sliced)
        row_index += batch.num_rows
    raise ValueError("source_row out of range for parquet: %s row=%s" % (parquet_path, source_row))


def main():
    rng = random.Random(SEED)
    outdir = os.path.dirname(OUT_RL)
    os.makedirs(outdir, exist_ok=True)

    n_in = 0
    n_out = 0
    with open(IN_QAS, "r", encoding="utf-8") as fin, open(OUT_RL, "w", encoding="utf-8") as fout:
        for line in fin:
            if LIMIT is not None and n_in >= LIMIT:
                break
            line = line.strip()
            if not line:
                continue
            qa = json.loads(line)
            ctx = qa["context"]

            context_text = load_context_text(
                ctx["source_path"],
                int(ctx["source_row"]),
                int(ctx["page_start"]),
                int(ctx["page_end"]),
            )

            question = qa["question"]
            prompt = build_prompt(question, context_text, rng)
            out = {
                "id": qa["qa_id"],
                "question": question,
                "prompt": prompt,
                "answer": qa["answer"],
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")

            n_in += 1
            n_out += 1

    print("wrote", n_out, "records to", OUT_RL)


if __name__ == "__main__":
    main()
