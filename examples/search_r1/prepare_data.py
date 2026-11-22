"""Audit local parquet and adapt its questions to the multi-message recipe.

No downloads, no modification of the source data. Run from the verl root.
"""

import argparse
import copy
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .protocol import SYSTEM_PROMPT


def adapt_row(row, index, preserve_prompt=False, reward_metric="em"):
    info = row.get("extra_info") or {}
    question = row.get("question") or info.get("question")
    if not question:
        for message in reversed(row.get("prompt") or []):
            if message.get("role") == "user":
                content = message.get("content", "")
                if "Question:" in content:
                    # The question itself can begin with "Question:" (12 HotpotQA
                    # training rows do). Only remove the instruction's first marker.
                    question = content.split("Question:", 1)[1].strip()
                    break
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"Row {index} has no unambiguous question")
    reward = row.get("reward_model") or {}
    truth = reward.get("ground_truth", row.get("golden_answers"))
    targets = truth.get("target") if isinstance(truth, dict) else truth
    if isinstance(targets, str):
        targets = [targets]
    if not isinstance(targets, list) or not targets or any(not isinstance(x, str) or not x.strip() for x in targets):
        raise ValueError(f"Row {index} has invalid gold answers")
    if reward_metric != "em":
        raise ValueError("Reward alignment is deferred; only EM is currently enabled")
    if preserve_prompt and not row.get("prompt"):
        raise ValueError("Prompt preservation requires a source prompt")
    prompt = (
        copy.deepcopy(row["prompt"])
        if preserve_prompt
        else [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question.strip()}]
    )
    extra_info = dict(info) if preserve_prompt else {"index": index, "split": str(info.get("split", "unknown"))}
    extra_info.update(question=question.strip(), reward_metric=reward_metric)
    return {
        "data_source": str(row.get("data_source") or "search_r1"),
        "agent_name": "search_r1",
        "prompt": prompt,
        "ability": row.get("ability") or "fact-reasoning",
        "reward_model": {"style": "rule", "ground_truth": {"target": targets}},
        "extra_info": extra_info,
    }


def prepare(source, destination, preserve_prompt=False, reward_metric="em"):
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".parquet.partial")
    if temporary.exists():
        raise FileExistsError(temporary)
    count, sources, writer = 0, {}, None
    try:
        for batch in pq.ParquetFile(source).iter_batches(batch_size=1024):
            rows = []
            for row in batch.to_pylist():
                converted = adapt_row(row, count, preserve_prompt=preserve_prompt, reward_metric=reward_metric)
                rows.append(converted)
                key = converted["data_source"]
                sources[key] = sources.get(key, 0) + 1
                count += 1
            table = pa.Table.from_pylist(rows)
            if writer is None:
                writer = pq.ParquetWriter(temporary, table.schema)
            writer.write_table(table)
        if writer is None:
            raise ValueError(f"Empty dataset: {source}")
    except BaseException:
        if writer is not None:
            writer.close()
        temporary.unlink(missing_ok=True)
        raise
    writer.close()
    temporary.rename(destination)
    report = {
        "source": str(source.resolve()),
        "destination": str(destination.resolve()),
        "rows": count,
        "data_sources": sources,
        "context_format": "source-prompt/(assistant/tool)*" if preserve_prompt else "system/user/(assistant/tool)*",
        "source_prompt_preserved": preserve_prompt,
        "reward_metric": reward_metric,
    }
    destination.with_suffix(".audit.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--preserve-prompt", action="store_true")
    parser.add_argument("--reward-metric", choices=["em"], default="em")
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.destination, args.preserve_prompt, args.reward_metric), indent=2))
