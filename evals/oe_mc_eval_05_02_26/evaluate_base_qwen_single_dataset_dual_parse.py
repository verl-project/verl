#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from datasets import load_dataset
from tqdm import tqdm

# Ensure repo-root + local imports regardless of CWD.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eval_utils import (
    GenerationConfig,
    INFERENCE_OVERRIDE_FLAGS,
    ParseMethod,
    all_gather_objects,
    barrier_if_needed,
    build_gsm8k_mc_prompt,
    build_gsm8k_prompt,
    cleanup_distributed_if_needed,
    create_generation_backend_for_eval,
    find_explicit_cli_flags,
    generate_batched_multi,
    generate_batched_with_backend_multi,
    get_dist_context,
    init_distributed_if_needed,
    list_eval_inference_ids,
    list_eval_model_ids,
    load_model_and_tokenizer,
    now_compact,
    qwen_is_correct_choice,
    qwen_is_correct_number,
    resolve_eval_inference_profile,
    resolve_num_samples,
    score_gsm8k_mc_with_verl,
    score_gsm8k_with_verl,
    shard_for_rank,
    validate_inference_id_args,
    write_json,
)

from verl.utils.reward_score import gsm8k as verl_gsm8k


DatasetKind = Literal["gsm8k", "gsm8k_mc"]
MODEL_OVERRIDE_FLAGS = ("--torch_dtype", "--attn_implementation")


def _subset(dataset, num_samples: Optional[int]):
    if num_samples is None:
        return dataset
    return dataset.select(range(min(int(num_samples), len(dataset))))


def _print_run_summary(run: Dict[str, Any]) -> None:
    label = run["model_label"]
    ds = run["dataset"]
    m = run["metrics"]
    print(f"\n{label} → {ds}")
    for k, v in m.items():
        if "accuracy" in v:
            extra = ""
            if "format_ok_rate" in v:
                extra = f", format_ok={v['format_ok_rate']:.2%}"
            print(f"  {k}: acc={v['accuracy']:.2%}{extra} ({v['correct']}/{v['total']})")


def _default_answer_out_json(*, timestamp: str) -> str:
    return f"evals/oe_mc_eval_05_02_26/{timestamp}_answer.json"


def _answer_shard_json_path(answer_out_json: str, *, rank: int) -> str:
    out_path = Path(answer_out_json)
    return str(out_path.with_name(f"{out_path.stem}.rank{rank:02d}.part.json"))


def _shared_run_timestamp(*, dist_ctx) -> str:
    local_value = now_compact() if dist_ctx.rank == 0 else None
    gathered = all_gather_objects(local_value, dist_ctx)
    for value in gathered:
        if value:
            return str(value)
    return now_compact()


class JsonArrayWriter:
    def __init__(self, path: str) -> None:
        self.path = path
        self._handle = None
        self._needs_comma = False

    def __enter__(self) -> "JsonArrayWriter":
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._handle = open(self.path, "w", encoding="utf-8")
        self._handle.write("[\n")
        return self

    def write(self, record: Dict[str, Any]) -> None:
        if self._handle is None:
            raise RuntimeError("writer is not open")
        if self._needs_comma:
            self._handle.write(",\n")
        json.dump(record, self._handle, ensure_ascii=False)
        self._handle.flush()
        self._needs_comma = True

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        if self._handle is not None:
            self._handle.write("\n]\n")
            self._handle.close()
            self._handle = None


def _resolve_model_path(model_config: Dict[str, Any]) -> Optional[str]:
    if model_config.get("model_source") == "raw_model":
        return model_config.get("model")

    model_profile = model_config.get("model_profile") or {}
    model_runtime = model_config.get("model_runtime") or {}
    return (
        model_profile.get("model_name")
        or model_profile.get("model_name_or_path")
        or model_runtime.get("model_name")
        or model_runtime.get("model_name_or_path")
    )


def _build_question_id(*, dataset_name: str, example: Dict[str, Any]) -> Any:
    if dataset_name != "gsm8k":
        return example.get("index")

    question = example.get("question")
    if not isinstance(question, str) or not question:
        raise ValueError("GSM8K answer records require a non-empty 'question' field to derive question_id.")
    return hashlib.md5(question.encode("utf-8")).hexdigest()


def _build_answer_records(*, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    config = payload["config"]
    model_config = {
        "model_source": config.get("model_source"),
        "model": config.get("model"),
        "model_id": config.get("model_id"),
        "model_profile": config.get("model_profile"),
        "model_runtime": config.get("model_runtime"),
    }
    model_path = _resolve_model_path(model_config)
    inference_id = config.get("inference_id")

    answer_records: List[Dict[str, Any]] = []
    for run in payload["runs"]:
        dataset_name = run["dataset"]
        split = run["split"]
        for example in run["examples"]:
            answer_records.append(
                {
                    "identity": {
                        "question_id": _build_question_id(dataset_name=dataset_name, example=example),
                        "dataset_name": dataset_name,
                        "split": split,
                        "inference_id": inference_id,
                        "model_id": config.get("model_id"),
                        "model_path": model_path,
                    },
                    "payload": {
                        "responses": list(example.get("responses") or [example["response"]]),
                    },
                }
            )
    return answer_records


def _build_answer_record(
    *,
    example: Dict[str, Any],
    dataset_name: str,
    split: str,
    inference_id: Optional[str],
    model_config: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "identity": {
            "question_id": _build_question_id(dataset_name=dataset_name, example=example),
            "dataset_name": dataset_name,
            "split": split,
            "inference_id": inference_id,
            "model_id": model_config.get("model_id"),
            "model_path": _resolve_model_path(model_config),
        },
        "payload": {
            "responses": list(example.get("responses") or [example["response"]]),
        },
    }


def _question_id_sort_key(question_id: Any):
    text = str(question_id or "")
    if text.isdigit():
        return (0, int(text))
    return (1, text)


def _merge_answer_shards(*, shard_paths: List[str], answer_out_json: str) -> None:
    merged_records: List[Dict[str, Any]] = []
    for shard_path in shard_paths:
        if not os.path.exists(shard_path):
            continue
        with open(shard_path, "r", encoding="utf-8") as handle:
            merged_records.extend(json.load(handle))

    merged_records.sort(
        key=lambda record: (
            str(record["identity"].get("dataset_name") or ""),
            str(record["identity"].get("split") or ""),
            _question_id_sort_key(record["identity"].get("question_id")),
        )
    )
    write_json(answer_out_json, merged_records)
    for shard_path in shard_paths:
        if os.path.exists(shard_path):
            os.remove(shard_path)


def _free_runtime(*, backend=None, model=None, tokenizer=None) -> None:
    try:
        import torch

        del backend
        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def _generate_responses(
    *,
    prompts,
    gen_cfg: GenerationConfig,
    batch_size: int,
    progress_desc: Optional[str] = None,
    progress_position: Optional[int] = None,
    backend=None,
    model=None,
    tokenizer=None,
) -> List[List[str]]:
    if backend is not None:
        return generate_batched_with_backend_multi(
            backend=backend,
            prompts=prompts,
            gen_cfg=gen_cfg,
            batch_size=batch_size,
            progress_desc=progress_desc,
            progress_position=progress_position,
        )
    if model is None or tokenizer is None:
        raise ValueError("Expected either a backend or a model/tokenizer pair for generation.")
    return generate_batched_multi(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        gen_cfg=gen_cfg,
        batch_size=batch_size,
        progress_desc=progress_desc,
        progress_position=progress_position,
    )


def _summarize_sample_scores(sample_scores: List[Dict[str, Any]], *, include_format: bool) -> Dict[str, Any]:
    total = len(sample_scores)
    correct_count = sum(int(bool(score["correct"])) for score in sample_scores)
    first_sample = dict(sample_scores[0])
    first_sample["accuracy"] = float(bool(first_sample["correct"]))
    first_sample["total"] = 1

    average_summary: Dict[str, Any] = {
        "accuracy": (correct_count / total) if total else 0.0,
        "correct": correct_count,
        "total": total,
    }
    pass_at_k_summary: Dict[str, Any] = {
        "accuracy": 1.0 if any(bool(score["correct"]) for score in sample_scores) else 0.0,
        "correct": any(bool(score["correct"]) for score in sample_scores),
        "total": 1,
        "k": total,
    }

    if include_format:
        format_ok_count = sum(int(bool(score["format_ok"])) for score in sample_scores)
        first_sample["format_ok_rate"] = float(bool(first_sample["format_ok"]))
        average_summary["format_ok_rate"] = (format_ok_count / total) if total else 0.0
        average_summary["format_ok"] = format_ok_count
        pass_at_k_summary["format_ok_rate"] = 1.0 if any(bool(score["format_ok"]) for score in sample_scores) else 0.0
        pass_at_k_summary["format_ok"] = any(bool(score["format_ok"]) for score in sample_scores)

    return {
        "first_sample": first_sample,
        "average": average_summary,
        "pass_at_k": pass_at_k_summary,
    }


def _aggregate_metric_family(
    examples: List[Dict[str, Any]],
    *,
    parser_key: str,
    family: str,
    include_format: bool,
) -> Dict[str, Any]:
    correct = 0
    total = 0
    format_ok = 0
    for ex in examples:
        summary = ex["scores"][parser_key][family]
        if family == "first_sample":
            correct += int(bool(summary["correct"]))
            total += 1
            if include_format:
                format_ok += int(bool(summary["format_ok"]))
        else:
            correct += int(summary["correct"])
            total += int(summary["total"])
            if include_format:
                format_ok += int(summary["format_ok"])

    metric = {
        "accuracy": (correct / total) if total else 0.0,
        "correct": correct,
        "total": total,
    }
    if include_format:
        metric["format_ok_rate"] = (format_ok / total) if total else 0.0
    if family == "pass_at_k" and examples:
        metric["k"] = len(examples[0].get("responses", []))
    return metric


def _compute_gsm8k_metrics(examples: List[Dict[str, Any]], parse_methods: List[ParseMethod]) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    families = ("first_sample", "average", "pass_at_k")
    for m in parse_methods:
        parser_key = f"verl_{m}"
        for family in families:
            metrics[f"{parser_key}__{family}"] = _aggregate_metric_family(
                examples,
                parser_key=parser_key,
                family=family,
                include_format=True,
            )
    for family in families:
        metrics[f"qwen_original__{family}"] = _aggregate_metric_family(
            examples,
            parser_key="qwen_original",
            family=family,
            include_format=False,
        )
    return metrics


def _compute_gsm8k_mc_metrics(examples: List[Dict[str, Any]], parse_methods: List[ParseMethod]) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    families = ("first_sample", "average", "pass_at_k")
    for m in parse_methods:
        parser_key = f"verl_{m}"
        for family in families:
            metrics[f"{parser_key}__{family}"] = _aggregate_metric_family(
                examples,
                parser_key=parser_key,
                family=family,
                include_format=True,
            )
    for family in families:
        metrics[f"qwen_original__{family}"] = _aggregate_metric_family(
            examples,
            parser_key="qwen_original",
            family=family,
            include_format=True,
        )
    return metrics


def _evaluate_gsm8k(
    *,
    dataset,
    dataset_split: str,
    gen_cfg: GenerationConfig,
    batch_size: int,
    prompt_style: str,
    include_cot_phrase: bool,
    parse_methods: List[ParseMethod],
    inference_id: Optional[str],
    model_config: Dict[str, Any],
    answer_writer: Optional[JsonArrayWriter],
    dist_ctx,
    backend=None,
    model=None,
    tokenizer=None,
) -> Dict[str, Any]:
    prompts = []
    metas = []
    for idx, ex in enumerate(dataset):
        q = ex["question"]
        ans = ex["answer"]
        gt = verl_gsm8k.extract_solution(ans, method="strict") or ""
        prompt = build_gsm8k_prompt(question=q, prompt_style=prompt_style, include_cot_phrase=include_cot_phrase)
        prompts.append(prompt)
        metas.append({"index": idx, "question": q, "answer": ans, "ground_truth": gt, "prompt": prompt})

    local_prompts = shard_for_rank(prompts, dist_ctx)
    local_metas = shard_for_rank(metas, dist_ctx)
    responses = _generate_responses(
        backend=backend,
        model=model,
        tokenizer=tokenizer,
        prompts=local_prompts,
        gen_cfg=gen_cfg,
        batch_size=batch_size,
        progress_desc=f"rank {dist_ctx.rank} generate",
        progress_position=dist_ctx.local_rank * 2,
    )
    local_examples = []
    scoring_progress = tqdm(
        total=len(local_metas),
        desc=f"rank {dist_ctx.rank} score",
        position=dist_ctx.local_rank * 2 + 1,
        leave=True,
        dynamic_ncols=True,
    )
    for sample_responses, meta in zip(responses, local_metas):
        gt = meta["ground_truth"]
        sample_scores: List[Dict[str, Any]] = []
        for resp in sample_responses:
            per_sample_scores: Dict[str, Any] = {}
            for m in parse_methods:
                s = score_gsm8k_with_verl(completion=resp, ground_truth=gt, method=m)
                per_sample_scores[f"verl_{m}"] = s
            per_sample_scores["qwen_original"] = qwen_is_correct_number(completion=resp, answer=meta["answer"])
            sample_scores.append(per_sample_scores)

        scores: Dict[str, Any] = {}
        for m in parse_methods:
            parser_key = f"verl_{m}"
            scores[parser_key] = _summarize_sample_scores(
                [sample_score[parser_key] for sample_score in sample_scores],
                include_format=True,
            )
        scores["qwen_original"] = _summarize_sample_scores(
            [sample_score["qwen_original"] for sample_score in sample_scores],
            include_format=False,
        )
        local_examples.append(
            {
                **meta,
                "response": sample_responses[0],
                "responses": sample_responses,
                "sample_scores": sample_scores,
                "scores": scores,
            }
        )
        if answer_writer is not None:
            answer_writer.write(
                _build_answer_record(
                    example=local_examples[-1],
                    dataset_name="gsm8k",
                    split=dataset_split,
                    inference_id=inference_id,
                    model_config=model_config,
                )
            )
        scoring_progress.update(1)
    scoring_progress.close()

    gathered = all_gather_objects(local_examples, dist_ctx)
    if dist_ctx.rank != 0:
        return {}

    examples = []
    for shard_examples in gathered:
        examples.extend(shard_examples)
    examples.sort(key=lambda ex: int(ex["index"]))
    return {
        "model_label": "BASE",
        "dataset": "gsm8k",
        "split": dataset_split,
        "prompt_style": prompt_style,
        "use_chat_template": gen_cfg.use_chat_template,
        "metrics": _compute_gsm8k_metrics(examples, parse_methods),
        "examples": examples,
    }


def _evaluate_gsm8k_mc(
    *,
    dataset,
    dataset_split: str,
    gen_cfg: GenerationConfig,
    batch_size: int,
    prompt_style: str,
    include_cot_phrase: bool,
    parse_methods: List[ParseMethod],
    inference_id: Optional[str],
    model_config: Dict[str, Any],
    answer_writer: Optional[JsonArrayWriter],
    dist_ctx,
    backend=None,
    model=None,
    tokenizer=None,
) -> Dict[str, Any]:
    prompts = []
    metas = []
    for idx, ex in enumerate(dataset):
        q = ex.get("Question") or ex.get("question") or ""
        gt = (ex.get("Answer") or ex.get("answer") or "").strip()
        prompt = build_gsm8k_mc_prompt(
            question=q,
            example=dict(ex),
            prompt_style=prompt_style,
            include_cot_phrase=include_cot_phrase,
        )
        prompts.append(prompt)
        metas.append(
            {
                "index": idx,
                "question": q,
                "ground_truth": gt,
                "prompt": prompt,
                "choices": {k: ex.get(k) for k in ["A", "B", "C", "D"] if ex.get(k) not in (None, "")},
            }
        )

    local_prompts = shard_for_rank(prompts, dist_ctx)
    local_metas = shard_for_rank(metas, dist_ctx)
    responses = _generate_responses(
        backend=backend,
        model=model,
        tokenizer=tokenizer,
        prompts=local_prompts,
        gen_cfg=gen_cfg,
        batch_size=batch_size,
        progress_desc=f"rank {dist_ctx.rank} generate",
        progress_position=dist_ctx.local_rank * 2,
    )
    local_examples = []
    scoring_progress = tqdm(
        total=len(local_metas),
        desc=f"rank {dist_ctx.rank} score",
        position=dist_ctx.local_rank * 2 + 1,
        leave=True,
        dynamic_ncols=True,
    )
    for sample_responses, meta in zip(responses, local_metas):
        gt = meta["ground_truth"]
        sample_scores: List[Dict[str, Any]] = []
        for resp in sample_responses:
            per_sample_scores: Dict[str, Any] = {}
            for m in parse_methods:
                s = score_gsm8k_mc_with_verl(completion=resp, ground_truth_letter=gt, method=m)
                per_sample_scores[f"verl_{m}"] = s

            q = qwen_is_correct_choice(completion=resp, gold_letter=gt)
            q["format_ok"] = q["pred"] is not None
            per_sample_scores["qwen_original"] = q
            sample_scores.append(per_sample_scores)

        scores: Dict[str, Any] = {}
        for m in parse_methods:
            parser_key = f"verl_{m}"
            scores[parser_key] = _summarize_sample_scores(
                [sample_score[parser_key] for sample_score in sample_scores],
                include_format=True,
            )
        scores["qwen_original"] = _summarize_sample_scores(
            [sample_score["qwen_original"] for sample_score in sample_scores],
            include_format=True,
        )
        local_examples.append(
            {
                **meta,
                "response": sample_responses[0],
                "responses": sample_responses,
                "sample_scores": sample_scores,
                "scores": scores,
            }
        )
        if answer_writer is not None:
            answer_writer.write(
                _build_answer_record(
                    example=local_examples[-1],
                    dataset_name="gsm8k_mc",
                    split=dataset_split,
                    inference_id=inference_id,
                    model_config=model_config,
                )
            )
        scoring_progress.update(1)
    scoring_progress.close()

    gathered = all_gather_objects(local_examples, dist_ctx)
    if dist_ctx.rank != 0:
        return {}

    examples = []
    for shard_examples in gathered:
        examples.extend(shard_examples)
    examples.sort(key=lambda ex: int(ex["index"]))
    return {
        "model_label": "BASE",
        "dataset": "gsm8k_mc",
        "split": dataset_split,
        "prompt_style": prompt_style,
        "use_chat_template": gen_cfg.use_chat_template,
        "metrics": _compute_gsm8k_mc_metrics(examples, parse_methods),
        "examples": examples,
    }


def evaluate_one_model(
    *,
    model: Optional[str],
    model_id: Optional[str],
    dataset_kind: DatasetKind,
    dataset,
    dataset_split: str,
    gen_cfg: GenerationConfig,
    batch_size: int,
    prompt_style: str,
    include_cot_phrase: bool,
    parse_methods: List[ParseMethod],
    inference_id: Optional[str],
    torch_dtype: str,
    attn_implementation: Optional[str],
    wandb_mode: str,
    answer_shard_path: str,
    dist_ctx,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    backend = None
    loaded_model = None
    tokenizer = None
    model_config: Dict[str, Any]
    if model_id is not None:
        backend, model_profile, model_runtime = create_generation_backend_for_eval(
            model_id=model_id,
            dist_ctx=dist_ctx,
        )
        model_config = {
            "model_source": "model_profile",
            "model_id": model_id,
            "model_profile": model_profile,
            "model_runtime": model_runtime,
        }
    else:
        loaded_model, tokenizer = load_model_and_tokenizer(
            model or "",
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            wandb_mode=wandb_mode,
            device_map=None if dist_ctx.enabled else "auto",
            local_rank=dist_ctx.local_rank,
        )
        model_config = {
            "model_source": "raw_model",
            "model": model,
            "torch_dtype": torch_dtype,
            "attn_implementation": attn_implementation,
        }

    try:
        with JsonArrayWriter(answer_shard_path) as answer_writer:
            if dataset_kind == "gsm8k":
                run = _evaluate_gsm8k(
                    dataset=dataset,
                    dataset_split=dataset_split,
                    gen_cfg=gen_cfg,
                    batch_size=batch_size,
                    prompt_style=prompt_style,
                    include_cot_phrase=include_cot_phrase,
                    parse_methods=parse_methods,
                    inference_id=inference_id,
                    model_config=model_config,
                    answer_writer=answer_writer,
                    dist_ctx=dist_ctx,
                    backend=backend,
                    model=loaded_model,
                    tokenizer=tokenizer,
                )
            else:
                run = _evaluate_gsm8k_mc(
                    dataset=dataset,
                    dataset_split=dataset_split,
                    gen_cfg=gen_cfg,
                    batch_size=batch_size,
                    prompt_style=prompt_style,
                    include_cot_phrase=include_cot_phrase,
                    parse_methods=parse_methods,
                    inference_id=inference_id,
                    model_config=model_config,
                    answer_writer=answer_writer,
                    dist_ctx=dist_ctx,
                    backend=backend,
                    model=loaded_model,
                    tokenizer=tokenizer,
                )
    finally:
        _free_runtime(backend=backend, model=loaded_model, tokenizer=tokenizer)
        barrier_if_needed(dist_ctx)

    if dist_ctx.rank != 0:
        return [], model_config
    return [run], model_config


def main() -> None:
    explicit_decoding_flags = find_explicit_cli_flags(sys.argv[1:], INFERENCE_OVERRIDE_FLAGS)
    explicit_model_override_flags = find_explicit_cli_flags(sys.argv[1:], MODEL_OVERRIDE_FLAGS)
    parser = argparse.ArgumentParser(description="Eval a model or model profile on one dataset with dual parsing.")
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--model", help="Model (local dir, HF repo id, or W&B artifact ref).")
    model_group.add_argument(
        "--model_id",
        choices=list_eval_model_ids(),
        help="Model profile id from reliable_gsm8k.profiles.MODEL_PROFILES.",
    )
    parser.add_argument("--dataset_kind", required=True, choices=["gsm8k", "gsm8k_mc"], help="Which dataset to evaluate.")

    parser.add_argument("--gsm8k_dataset", default="openai/gsm8k")
    parser.add_argument("--gsm8k_config", default="main")
    parser.add_argument("--gsm8k_split", default="test")
    parser.add_argument("--mc_dataset", default="guipenedo/gsm8k-mc")
    parser.add_argument("--mc_split", default="test")

    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Optional cap on number of examples per dataset. Omit for full split. "
        "If omitted, a positive $NUM_SAMPLES value is used when set.",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Generation batch size per GPU process.")
    parser.add_argument(
        "--number_of_samples",
        type=int,
        default=1,
        help="Number of generated responses to sample per question.",
    )
    parser.add_argument(
        "--inference_id",
        default=None,
        choices=list_eval_inference_ids(),
        help="Named inference profile. Cannot be combined with manual decoding flags.",
    )

    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--no_chat_template", action="store_true")
    parser.add_argument("--system_prompt", default="You are a helpful assistant.")
    parser.add_argument(
        "--torch_dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
        help="torch_dtype for raw --model loading (must be supported by your GPU).",
    )
    parser.add_argument(
        "--attn_implementation",
        choices=["eager", "sdpa", "flash_attention_2"],
        default=None,
        help="Optional attention backend for raw --model loading.",
    )

    parser.add_argument("--prompt_style", choices=["train", "raw"], default="train")
    parser.add_argument("--no_cot_phrase", action="store_true")
    parser.add_argument("--parse_methods", nargs="+", choices=["strict", "flexible"], default=["strict", "flexible"])
    parser.add_argument("--save_main_eval", action="store_true", help="Save the full main eval JSON.")

    parser.add_argument("--out_json", default=None)

    # Optional W&B logging (stores the JSON as an artifact).
    parser.add_argument("--wandb_project", default=os.environ.get("WANDB_PROJECT", ""), help="If set, log to W&B.")
    parser.add_argument("--wandb_entity", default=os.environ.get("WANDB_ENTITY", ""))
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--wandb_group", default=os.environ.get("WANDB_GROUP", None))
    parser.add_argument("--wandb_tags", nargs="*", default=[])
    parser.add_argument(
        "--wandb_mode",
        choices=["online", "offline", "disabled"],
        default=os.environ.get("WANDB_MODE", "online"),
    )
    parser.add_argument("--wandb_job_type", default="evaluation")
    parser.add_argument("--wandb_artifact_name", default="base_qwen_dual_parse_results")
    parser.add_argument("--wandb_artifact_type", default="eval_results")

    args = parser.parse_args()
    if args.number_of_samples <= 0:
        parser.error("--number_of_samples must be > 0.")
    if args.out_json and not args.save_main_eval:
        parser.error("--out_json requires --save_main_eval.")
    try:
        validate_inference_id_args(
            inference_id=args.inference_id,
            explicit_decoding_flags=explicit_decoding_flags,
        )
    except ValueError as e:
        parser.error(str(e))
    if args.model_id is not None and explicit_model_override_flags:
        joined = ", ".join(explicit_model_override_flags)
        parser.error(f"--model_id cannot be combined with manual model-loading flags: {joined}")

    dist_ctx = get_dist_context()
    init_distributed_if_needed(dist_ctx)
    try:
        num_samples, num_samples_source = resolve_num_samples(args.num_samples, os.environ.get("NUM_SAMPLES"))
        dataset_kind: DatasetKind = args.dataset_kind
        if dataset_kind == "gsm8k":
            dataset = load_dataset(args.gsm8k_dataset, args.gsm8k_config, split=args.gsm8k_split)
            if num_samples is not None:
                dataset = _subset(dataset, num_samples)
        else:
            dataset = load_dataset(args.mc_dataset, split=args.mc_split)
            if num_samples is not None:
                dataset = _subset(dataset, num_samples)

        inference_profile = resolve_eval_inference_profile(args.inference_id)
        if inference_profile is not None:
            gen_cfg = GenerationConfig(
                max_new_tokens=int(inference_profile["max_new_tokens"]),
                max_length=int(inference_profile["max_length"]),
                number_of_samples=args.number_of_samples,
                do_sample=bool(inference_profile["do_sample"]),
                temperature=float(inference_profile["temperature"]),
                top_p=float(inference_profile["top_p"]),
                repetition_penalty=float(inference_profile["repetition_penalty"]),
                use_chat_template=bool(inference_profile["use_chat_template"]),
                system_prompt=str(inference_profile["system_prompt"]),
            )
        else:
            gen_cfg = GenerationConfig(
                max_new_tokens=args.max_new_tokens,
                max_length=args.max_length,
                number_of_samples=args.number_of_samples,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                use_chat_template=not args.no_chat_template,
                system_prompt=args.system_prompt,
            )

        parse_methods: List[ParseMethod] = [m for m in args.parse_methods]  # type: ignore[assignment]
        include_cot_phrase = not args.no_cot_phrase
        run_timestamp = _shared_run_timestamp(dist_ctx=dist_ctx)
        answer_out_json = _default_answer_out_json(timestamp=run_timestamp)
        answer_shard_path = _answer_shard_json_path(answer_out_json, rank=dist_ctx.rank)

        runs, model_config = evaluate_one_model(
            model=args.model,
            model_id=args.model_id,
            dataset_kind=dataset_kind,
            dataset=dataset,
            dataset_split=args.gsm8k_split if dataset_kind == "gsm8k" else args.mc_split,
            gen_cfg=gen_cfg,
            batch_size=args.batch_size,
            prompt_style=args.prompt_style,
            include_cot_phrase=include_cot_phrase,
            parse_methods=parse_methods,
            inference_id=args.inference_id,
            torch_dtype=args.torch_dtype,
            attn_implementation=args.attn_implementation,
            wandb_mode=args.wandb_mode,
            answer_shard_path=answer_shard_path,
            dist_ctx=dist_ctx,
        )

        if dist_ctx.rank != 0:
            return

        config_payload: Dict[str, Any] = {
            **model_config,
            "dataset_kind": dataset_kind,
            "gsm8k_dataset": args.gsm8k_dataset,
            "gsm8k_config": args.gsm8k_config,
            "gsm8k_split": args.gsm8k_split,
            "mc_dataset": args.mc_dataset,
            "mc_split": args.mc_split,
            "num_samples": num_samples,
            "num_samples_source": num_samples_source,
            "number_of_samples": args.number_of_samples,
            "inference_id": args.inference_id,
            "inference_profile": inference_profile,
            "prompt_style": args.prompt_style,
            "include_cot_phrase": include_cot_phrase,
            "parse_methods": parse_methods,
            "generation": gen_cfg.__dict__,
            "wandb": {
                "project": args.wandb_project,
                "entity": args.wandb_entity,
                "mode": args.wandb_mode,
            },
            "distributed": {
                "enabled": dist_ctx.enabled,
                "world_size": dist_ctx.world_size,
            },
        }
        shard_paths = [_answer_shard_json_path(answer_out_json, rank=rank) for rank in range(dist_ctx.world_size)]
        _merge_answer_shards(shard_paths=shard_paths, answer_out_json=answer_out_json)

        payload: Optional[Dict[str, Any]] = None
        out_json: Optional[str] = None
        if args.save_main_eval:
            out_json = args.out_json or (
                f"evals/oe_mc_eval_05_02_26/base_qwen_single_dataset_dual_parse_{run_timestamp}.json"
            )
            payload = {
                "config": config_payload,
                "runs": runs,
            }
            write_json(out_json, payload)

        print("\n" + "#" * 80)
        print("SUMMARY")
        print("#" * 80)
        for r in runs:
            _print_run_summary(r)
        if out_json is not None:
            print(f"\nSaved: {out_json}")
        print(f"Saved answers: {answer_out_json}")

        # Log to W&B (optional).
        if args.wandb_project and args.wandb_mode != "disabled":
            import wandb

            run = wandb.init(
                project=args.wandb_project,
                entity=(args.wandb_entity or None),
                name=args.wandb_run_name,
                group=args.wandb_group,
                tags=args.wandb_tags,
                job_type=args.wandb_job_type,
                mode=args.wandb_mode,
                config=config_payload,
            )

            flat_metrics: Dict[str, Any] = {}
            for r in runs:
                model_label = r["model_label"]
                dataset_name = r["dataset"]
                for parser_name, m in r["metrics"].items():
                    prefix = f"{model_label}/{dataset_name}/{parser_name}"
                    if "accuracy" in m:
                        flat_metrics[f"{prefix}/accuracy"] = m["accuracy"]
                    if "format_ok_rate" in m:
                        flat_metrics[f"{prefix}/format_ok_rate"] = m["format_ok_rate"]
                    if "correct" in m:
                        flat_metrics[f"{prefix}/correct"] = m["correct"]
                    if "total" in m:
                        flat_metrics[f"{prefix}/total"] = m["total"]

            wandb.log(flat_metrics)

            artifact = wandb.Artifact(
                name=args.wandb_artifact_name,
                type=args.wandb_artifact_type,
                metadata=config_payload,
            )
            if out_json is not None:
                artifact.add_file(out_json)
            artifact.add_file(answer_out_json)
            run.log_artifact(artifact)
            run.finish()
    finally:
        cleanup_distributed_if_needed(dist_ctx)


if __name__ == "__main__":
    main()
