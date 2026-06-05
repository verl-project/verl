#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import load_dataset

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
    generate_batched,
    generate_batched_with_backend,
    get_dist_context,
    init_distributed_if_needed,
    list_eval_model_ids,
    load_model_and_tokenizer,
    now_compact,
    qwen_is_correct_choice,
    qwen_is_correct_number,
    find_explicit_cli_flags,
    list_eval_inference_ids,
    resolve_eval_inference_profile,
    resolve_num_samples,
    score_gsm8k_mc_with_verl,
    score_gsm8k_with_verl,
    shard_for_rank,
    validate_inference_id_args,
    write_json,
)

from verl.utils.reward_score import gsm8k as verl_gsm8k


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
    backend=None,
    model=None,
    tokenizer=None,
) -> List[str]:
    if backend is not None:
        return generate_batched_with_backend(
            backend=backend,
            prompts=prompts,
            gen_cfg=gen_cfg,
            batch_size=batch_size,
        )
    if model is None or tokenizer is None:
        raise ValueError("Expected either a backend or a model/tokenizer pair for generation.")
    return generate_batched(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        gen_cfg=gen_cfg,
        batch_size=batch_size,
    )


def _compute_gsm8k_metrics(examples: List[Dict[str, Any]], parse_methods: List[ParseMethod]) -> Dict[str, Any]:
    agg = {m: {"correct": 0, "format_ok": 0, "total": 0} for m in parse_methods}
    qwen_agg = {"correct": 0, "total": 0}
    for ex in examples:
        scores = ex["scores"]
        for m in parse_methods:
            s = scores[f"verl_{m}"]
            agg[m]["correct"] += int(s["correct"])
            agg[m]["format_ok"] += int(s["format_ok"])
            agg[m]["total"] += 1
        q = scores["qwen_original"]
        qwen_agg["correct"] += int(q["correct"])
        qwen_agg["total"] += 1

    metrics = {
        f"verl_{m}": {
            "accuracy": (agg[m]["correct"] / agg[m]["total"]) if agg[m]["total"] else 0.0,
            "format_ok_rate": (agg[m]["format_ok"] / agg[m]["total"]) if agg[m]["total"] else 0.0,
            "correct": agg[m]["correct"],
            "total": agg[m]["total"],
        }
        for m in parse_methods
    }
    metrics["qwen_original"] = {
        "accuracy": (qwen_agg["correct"] / qwen_agg["total"]) if qwen_agg["total"] else 0.0,
        "correct": qwen_agg["correct"],
        "total": qwen_agg["total"],
    }
    return metrics


def _compute_gsm8k_mc_metrics(examples: List[Dict[str, Any]], parse_methods: List[ParseMethod]) -> Dict[str, Any]:
    agg = {m: {"correct": 0, "format_ok": 0, "total": 0} for m in parse_methods}
    qwen_agg = {"correct": 0, "format_ok": 0, "total": 0}
    for ex in examples:
        scores = ex["scores"]
        for m in parse_methods:
            s = scores[f"verl_{m}"]
            agg[m]["correct"] += int(s["correct"])
            agg[m]["format_ok"] += int(s["format_ok"])
            agg[m]["total"] += 1

        q = scores["qwen_original"]
        qwen_agg["correct"] += int(q["correct"])
        qwen_agg["format_ok"] += int(q["pred"] is not None)
        qwen_agg["total"] += 1

    metrics = {
        f"verl_{m}": {
            "accuracy": (agg[m]["correct"] / agg[m]["total"]) if agg[m]["total"] else 0.0,
            "format_ok_rate": (agg[m]["format_ok"] / agg[m]["total"]) if agg[m]["total"] else 0.0,
            "correct": agg[m]["correct"],
            "total": agg[m]["total"],
        }
        for m in parse_methods
    }
    metrics["qwen_original"] = {
        "accuracy": (qwen_agg["correct"] / qwen_agg["total"]) if qwen_agg["total"] else 0.0,
        "format_ok_rate": (qwen_agg["format_ok"] / qwen_agg["total"]) if qwen_agg["total"] else 0.0,
        "correct": qwen_agg["correct"],
        "total": qwen_agg["total"],
    }
    return metrics


def evaluate_one_model(
    *,
    model: Optional[str],
    model_id: Optional[str],
    gsm8k,
    gsm8k_mc,
    gen_cfg: GenerationConfig,
    batch_size: int,
    prompt_style: str,
    include_cot_phrase: bool,
    parse_methods: List[ParseMethod],
    torch_dtype: str,
    attn_implementation: Optional[str],
    wandb_mode: str,
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
        runs: List[Dict[str, Any]] = []

        # GSM8K
        prompts = []
        metas = []
        for idx, ex in enumerate(gsm8k):
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
            model=loaded_model,
            tokenizer=tokenizer,
            prompts=local_prompts,
            gen_cfg=gen_cfg,
            batch_size=batch_size,
        )
        local_examples = []
        for resp, meta in zip(responses, local_metas):
            gt = meta["ground_truth"]
            scores: Dict[str, Any] = {}
            for m in parse_methods:
                s = score_gsm8k_with_verl(completion=resp, ground_truth=gt, method=m)
                scores[f"verl_{m}"] = s
            q = qwen_is_correct_number(completion=resp, answer=meta["answer"])
            scores["qwen_original"] = q
            local_examples.append({**meta, "response": resp, "scores": scores})

        gathered = all_gather_objects(local_examples, dist_ctx)
        if dist_ctx.rank == 0:
            examples = []
            for shard_examples in gathered:
                examples.extend(shard_examples)
            examples.sort(key=lambda ex: int(ex["index"]))
            runs.append(
                {
                    "model_label": "BASE",
                    "dataset": "gsm8k",
                    "split": "test",
                    "prompt_style": prompt_style,
                    "use_chat_template": gen_cfg.use_chat_template,
                    "metrics": _compute_gsm8k_metrics(examples, parse_methods),
                    "examples": examples,
                }
            )

        # GSM8K-MC
        prompts = []
        metas = []
        for idx, ex in enumerate(gsm8k_mc):
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
            model=loaded_model,
            tokenizer=tokenizer,
            prompts=local_prompts,
            gen_cfg=gen_cfg,
            batch_size=batch_size,
        )
        local_examples = []
        for resp, meta in zip(responses, local_metas):
            gt = meta["ground_truth"]
            scores: Dict[str, Any] = {}
            for m in parse_methods:
                s = score_gsm8k_mc_with_verl(completion=resp, ground_truth_letter=gt, method=m)
                scores[f"verl_{m}"] = s

            q = qwen_is_correct_choice(completion=resp, gold_letter=gt)
            scores["qwen_original"] = q
            local_examples.append({**meta, "response": resp, "scores": scores})

        gathered = all_gather_objects(local_examples, dist_ctx)
        if dist_ctx.rank == 0:
            examples = []
            for shard_examples in gathered:
                examples.extend(shard_examples)
            examples.sort(key=lambda ex: int(ex["index"]))
            runs.append(
                {
                    "model_label": "BASE",
                    "dataset": "gsm8k_mc",
                    "split": "test",
                    "prompt_style": prompt_style,
                    "use_chat_template": gen_cfg.use_chat_template,
                    "metrics": _compute_gsm8k_mc_metrics(examples, parse_methods),
                    "examples": examples,
                }
            )
    finally:
        _free_runtime(backend=backend, model=loaded_model, tokenizer=tokenizer)
        barrier_if_needed(dist_ctx)
    return runs, model_config


def main() -> None:
    explicit_decoding_flags = find_explicit_cli_flags(sys.argv[1:], INFERENCE_OVERRIDE_FLAGS)
    explicit_model_override_flags = find_explicit_cli_flags(sys.argv[1:], MODEL_OVERRIDE_FLAGS)
    parser = argparse.ArgumentParser(description="Eval a model or model profile on GSM8K + GSM8K-MC with dual parsing.")
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--model", help="Model (local dir, HF repo id, or W&B artifact ref).")
    model_group.add_argument(
        "--model_id",
        choices=list_eval_model_ids(),
        help="Model profile id from reliable_gsm8k.profiles.MODEL_PROFILES.",
    )

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
        gsm8k = load_dataset(args.gsm8k_dataset, args.gsm8k_config, split=args.gsm8k_split)
        gsm8k_mc = load_dataset(args.mc_dataset, split=args.mc_split)
        if num_samples is not None:
            gsm8k = _subset(gsm8k, num_samples)
            gsm8k_mc = _subset(gsm8k_mc, num_samples)

        inference_profile = resolve_eval_inference_profile(args.inference_id)
        if inference_profile is not None:
            gen_cfg = GenerationConfig(
                max_new_tokens=int(inference_profile["max_new_tokens"]),
                max_length=int(inference_profile["max_length"]),
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
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                use_chat_template=not args.no_chat_template,
                system_prompt=args.system_prompt,
            )

        parse_methods: List[ParseMethod] = [m for m in args.parse_methods]  # type: ignore[assignment]
        include_cot_phrase = not args.no_cot_phrase

        runs, model_config = evaluate_one_model(
            model=args.model,
            model_id=args.model_id,
            gsm8k=gsm8k,
            gsm8k_mc=gsm8k_mc,
            gen_cfg=gen_cfg,
            batch_size=args.batch_size,
            prompt_style=args.prompt_style,
            include_cot_phrase=include_cot_phrase,
            parse_methods=parse_methods,
            torch_dtype=args.torch_dtype,
            attn_implementation=args.attn_implementation,
            wandb_mode=args.wandb_mode,
            dist_ctx=dist_ctx,
        )

        if dist_ctx.rank != 0:
            return

        out_json = args.out_json or f"evals/oe_mc_eval_05_02_26/base_qwen_dual_parse_{now_compact()}.json"
        payload: Dict[str, Any] = {
            "config": {
                **model_config,
                "gsm8k_dataset": args.gsm8k_dataset,
                "gsm8k_config": args.gsm8k_config,
                "gsm8k_split": args.gsm8k_split,
                "mc_dataset": args.mc_dataset,
                "mc_split": args.mc_split,
                "num_samples": num_samples,
                "num_samples_source": num_samples_source,
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
            },
            "runs": runs,
        }
        write_json(out_json, payload)

        print("\n" + "#" * 80)
        print("SUMMARY")
        print("#" * 80)
        for r in runs:
            _print_run_summary(r)
        print(f"\nSaved: {out_json}")

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
                config=payload["config"],
            )

            flat_metrics: Dict[str, Any] = {}
            for r in runs:
                model_label = r["model_label"]
                dataset = r["dataset"]
                for parser_name, m in r["metrics"].items():
                    prefix = f"{model_label}/{dataset}/{parser_name}"
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
                metadata=payload["config"],
            )
            artifact.add_file(out_json)
            run.log_artifact(artifact)
            run.finish()
    finally:
        cleanup_distributed_if_needed(dist_ctx)


if __name__ == "__main__":
    main()
