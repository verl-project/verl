#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import math
import os
import re
import time
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

from tqdm import tqdm

# Ensure repo-root imports regardless of CWD (so `import verl` works).
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
BUILDER_SRC = REPO_ROOT / "reliable-gsm8k-builder" / "src"
if str(BUILDER_SRC) not in sys.path:
    sys.path.insert(0, str(BUILDER_SRC))


# -----------------------------------------------------------------------------
# VERL parsers (authoritative for "your verl methods")
# -----------------------------------------------------------------------------

from verl.utils.reward_score import gsm8k as verl_gsm8k
from verl.utils.reward_score import gsm8k_mc as verl_gsm8k_mc
from reliable_gsm8k.backends import create_generation_backend, get_backend_runtime_metadata
from reliable_gsm8k.profiles import INFERENCE_PROFILES, MODEL_PROFILES, get_generator_profile, get_inference_profile


# -----------------------------------------------------------------------------
# Qwen-original style parser (mirrors evals/qwen_original/evaluate_chat_gsm8k.py)
# -----------------------------------------------------------------------------

_PAT_LAST_DIGIT = re.compile(
    r"([+-])?(?=([0-9]|\.[0-9]))(0|([1-9](\d{0,2}(,\d{3})*)|\d*))?(\.\d*)?(?=\D|$)"
)


def qwen_extract_last_number(text: str) -> Optional[str]:
    matches = list(_PAT_LAST_DIGIT.finditer(text))
    if not matches:
        return None
    return matches[-1].group().replace(",", "").replace("+", "").strip()


def qwen_is_correct_number(completion: str, answer: str, *, abs_tol: float = 1e-4) -> Dict[str, Any]:
    gold = qwen_extract_last_number(answer)
    pred = qwen_extract_last_number(completion)
    if gold is None:
        raise ValueError("No ground truth number found in the GSM8K answer field.")
    if pred is None:
        return {"correct": False, "gold": gold, "pred": None}
    try:
        # The regex only returns numeric strings, so eval(...) is safe and matches the original script.
        ok = math.isclose(eval(gold), eval(pred), rel_tol=0, abs_tol=abs_tol)
    except Exception:
        ok = False
    return {"correct": ok, "gold": gold, "pred": pred}


def qwen_extract_last_choice(text: str, *, clip_chars: int = 300) -> Optional[str]:
    if len(text) > clip_chars:
        text = text[-clip_chars:]
    candidates = re.findall(r"\b([A-D])\b", text)
    if not candidates:
        return None
    return candidates[-1]


def qwen_is_correct_choice(completion: str, gold_letter: str) -> Dict[str, Any]:
    pred = qwen_extract_last_choice(completion)
    if pred is None:
        return {"correct": False, "gold": gold_letter, "pred": None}
    return {"correct": pred.strip() == (gold_letter or "").strip(), "gold": gold_letter, "pred": pred}


# -----------------------------------------------------------------------------
# Prompt builders
# -----------------------------------------------------------------------------

PromptStyle = Literal["train", "raw"]


def build_gsm8k_prompt(*, question: str, prompt_style: PromptStyle, include_cot_phrase: bool) -> str:
    question = (question or "").strip()
    if prompt_style == "raw":
        return question

    suffix = 'output the final answer after "####".'
    if include_cot_phrase:
        suffix = f"Let's think step by step and {suffix}"
    return f"{question} {suffix}"


def _collect_mc_options(example: Dict[str, Any]) -> List[Tuple[str, str]]:
    options: List[Tuple[str, str]] = []
    for label in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        if label in example and example[label] not in (None, ""):
            options.append((label, str(example[label]).strip()))
    return options


def build_gsm8k_mc_prompt(
    *,
    question: str,
    example: Dict[str, Any],
    prompt_style: PromptStyle,
    include_cot_phrase: bool,
) -> str:
    question = (question or "").strip()
    options = _collect_mc_options(example)

    if prompt_style == "raw":
        # Minimal formatting: question + A. ... lines, no additional instructions.
        lines = [question] if question else []
        for label, text in options:
            lines.append(f"{label}. {text}")
        return "\n".join(lines).strip()

    # Training-style formatting: matches examples/data_preprocess/gsm8k_mc.py conventions.
    option_lines = [f"{label}: {text}" for label, text in options]
    options_block = "\n".join(option_lines)

    suffix = 'output the letter of the final answer choice after "####".'
    if include_cot_phrase:
        suffix = f"Let's think step by step and {suffix}"

    return f"{question}\n\nOptions:\n{options_block}\n\n{suffix}".strip()


# -----------------------------------------------------------------------------
# Scoring helpers (returns normalized dicts)
# -----------------------------------------------------------------------------

ParseMethod = Literal["strict", "flexible"]


@dataclass(frozen=True)
class DistContext:
    enabled: bool
    world_size: int
    rank: int
    local_rank: int


def get_dist_context() -> DistContext:
    world_size = max(int(os.environ.get("WORLD_SIZE", "1")), 1)
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    return DistContext(
        enabled=world_size > 1,
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
    )


def init_distributed_if_needed(dist_ctx: DistContext) -> None:
    if not dist_ctx.enabled:
        return

    import torch
    import torch.distributed as dist

    if torch.cuda.is_available():
        torch.cuda.set_device(dist_ctx.local_rank)

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            rank=dist_ctx.rank,
            world_size=dist_ctx.world_size,
        )


def barrier_if_needed(dist_ctx: DistContext) -> None:
    if not dist_ctx.enabled:
        return
    import torch.distributed as dist

    if dist.is_initialized():
        dist.barrier()


def all_gather_objects(local_obj: Any, dist_ctx: DistContext) -> List[Any]:
    if not dist_ctx.enabled:
        return [local_obj]
    import torch.distributed as dist

    if not dist.is_initialized():
        return [local_obj]

    gathered: List[Any] = [None for _ in range(dist_ctx.world_size)]
    dist.all_gather_object(gathered, local_obj)
    return gathered


def cleanup_distributed_if_needed(dist_ctx: DistContext) -> None:
    if not dist_ctx.enabled:
        return
    import torch.distributed as dist

    if dist.is_initialized():
        dist.destroy_process_group()


def shard_for_rank(items: Sequence[Any], dist_ctx: DistContext) -> List[Any]:
    if not dist_ctx.enabled:
        return list(items)
    return [item for idx, item in enumerate(items) if idx % dist_ctx.world_size == dist_ctx.rank]


def score_gsm8k_with_verl(*, completion: str, ground_truth: str, method: ParseMethod) -> Dict[str, Any]:
    pred = verl_gsm8k.extract_solution(completion, method=method)
    score = float(
        verl_gsm8k.compute_score(
            solution_str=completion,
            ground_truth=ground_truth,
            method=method,
            format_score=0.0,
            score=1.0,
        )
    )
    return {
        "method": method,
        "pred": pred,
        "format_ok": pred is not None,
        "score": score,
        "correct": score >= 1.0,
    }


def score_gsm8k_mc_with_verl(*, completion: str, ground_truth_letter: str, method: ParseMethod) -> Dict[str, Any]:
    out = verl_gsm8k_mc.compute_score(
        data_source="gsm8k_mc",
        solution_str=completion,
        ground_truth=ground_truth_letter,
        method=method,
        format_score=0.0,
        score=1.0,
    )
    # normalize
    return {
        "method": method,
        "pred": (out.get("pred") or "").strip(),
        "format_ok": bool(out.get("format_ok", False)),
        "score": float(out.get("score", 0.0)),
        "correct": float(out.get("score", 0.0)) >= 1.0,
    }


# -----------------------------------------------------------------------------
# Generation + model loading
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 512
    max_length: int = 2048
    number_of_samples: int = 1
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    use_chat_template: bool = True
    system_prompt: str = "You are a helpful assistant."


INFERENCE_OVERRIDE_FLAGS: Tuple[str, ...] = (
    "--max_new_tokens",
    "--max_length",
    "--do_sample",
    "--temperature",
    "--top_p",
    "--repetition_penalty",
    "--no_chat_template",
    "--system_prompt",
)


def list_eval_inference_ids() -> List[str]:
    return sorted(INFERENCE_PROFILES.keys())


def list_eval_model_ids() -> List[str]:
    return sorted(MODEL_PROFILES.keys())


def resolve_eval_model_profile(name: str) -> Dict[str, Any]:
    if name not in MODEL_PROFILES:
        known = ", ".join(list_eval_model_ids())
        raise ValueError(f"Unknown --model_id '{name}'. Available: {known}")
    return get_generator_profile(name)


def resolve_eval_inference_profile(name: Optional[str]) -> Optional[Dict[str, Any]]:
    if name is None:
        return None
    if name not in INFERENCE_PROFILES:
        known = ", ".join(list_eval_inference_ids())
        raise ValueError(f"Unknown --inference_id '{name}'. Available: {known}")
    return get_inference_profile(name)


def find_explicit_cli_flags(argv: Sequence[str], flags: Sequence[str]) -> List[str]:
    flag_set = set(flags)
    explicit: set[str] = set()
    for token in argv:
        if not token.startswith("--"):
            continue
        option = token.split("=", 1)[0]
        if option in flag_set:
            explicit.add(option)
    return sorted(explicit)


def validate_inference_id_args(*, inference_id: Optional[str], explicit_decoding_flags: Sequence[str]) -> None:
    if inference_id is None:
        return
    if not explicit_decoding_flags:
        return
    joined = ", ".join(explicit_decoding_flags)
    raise ValueError(f"--inference_id cannot be combined with manual decoding flags: {joined}")


def _backend_prefers_local_transformers(config: Dict[str, Any]) -> bool:
    return str(config.get("backend")) == "transformers_causal_lm"


def _runtime_profile_for_dist(config: Dict[str, Any], dist_ctx: DistContext) -> Dict[str, Any]:
    runtime_config = dict(config)
    if _backend_prefers_local_transformers(runtime_config) and dist_ctx.enabled:
        runtime_config["device_map"] = None
        try:
            import torch

            if torch.cuda.is_available():
                runtime_config["device"] = f"cuda:{dist_ctx.local_rank}"
            else:
                runtime_config["device"] = "cpu"
        except Exception:
            runtime_config["device"] = "cpu"
    return runtime_config


def create_generation_backend_for_eval(
    *,
    model_id: str,
    dist_ctx: DistContext,
) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
    resolved_profile = resolve_eval_model_profile(model_id)
    runtime_profile = _runtime_profile_for_dist(resolved_profile, dist_ctx)
    backend = create_generation_backend(runtime_profile)
    runtime_info = get_backend_runtime_metadata(backend, runtime_profile)
    return backend, resolved_profile, runtime_info


def generation_sampling_from_config(gen_cfg: GenerationConfig) -> Dict[str, Any]:
    sampling: Dict[str, Any] = {
        "max_tokens": int(gen_cfg.max_new_tokens),
        "max_length": int(gen_cfg.max_length),
        "do_sample": bool(gen_cfg.do_sample),
        "use_chat_template": bool(gen_cfg.use_chat_template),
        "system_prompt": str(gen_cfg.system_prompt),
        "n": int(gen_cfg.number_of_samples),
    }
    repetition_penalty = float(gen_cfg.repetition_penalty)
    if repetition_penalty > 0:
        sampling["repetition_penalty"] = repetition_penalty
    if gen_cfg.do_sample:
        sampling["temperature"] = float(gen_cfg.temperature)
        sampling["top_p"] = float(gen_cfg.top_p)
    return sampling


def generate_batched_multi(
    *,
    model,
    tokenizer,
    prompts: Sequence[str],
    gen_cfg: GenerationConfig,
    batch_size: int,
    progress_desc: Optional[str] = None,
    progress_position: Optional[int] = None,
) -> List[List[str]]:
    import torch

    formatted = _format_for_generation(
        tokenizer=tokenizer,
        prompts=prompts,
        use_chat_template=gen_cfg.use_chat_template,
        system_prompt=gen_cfg.system_prompt,
    )

    results: List[List[str]] = [[] for _ in formatted]
    num_batches = math.ceil(len(formatted) / batch_size) if formatted else 0
    progress = tqdm(
        total=int(gen_cfg.number_of_samples) * num_batches,
        desc=progress_desc,
        position=progress_position,
        leave=True,
        dynamic_ncols=True,
        disable=progress_desc is None or num_batches == 0,
    )
    for sample_index in range(int(gen_cfg.number_of_samples)):
        sample_results: List[str] = []
        for batch_index, start in enumerate(range(0, len(formatted), batch_size), start=1):
            batch = formatted[start : start + batch_size]
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=int(gen_cfg.max_length),
            )
            input_device = _get_input_device(model)
            inputs = {k: v.to(input_device) for k, v in inputs.items()}

            gen_kwargs: Dict[str, Any] = dict(
                max_new_tokens=int(gen_cfg.max_new_tokens),
                do_sample=bool(gen_cfg.do_sample),
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=float(gen_cfg.repetition_penalty),
            )
            if gen_cfg.do_sample:
                gen_kwargs.update(
                    temperature=float(gen_cfg.temperature),
                    top_p=float(gen_cfg.top_p),
                )

            with torch.inference_mode():
                outputs = model.generate(**inputs, **gen_kwargs)

            prompt_len = inputs["input_ids"].shape[1]
            gen_ids = outputs[:, prompt_len:]
            texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            sample_results.extend([t.strip() for t in texts])
            progress.set_postfix(sample=f"{sample_index + 1}/{int(gen_cfg.number_of_samples)}", batch=f"{batch_index}/{num_batches}")
            progress.update(1)

        for prompt_idx, text in enumerate(sample_results):
            results[prompt_idx].append(text)
    progress.close()
    return results


def generate_batched_with_backend_multi(
    *,
    backend,
    prompts: Sequence[str],
    gen_cfg: GenerationConfig,
    batch_size: int,
    progress_desc: Optional[str] = None,
    progress_position: Optional[int] = None,
) -> List[List[str]]:
    model = getattr(backend, "_model", None)
    tokenizer = getattr(backend, "_tokenizer", None)
    if model is not None and tokenizer is not None:
        return generate_batched_multi(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            gen_cfg=gen_cfg,
            batch_size=batch_size,
            progress_desc=progress_desc,
            progress_position=progress_position,
        )

    del batch_size
    sampling = generation_sampling_from_config(gen_cfg)
    results: List[List[str]] = []
    progress = tqdm(
        total=len(prompts),
        desc=progress_desc,
        position=progress_position,
        leave=True,
        dynamic_ncols=True,
        disable=progress_desc is None or not prompts,
    )
    for idx, prompt in enumerate(prompts):
        responses = backend.generate(prompt=prompt, sampling=sampling, metadata={"prompt_index": idx})
        if not responses:
            raise RuntimeError("Generation backend returned no responses for prompt.")
        results.append([(response.text or "").strip() for response in responses])
        progress.update(1)
    progress.close()
    return results


def generate_batched_with_backend(
    *,
    backend,
    prompts: Sequence[str],
    gen_cfg: GenerationConfig,
    batch_size: int,
) -> List[str]:
    return [sample_texts[0] if sample_texts else "" for sample_texts in generate_batched_with_backend_multi(
        backend=backend,
        prompts=prompts,
        gen_cfg=gen_cfg,
        batch_size=batch_size,
    )]


def _get_input_device(model) -> "Any":
    # device_map="auto" shards params; inputs must be on the device of the first param.
    return next(model.parameters()).device


def _looks_like_wandb_artifact_ref(ref: str) -> bool:
    # Common form: entity/project/artifact_name:alias
    # Example: tommaso-bendinelli-eth-zurich/multiple_choice_question_study/qwen25_3B_gsm8k:v0
    if not isinstance(ref, str):
        return False
    if os.path.exists(ref):
        return False
    if ":" not in ref:
        return False
    parts = ref.split("/")
    if len(parts) < 3:
        return False
    return True


def _wandb_artifact_cache_dir() -> str:
    base_dir = os.environ.get("VERL_RUN_DIR", os.path.expanduser("~/.cache/verl"))
    return os.path.join(os.path.abspath(os.path.expanduser(base_dir)), "models")


def maybe_download_wandb_artifact_model(
    model_ref: str,
    *,
    wandb_mode: str = "online",
    cache_root: Optional[str] = None,
) -> str:
    """
    If `model_ref` is a W&B artifact reference (entity/project/artifact:alias), download it and return local dir.
    Otherwise return `model_ref` unchanged.
    """
    model_ref = str(model_ref)
    if os.path.isdir(model_ref) and os.listdir(model_ref):
        return model_ref
    if not _looks_like_wandb_artifact_ref(model_ref):
        return model_ref

    cache_root = cache_root or _wandb_artifact_cache_dir()
    os.makedirs(cache_root, exist_ok=True)

    safe_name = model_ref.replace("/", "__").replace(":", "_")
    target_dir = os.path.join(cache_root, safe_name)
    if os.path.isdir(target_dir) and os.listdir(target_dir):
        return target_dir

    if wandb_mode == "disabled":
        raise ValueError(
            "Model reference looks like a W&B artifact, but wandb_mode=disabled. "
            "Enable W&B (WANDB_MODE=online/offline) or pass a local model directory instead."
        )

    import wandb

    # Use the project implied by the artifact ref if available (entity/project/...),
    # otherwise fall back to a generic project name.
    parts = model_ref.split("/")
    implied_project = parts[1] if len(parts) >= 2 else "artifact_download"

    run = wandb.init(
        project=implied_project,
        job_type="artifact_download",
        mode=wandb_mode,
        reinit=True,
    )
    try:
        artifact = run.use_artifact(model_ref, type="model")
        artifact.download(root=target_dir)
    finally:
        try:
            run.finish()
        except Exception:
            pass

    return target_dir


def load_model_and_tokenizer(
    model_name_or_path: str,
    *,
    torch_dtype: str = "float16",
    attn_implementation: Optional[str] = None,
    wandb_mode: str = "online",
    wandb_cache_root: Optional[str] = None,
    device_map: Optional[str] = "auto",
    local_rank: Optional[int] = None,
):
    # Imports are inside the function to keep file import lightweight.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name_or_path = maybe_download_wandb_artifact_model(
        model_name_or_path,
        wandb_mode=wandb_mode,
        cache_root=wandb_cache_root,
    )

    if attn_implementation == "flash_attention_2":
        try:
            import flash_attn  # noqa: F401
        except Exception as e:
            # Transformers will raise if flash-attn isn't installed; fall back to SDPA.
            print(
                f"[WARN] attn_implementation=flash_attention_2 requested but flash_attn import failed ({e}). "
                "Falling back to attn_implementation=sdpa.",
                file=sys.stderr,
            )
            attn_implementation = "sdpa"

    dtype = getattr(torch, torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: Dict[str, Any] = dict(
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    if device_map is not None:
        model_kwargs["device_map"] = device_map
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        **model_kwargs,
    ).eval()
    if device_map is None:
        if torch.cuda.is_available():
            if local_rank is None:
                local_rank = 0
            model.to(torch.device(f"cuda:{local_rank}"))
        else:
            model.to(torch.device("cpu"))
    return model, tokenizer


def _format_for_generation(
    *,
    tokenizer,
    prompts: Sequence[str],
    use_chat_template: bool,
    system_prompt: str,
) -> List[str]:
    if not use_chat_template:
        return list(prompts)

    formatted: List[str] = []
    for p in prompts:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": p})
        formatted.append(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )
    return formatted


def generate_batched(
    *,
    model,
    tokenizer,
    prompts: Sequence[str],
    gen_cfg: GenerationConfig,
    batch_size: int,
) -> List[str]:
    return [sample_texts[0] if sample_texts else "" for sample_texts in generate_batched_multi(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        gen_cfg=gen_cfg,
        batch_size=batch_size,
    )]


def now_compact() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def write_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def resolve_num_samples(cli_value: Optional[int], env_value: Optional[str]) -> Tuple[Optional[int], str]:
    """
    Resolve the effective sample cap.

    Returns:
        (num_samples, source)
        - num_samples: None means full split
        - source: one of {"all", "env", "cli"}
    """
    if cli_value is not None:
        if cli_value <= 0:
            raise ValueError("--num_samples must be > 0 when provided. Omit it to evaluate the full split.")
        return int(cli_value), "cli"

    if env_value is None or str(env_value).strip() == "":
        return None, "all"

    raw = str(env_value).strip()
    try:
        value = int(raw)
    except ValueError as e:
        raise ValueError("NUM_SAMPLES must be a positive integer when set.") from e

    if value <= 0:
        raise ValueError("NUM_SAMPLES must be > 0 when set. Unset it to evaluate the full split.")
    return value, "env"
