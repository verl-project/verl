#!/usr/bin/env python3
"""Evaluate step0 HF model + all `global_step_*` checkpoints with vLLM."""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import requests  # type: ignore
except Exception:
    requests = None  # type: ignore

try:
    import asyncio
    import aiohttp  # type: ignore
except Exception:
    asyncio = None  # type: ignore
    aiohttp = None  # type: ignore

# ---------------------------
# Parquet loading helpers
# ---------------------------


def _de_numpy(x: Any) -> Any:
    """Recursively convert numpy containers/scalars into plain Python."""
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return [_de_numpy(v) for v in x.tolist()]
    if isinstance(x, dict):
        return {k: _de_numpy(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_de_numpy(v) for v in x]
    return x


def load_records_from_parquet(parquet_path: str | Path) -> List[Dict[str, Any]]:
    """Load records and sanitize types.

    Fixes the common case where `prompt` becomes a numpy array after
    `df.to_dict("records")`.

    Ensures each record has:
      - record["prompt"]: List[{"role":..., "content":...}]
    """
    df = pd.read_parquet(str(parquet_path))
    records = df.to_dict("records")

    fixed: List[Dict[str, Any]] = []
    for r in records:
        r = _de_numpy(r)

        p = r.get("prompt")
        if p is None:
            r["prompt"] = []
        elif isinstance(p, dict):
            r["prompt"] = [p]
        elif isinstance(p, list):
            r["prompt"] = p
        else:
            # best-effort fallback
            try:
                r["prompt"] = list(p)
            except Exception:
                r["prompt"] = []

        fixed.append(r)

    return fixed


def sample_indices(n_total: int, n: int, seed: int) -> List[int]:
    """Return a deterministic subset of indices."""
    if n <= 0 or n >= n_total:
        return list(range(n_total))
    rng = np.random.default_rng(seed)
    idx = rng.choice(n_total, size=n, replace=False)
    return sorted(int(i) for i in idx.tolist())


# ---------------------------
# Checkpoint scanning + merging
# ---------------------------

_STEP_RE = re.compile(r"global_step_(\d+)$")


@dataclass(frozen=True)
class StepCkpt:
    step: int
    step_dir: Path          # .../global_step_XX
    actor_dir: Path         # .../global_step_XX/actor (preferred for VERL FSDP runs)

    @property
    def hf_config_dir(self) -> Path:
        # VERL's merger expects a HuggingFace config directory at <local_dir>/huggingface
        return self.actor_dir / "huggingface"



def scan_exp_dir(exp_dir: str | Path) -> List[StepCkpt]:
    """Scan an experiment directory for `global_step_*` checkpoints.

    For VERL FSDP checkpoints, model shards + HF config are usually under:
      global_step_XX/actor/...
      global_step_XX/actor/huggingface/

    We set `actor_dir` to that path when present; otherwise we fall back to `step_dir`.
    """
    exp_dir = Path(exp_dir)
    if not exp_dir.exists():
        raise FileNotFoundError(f"checkpoint-dir not found: {exp_dir}")

    ckpts: List[StepCkpt] = []
    for p in exp_dir.iterdir():
        if not p.is_dir():
            continue
        m = _STEP_RE.search(p.name)
        if not m:
            continue

        step = int(m.group(1))
        actor_dir = p / "actor"
        if actor_dir.is_dir():
            ckpts.append(StepCkpt(step=step, step_dir=p, actor_dir=actor_dir))
        else:
            ckpts.append(StepCkpt(step=step, step_dir=p, actor_dir=p))

    ckpts.sort(key=lambda x: x.step)
    return ckpts


def merged_hf_dir_for_step(exp_dir: Path, step: int) -> Path:
    return exp_dir / "_merged_hf_cache" / f"global_step_{step}"


def run_verl_merge_fsdp_to_hf(
    local_dir: Path,
    out_dir: Path,
    python_exec: str,
    overwrite: bool,
) -> None:
    """Run VERL's merger (FSDP -> HuggingFace safetensors) as a subprocess.

    Important: for VERL GRPO/FSDP runs, shards + HF config are typically under
    `global_step_XX/actor/`, so `local_dir` should be that `actor` directory.
    """
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    # VERL's FSDP merger expects HF config at: <local_dir>/huggingface/config.json
    hf_cfg = local_dir / "huggingface" / "config.json"
    if not hf_cfg.exists():
        raise FileNotFoundError(
            "Missing HuggingFace config for VERL merger."
            f"Expected: {hf_cfg}"
            f"Got local_dir={local_dir}"
            "Your checkpoint likely has layout: global_step_XX/actor/huggingface/. "
            "If so, local_dir must be the 'actor' subdir."
        )

    if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
        return

    if out_dir.exists() and overwrite:
        subprocess.run(["rm", "-rf", str(out_dir)], check=False)

    cmd = [
        python_exec,
        "-m",
        "verl.model_merger",
        "merge",
        "--backend",
        "fsdp",
        "--local_dir",
        str(local_dir),
        "--target_dir",
        str(out_dir),
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "VERL merge failed.\n"
            f"Command: {' '.join(cmd)}\n\n"
            f"STDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}"
        )

    # Basic sanity: HF dir should have config.json.
    if not (out_dir / "config.json").exists():
        raise RuntimeError(
            f"VERL merge finished but config.json is missing at: {out_dir}\n"
            "This usually means the target_dir is wrong or the merge did not write HF artifacts."
        )


# ---------------------------
# vLLM inference + scoring
# ---------------------------


def _infer_tensor_parallel_size() -> int:
    """Infer TP size from CUDA_VISIBLE_DEVICES, else torch.cuda.device_count."""
    v = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if v:
        parts = [p.strip() for p in v.split(",") if p.strip() != ""]
        if parts:
            return len(parts)
    try:
        import torch

        return max(1, int(torch.cuda.device_count()))
    except Exception:
        return 1


def get_score_fn():
    """Return a function (solution_str, ground_truth_str) -> bool."""
    from verl.utils.reward_score.math_verify import compute_score as verl_compute_score

    def _score(sol: str, gt: str) -> bool:
        return bool(verl_compute_score(sol, gt))

    return _score


def build_vllm_llm(model_dir_or_name: str, tokenizer_dir_or_name: str):
    """Instantiate vLLM engine.

    We always pass `tokenizer=step0_model` to keep tokenization consistent across checkpoints
    and to avoid warnings caused by saved tokenizer.json in merged dirs.
    """
    # Avoid HF tokenizers fork warning spam.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    from vllm import LLM

    gpu_mem_util = float(os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.80"))
    tp = _infer_tensor_parallel_size()

    kwargs = dict(
        model=str(model_dir_or_name),
        tokenizer=str(tokenizer_dir_or_name),
        dtype="auto",
        tensor_parallel_size=tp,
        gpu_memory_utilization=gpu_mem_util,
        disable_log_stats=True,
        trust_remote_code=False,
    )

    return LLM(**kwargs)


def vllm_generate_solutions(
    llm,
    records: List[Dict[str, Any]],
    batch_size: int,
    temperature: float,
    top_p: float,
    max_tokens: Optional[int],
    seed: int,
    pass_k: int = 1,
) -> List[List[str]]:
    """Generate pass_k solutions per record.
    
    Returns: List[List[str]] where outer list is per record, inner list is k solutions.
    """
    from vllm import SamplingParams

    sp = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        seed=seed,
        n=pass_k,  # Generate k solutions per prompt
    )

    all_solutions: List[List[str]] = []
    n_batches = (len(records) + batch_size - 1) // batch_size

    for i in tqdm(
        range(0, len(records), batch_size),
        desc="vllm inference",
        total=n_batches,
        file=sys.stderr,
    ):
        batch = records[i : i + batch_size]
        messages_list = [r["prompt"] for r in batch]
        outs = llm.chat(messages_list, sampling_params=sp)
        # Each output has k solutions in outputs[0:k]
        for o in outs:
            solutions = [output.text for output in o.outputs]
            all_solutions.append(solutions)

    return all_solutions


def eval_records_with_vllm(
    model_dir_or_name: str,
    tokenizer_dir_or_name: str,
    records: List[Dict[str, Any]],
    batch_size: int,
    temperature: float,
    top_p: float,
    max_tokens: Optional[int],
    seed: int,
    pass_k: int = 1,
) -> Tuple[float, List[bool]]:
    llm = build_vllm_llm(model_dir_or_name, tokenizer_dir_or_name)

    solutions = vllm_generate_solutions(
        llm=llm,
        records=records,
        batch_size=batch_size,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        seed=seed,
        pass_k=pass_k,
    )

    score_fn = get_score_fn()

    oks: List[bool] = []
    for r, sol_list in tqdm(
        zip(records, solutions),
        total=len(records),
        desc="compute_score",
        file=sys.stderr,
    ):
        gt = r["reward_model"]["ground_truth"]
        # pass@k: check if ANY of the k solutions is correct
        any_correct = any(bool(score_fn(sol, gt)) for sol in sol_list)
        oks.append(any_correct)

    acc = float(sum(oks) / max(1, len(oks)))
    return acc, oks


# ---------------------------
# vLLM Serve (OpenAI API) helpers
# ---------------------------


def _infer_visible_gpu_count() -> int:
    v = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if v:
        parts = [p.strip() for p in v.split(",") if p.strip()]
        return max(1, len(parts))
    return 1


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _wait_vllm_ready(base_url: str, timeout_s: int) -> str:
    """Wait until vLLM OpenAI server is ready.

    Returns the first model id from /v1/models.
    """
    if requests is None:
        raise RuntimeError("requests is required for --use-vllm-serve, but it's not installed.")

    t0 = time.time()
    last_err: str | None = None
    while time.time() - t0 < timeout_s:
        try:
            r = requests.get(f"{base_url}/v1/models", timeout=2)
            if r.status_code == 200:
                obj = r.json()
                data = obj.get("data") or []
                if data and isinstance(data, list) and isinstance(data[0], dict) and "id" in data[0]:
                    return str(data[0]["id"])
                # Sometimes ready but models list empty momentarily
                last_err = f"models empty: {obj}"
            else:
                last_err = f"status={r.status_code} body={r.text[:200]}"
        except Exception as e:
            last_err = repr(e)
        time.sleep(0.5)

    raise RuntimeError(f"vLLM server not ready after {timeout_s}s. last_err={last_err}")


def _start_vllm_serve(
    *,
    model: str,
    tokenizer: str,
    host: str,
    port: int,
    tp_size: int,
    dp_size: int,
    gpu_mem_util: float,
    max_model_len: Optional[int],
) -> subprocess.Popen:
    """Start vLLM OpenAI API server as a subprocess.

    Uses a dedicated process group so we can reliably terminate all workers.
    """
    # Prefer module invocation for robustness across environments.
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--host",
        host,
        "--port",
        str(port),
        "--model",
        str(model),
        "--tokenizer",
        str(tokenizer),
        "--tensor-parallel-size",
        str(tp_size),
        "--data-parallel-size",
        str(dp_size),
        "--gpu-memory-utilization",
        str(gpu_mem_util),
        "--disable-log-stats",
    ]
    if max_model_len is not None:
        cmd += ["--max-model-len", str(max_model_len)]

    # Make logs unbuffered; keep stdout/stderr for debugging.
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")

    return subprocess.Popen(
        cmd,
        stdout=sys.stderr,
        stderr=sys.stderr,
        text=True,
        env=env,
        start_new_session=True,
        bufsize=1,
    )


def _stop_process_tree(proc: subprocess.Popen, timeout_s: int) -> Tuple[str, str]:
    """Terminate a process group; return (stdout, stderr) collected so far."""
    out = ""
    err = ""
    try:
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except Exception:
                proc.terminate()
        try:
            out, err = proc.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                proc.kill()
            out2, err2 = proc.communicate(timeout=5)
            out += out2
            err += err2
    except Exception:
        # Best effort.
        try:
            out, err = proc.communicate(timeout=1)
        except Exception:
            pass
    return out, err


def _openai_chat_completion(
    *,
    base_url: str,
    model_id: str,
    messages: List[Dict[str, Any]],
    temperature: float,
    top_p: float,
    max_tokens: Optional[int],
    seed: int,
    n: int = 1,
) -> List[str]:
    """Generate n completions via OpenAI API (sync version).
    
    Returns: List[str] of n solutions.
    """
    if requests is None:
        raise RuntimeError("requests is required for --use-vllm-serve, but it's not installed.")

    payload: Dict[str, Any] = {
        "model": model_id,
        "messages": messages,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "stream": False,
        "n": int(n),  # Request n completions
    }
    if max_tokens is not None:
        payload["max_tokens"] = int(max_tokens)
    # vLLM supports `seed` in OpenAI-compatible API.
    payload["seed"] = int(seed)

    r = requests.post(f"{base_url}/v1/chat/completions", json=payload, timeout=600)
    r.raise_for_status()
    obj = r.json()
    choices = obj.get("choices") or []
    if not choices:
        return [""] * n
    
    # Extract all n completions
    results = []
    for ch in choices:
        # OpenAI chat format
        msg = ch.get("message") if isinstance(ch, dict) else None
        if isinstance(msg, dict) and "content" in msg:
            results.append(str(msg.get("content") or ""))
        # Fallbacks
        elif isinstance(ch, dict) and "text" in ch:
            results.append(str(ch.get("text") or ""))
        else:
            results.append("")
    
    # Ensure we always return exactly n results
    while len(results) < n:
        results.append("")
    
    return results[:n]


async def _openai_chat_completion_async(
    *,
    session: "aiohttp.ClientSession",
    base_url: str,
    model_id: str,
    messages: List[Dict[str, Any]],
    temperature: float,
    top_p: float,
    max_tokens: Optional[int],
    seed: int,
    n: int = 1,
    max_retries: int = 3,
) -> List[str]:
    """Generate n completions via OpenAI API (async version with retries).
    
    Returns: List[str] of n solutions.
    """
    payload: Dict[str, Any] = {
        "model": model_id,
        "messages": messages,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "stream": False,
        "n": int(n),
    }
    if max_tokens is not None:
        payload["max_tokens"] = int(max_tokens)
    payload["seed"] = int(seed)

    url = f"{base_url}/v1/chat/completions"
    timeout = aiohttp.ClientTimeout(total=3600)  # 1 hour timeout for pass@k with long generations
    
    last_error = None
    for attempt in range(max_retries):
        try:
            async with session.post(url, json=payload, timeout=timeout) as response:
                response.raise_for_status()
                obj = await response.json()
            
            choices = obj.get("choices") or []
            if not choices:
                return [""] * n
            
            results = []
            for ch in choices:
                msg = ch.get("message") if isinstance(ch, dict) else None
                if isinstance(msg, dict) and "content" in msg:
                    results.append(str(msg.get("content") or ""))
                elif isinstance(ch, dict) and "text" in ch:
                    results.append(str(ch.get("text") or ""))
                else:
                    results.append("")
            
            while len(results) < n:
                results.append("")
            
            return results[:n]
        
        except (asyncio.TimeoutError, aiohttp.ClientError) as e:
            last_error = e
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # exponential backoff: 1s, 2s, 4s
                print(f"Request failed (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s: {e}", file=sys.stderr)
                await asyncio.sleep(wait_time)
            else:
                print(f"Request failed after {max_retries} attempts: {e}", file=sys.stderr)
                # Return empty results on final failure
                return [""] * n
    
    # Should never reach here, but just in case
    return [""] * n


def eval_records_with_vllm_serve(
    *,
    model_dir_or_name: str,
    tokenizer_dir_or_name: str,
    records: List[Dict[str, Any]],
    batch_size: int,
    temperature: float,
    top_p: float,
    max_tokens: Optional[int],
    seed: int,
    host: str,
    port: int,
    start_timeout_s: int,
    stop_timeout_s: int,
    tp_size: int,
    dp_size: int,
    gpu_mem_util: float,
    max_model_len: Optional[int],
    pass_k: int = 1,
    max_concurrent_requests: int = 64,
) -> Tuple[float, List[bool]]:
    """Evaluate records using vLLM serve with async concurrent requests.
    
    Args:
        max_concurrent_requests: Maximum number of concurrent requests to vLLM server.
            This controls how many requests are in-flight at once. Higher values = more
            throughput but more memory usage. Default 64 works well for DP=4.
    """
    if aiohttp is None or asyncio is None:
        raise RuntimeError(
            "aiohttp and asyncio are required for async vLLM serve eval. "
            "Install with: pip install aiohttp"
        )
    
    if port <= 0:
        port = _pick_free_port()

    base_url = f"http://{host}:{port}"

    proc = _start_vllm_serve(
        model=model_dir_or_name,
        tokenizer=tokenizer_dir_or_name,
        host=host,
        port=port,
        tp_size=tp_size,
        dp_size=dp_size,
        gpu_mem_util=gpu_mem_util,
        max_model_len=max_model_len,
    )

    try:
        # If server exits early, surface logs.
        t0 = time.time()
        while proc.poll() is None and time.time() - t0 < start_timeout_s:
            # Try ready probe; if it raises, keep looping.
            try:
                model_id = _wait_vllm_ready(base_url, timeout_s=2)
                break
            except Exception:
                time.sleep(0.5)
        else:
            # Either exited or timed out.
            out, err = _stop_process_tree(proc, timeout_s=stop_timeout_s)
            raise RuntimeError(
                "vLLM serve failed to become ready.\n"
                f"model={model_dir_or_name}\n"
                f"base_url={base_url}\n\n"
                f"STDOUT:\n{out}\n\nSTDERR:\n{err}"
            )

        # If we broke out via model_id, ensure it's defined.
        model_id = _wait_vllm_ready(base_url, timeout_s=start_timeout_s)

        # Run async inference with periodic server health checks
        try:
            oks = asyncio.run(_eval_records_async(
                base_url=base_url,
                model_id=model_id,
                records=records,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                seed=seed,
                pass_k=pass_k,
                max_concurrent_requests=max_concurrent_requests,
                proc=proc,  # Pass process for health monitoring
            ))
        except Exception as e:
            # Check if server crashed
            if proc.poll() is not None:
                out, err = _stop_process_tree(proc, timeout_s=stop_timeout_s)
                raise RuntimeError(
                    f"vLLM server crashed during inference: {e}\n"
                    f"STDOUT:\n{out}\n\nSTDERR:\n{err}"
                )
            else:
                raise

        acc = float(sum(oks) / max(1, len(oks)))
        return acc, oks
    finally:
        _stop_process_tree(proc, timeout_s=stop_timeout_s)


async def _eval_records_async(
    *,
    base_url: str,
    model_id: str,
    records: List[Dict[str, Any]],
    temperature: float,
    top_p: float,
    max_tokens: Optional[int],
    seed: int,
    pass_k: int,
    max_concurrent_requests: int,
    proc: Optional[subprocess.Popen] = None,
) -> List[bool]:
    """Async concurrent evaluation of records.
    
    Uses a semaphore to limit concurrent requests and maximize throughput.
    """
    score_fn = get_score_fn()
    semaphore = asyncio.Semaphore(max_concurrent_requests)
    
    async def process_one_record(session: "aiohttp.ClientSession", record: Dict[str, Any]) -> bool:
        """Process a single record with semaphore-controlled concurrency."""
        async with semaphore:
            sol_list = await _openai_chat_completion_async(
                session=session,
                base_url=base_url,
                model_id=model_id,
                messages=record["prompt"],
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                seed=seed,
                n=pass_k,
            )
            gt = record["reward_model"]["ground_truth"]
            # pass@k: check if ANY of the k solutions is correct
            any_correct = any(bool(score_fn(sol, gt)) for sol in sol_list)
            return any_correct
    
    # Create aiohttp session with connection pooling
    connector = aiohttp.TCPConnector(limit=max_concurrent_requests, limit_per_host=max_concurrent_requests)
    async with aiohttp.ClientSession(connector=connector) as session:
        # Create all tasks
        tasks = [process_one_record(session, r) for r in records]
        
        # Run all tasks concurrently with simple progress tracking
        print(f"Starting {len(tasks)} async inference tasks with max_concurrent={max_concurrent_requests}...", file=sys.stderr)
        
        # Use tqdm.asyncio.tqdm.gather for proper async progress bar
        try:
            from tqdm.asyncio import tqdm as async_tqdm
            results = await async_tqdm.gather(*tasks, desc="async inference")
        except ImportError:
            # Fallback: use manual progress tracking
            results = []
            completed = 0
            pending = set(tasks)
            pbar = tqdm(total=len(tasks), desc="async inference", file=sys.stderr)
            
            while pending:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    results.append(await task)
                    completed += 1
                    pbar.update(1)
            pbar.close()
        
        return results


# ---------------------------
# Child-mode: one model per process
# ---------------------------


def _run_child_mode(args: argparse.Namespace) -> None:
    """Evaluate exactly one model and print a JSON line."""
    records = load_records_from_parquet(args.validation_parquet)
    if len(records) == 0:
        raise RuntimeError("No records loaded from parquet.")

    idx = sample_indices(len(records), args.num_samples, args.seed)
    records = [records[i] for i in idx]

    if args.use_vllm_serve:
        dp = int(args.vllm_dp_size)
        if dp <= 0:
            dp = _infer_visible_gpu_count()
        acc, _ = eval_records_with_vllm_serve(
            model_dir_or_name=args._child_model,
            tokenizer_dir_or_name=args.step0_model,
            records=records,
            batch_size=args.batch_size,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            seed=args.seed,
            host=args.vllm_host,
            port=int(args.vllm_port),
            start_timeout_s=int(args.vllm_start_timeout_s),
            stop_timeout_s=int(args.vllm_stop_timeout_s),
            tp_size=int(args.vllm_tp_size),
            dp_size=dp,
            gpu_mem_util=float(args.vllm_gpu_memory_utilization),
            max_model_len=args.vllm_max_model_len,
            pass_k=args.pass_k,
            max_concurrent_requests=args.max_concurrent_requests,
        )
    else:
        acc, _ = eval_records_with_vllm(
            model_dir_or_name=args._child_model,
            tokenizer_dir_or_name=args.step0_model,
            records=records,
            batch_size=args.batch_size,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            seed=args.seed,
            pass_k=args.pass_k,
        )

    out = {
        "label": args._child_label,
        "model": args._child_model,
        "n": len(records),
        "acc": acc,
    }
    print(json.dumps(out, ensure_ascii=False))


def _run_one_model_subprocess(
    *,
    script_path: Path,
    base_args: List[str],
    label: str,
    model: str,
) -> Dict[str, Any]:
    """Run this script in a fresh process to evaluate one model.

    This is a reliability hack: vLLM can leak GPU memory / shared_memory across repeated
    in-process initializations, causing later checkpoints to fail intermittently.
    """
    cmd = [
        sys.executable,
        str(script_path),
        *base_args,
        "--_child_model",
        model,
        "--_child_label",
        label,
    ]

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,    
        stderr=None,              
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Child eval failed for {label}.\n"
            f"Command: {' '.join(cmd)}\n\nSTDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}"
        )

    # Child prints exactly one JSON line.
    last_line = ""
    for line in proc.stdout.splitlines()[::-1]:
        line = line.strip()
        if line:
            last_line = line
            break
    if not last_line:
        raise RuntimeError(f"Child eval produced no output for {label}. STDOUT was empty.")

    return json.loads(last_line)


# ---------------------------
# Main (parent)
# ---------------------------
def _int_or_none(x: str):
    if x.lower() == "none":
        return None
    return int(x)


@dataclass
class EvalResult:
    label: str
    step: Optional[int]
    model: str
    n: int
    acc: float



# ---------------------------
# Incremental progress writing / resume helpers
# ---------------------------

def _atomic_write_text(path: Path, text: str) -> None:
    """Write text to `path` atomically (best-effort) to avoid partial files on crash."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def _build_out_obj(args: argparse.Namespace, results: List["EvalResult"]) -> Dict[str, Any]:
    return {
        "pass_k": args.pass_k,
        "args": {k: v for k, v in vars(args).items() if not k.startswith("_")},
        "env": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "VLLM_GPU_MEMORY_UTILIZATION": os.environ.get("VLLM_GPU_MEMORY_UTILIZATION"),
        },
        "results": [asdict(r) for r in results],
        "last_update_unix": time.time(),
    }


def _flush_progress(args: argparse.Namespace, results: List["EvalResult"]) -> None:
    out_path = Path(args.output_json)
    out_obj = _build_out_obj(args, results)
    _atomic_write_text(out_path, json.dumps(out_obj, indent=2, ensure_ascii=False))
    print(f"[progress] wrote: {out_path.resolve()}")


def _load_existing_results(args: argparse.Namespace) -> Tuple[List["EvalResult"], set]:
    """Load existing output_json (if present) so reruns can skip completed checkpoints."""
    out_path = Path(args.output_json)
    if not out_path.exists():
        return [], set()

    try:
        obj = json.loads(out_path.read_text())
        prev: List[EvalResult] = []
        done = set()
        for r in obj.get("results", []):
            er = EvalResult(
                label=r["label"],
                step=r.get("step", None),
                model=r["model"],
                n=int(r["n"]),
                acc=float(r["acc"]),
            )
            prev.append(er)
            done.add(er.label)
        print(f"[resume] loaded {len(prev)} results from {out_path}")
        return prev, done
    except Exception as e:
        print(f"[resume] warning: failed to read existing {out_path}: {e}")
        return [], set()


def main() -> None:
    p = argparse.ArgumentParser()

    p.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="Experiment directory that directly contains global_step_* folders.",
    )
    p.add_argument(
        "--step0-model",
        type=str,
        required=True,
        help='Step-0 HuggingFace model name, e.g. "Qwen/Qwen3-0.6B".',
    )
    p.add_argument(
        "--validation-parquet",
        type=str,
        required=True,
        help="Path to validation parquet.",
    )
    p.add_argument(
        "--num-samples",
        type=int,
        default=0,
        help="Randomly evaluate N samples (0 means evaluate all records).",
    )
    p.add_argument("--seed", type=int, default=1)

    # vLLM sampling
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument(
        "--pass-k",
        type=int,
        default=1,
        help="Generate k solutions per problem for pass@k evaluation.",
    )

    # vLLM serve (OpenAI API) controls
    p.add_argument(
        "--use-vllm-serve",
        action="store_true",
        default=True,
        help="Evaluate via `vllm serve` (OpenAI API). Default: enabled.",
    )
    p.add_argument(
        "--no-vllm-serve",
        dest="use_vllm_serve",
        action="store_false",
        help="Disable `vllm serve` and use in-process vLLM Python API.",
    )
    p.add_argument("--vllm-host", type=str, default="127.0.0.1")
    p.add_argument(
        "--vllm-port",
        type=int,
        default=0,
        help="Port for vLLM serve. 0 means auto-pick a free port.",
    )
    p.add_argument(
        "--vllm-start-timeout-s",
        type=int,
        default=300,
        help="Seconds to wait for vLLM serve to become ready.",
    )
    p.add_argument(
        "--vllm-stop-timeout-s",
        type=int,
        default=15,
        help="Seconds to wait for vLLM serve to stop before SIGKILL.",
    )
    p.add_argument(
        "--vllm-tp-size",
        type=int,
        default=1,
        help="Tensor parallel size for vLLM serve. Recommended: 1.",
    )
    p.add_argument(
        "--vllm-dp-size",
        type=int,
        default=0,
        help="Data parallel size for vLLM serve. 0 means auto=number of visible GPUs.",
    )
    p.add_argument(
        "--vllm-gpu-memory-utilization",
        type=float,
        default=float(os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.75")),
        help="GPU memory utilization for vLLM serve.",
    )

    p.add_argument(
        "--vllm-max-model-len",
        type=_int_or_none,
        default=None,
        help="Max model length for vLLM. Use 'None' to let vLLM decide.",
    )
    p.add_argument(
        "--max-concurrent-requests",
        type=int,
        default=64,
        help="Max concurrent requests to vLLM serve (for async mode). Higher = more throughput but more memory.",
    )
    # outputs
    p.add_argument(
        "--output-json",
        type=str,
        default="eval_results.json",
        help="Write a summarized JSON with all results.",
    )
    p.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="W&B project name for logging results. If not set, W&B logging is disabled.",
    )
    p.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="W&B run name. If not set, uses default W&B naming.",
    )
    p.add_argument(
        "--wandb-run-id",
        type=str,
        default=None,
        help="W&B run ID for resuming existing run. If not set, creates new run.",
    )

    # Hidden child-mode flags (NOT shown to user)
    p.add_argument("--_child_model", type=str, default=None, help=argparse.SUPPRESS)
    p.add_argument("--_child_label", type=str, default=None, help=argparse.SUPPRESS)

    args = p.parse_args()

    # Child mode runs exactly one model and exits.
    if args._child_model is not None:
        if args._child_label is None:
            raise ValueError("--_child_label is required when --_child_model is set")
        _run_child_mode(args)
        return

    exp_dir = Path(args.checkpoint_dir)
    ckpts = scan_exp_dir(exp_dir)
    if len(ckpts) == 0:
        raise RuntimeError(
            f"No global_step_* folders found under: {exp_dir}. "
            "Make sure --checkpoint-dir is the parent directory of global_step_XX folders."
        )

    overwrite_merge = os.environ.get("VERL_OVERWRITE_MERGE", "0") == "1"

    # Convert/merge checkpoints as needed (progress bar).
    merged_dirs: Dict[int, Path] = {}
    for c in tqdm(ckpts, desc="merge fsdp->hf", total=len(ckpts)):
        out_dir = merged_hf_dir_for_step(exp_dir, c.step)
        run_verl_merge_fsdp_to_hf(
            local_dir=c.actor_dir,
            out_dir=out_dir,
            python_exec=sys.executable,
            overwrite=overwrite_merge,
        )
        merged_dirs[c.step] = out_dir

    # Common args passed to child processes.
    # Child will re-load parquet and select the same subset deterministically.
    base_args = [
        "--checkpoint-dir",
        str(exp_dir),
        "--step0-model",
        args.step0_model,
        "--validation-parquet",
        args.validation_parquet,
        "--num-samples",
        str(args.num_samples),
        "--seed",
        str(args.seed),
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--batch-size",
        str(args.batch_size),
        "--pass-k",
        str(args.pass_k),

        # vLLM serve settings (passed through to child)
        "--vllm-host",
        args.vllm_host,
        "--vllm-port",
        str(args.vllm_port),
        "--vllm-start-timeout-s",
        str(args.vllm_start_timeout_s),
        "--vllm-stop-timeout-s",
        str(args.vllm_stop_timeout_s),
        "--vllm-tp-size",
        str(args.vllm_tp_size),
        "--vllm-dp-size",
        str(args.vllm_dp_size),
        "--vllm-gpu-memory-utilization",
        str(args.vllm_gpu_memory_utilization),
        "--vllm-max-model-len",
        str(args.vllm_max_model_len),
        "--max-concurrent-requests",
        str(args.max_concurrent_requests),
    ]
    if args.max_tokens is not None:
        base_args.extend(["--max-tokens", str(args.max_tokens)])

    # Pass serve toggle to child.
    if not args.use_vllm_serve:
        base_args.append("--no-vllm-serve")

    script_path = Path(__file__).resolve()

    # Resume support: if output_json already exists, load it and skip completed checkpoints.
    results, done_labels = _load_existing_results(args)

    # Initialize W&B early so we can log per-checkpoint progress (and resume if run_id is fixed).
    wandb_run = None
    if args.wandb_project:
        try:
            import wandb  # type: ignore

            wandb_kwargs = {"project": args.wandb_project}
            if args.wandb_run_name:
                wandb_kwargs["name"] = args.wandb_run_name
            if args.wandb_run_id:
                wandb_kwargs["id"] = args.wandb_run_id
                wandb_kwargs["resume"] = "allow"

            wandb_run = wandb.init(**wandb_kwargs)

            # Create a *new* eval namespace that uses our own step axis.
            # This preserves any historical charts that use W&B's default internal Step.
            # New eval metrics will live under `eval/*` and use `eval/global_step` as X.
            # We log with `step=<global_step>` below so W&B's internal Step matches training steps.
            # Keeping this define lets you select `eval/global_step` as X-axis, but we
            # intentionally do NOT force all metrics to use it (forcing breaks old history).
            wandb.define_metric("eval/global_step")

            wandb.config.update({
                "pass_k": args.pass_k,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_tokens": args.max_tokens,
                "batch_size": args.batch_size,
                "num_samples": args.num_samples,
                "seed": args.seed,
                "step0_model": args.step0_model,
            })
        except ImportError:
            print("\nWarning: wandb not installed, skipping W&B logging")
            wandb_run = None
        except Exception as e:
            print(f"\nWarning: W&B init failed: {e}")
            wandb_run = None

    # Best-effort: flush progress on Ctrl+C / SIGTERM so we can resume.
    def _on_signal(signum, frame):
        print(f"\n[signal] got {signum}, flushing progress then exiting...")
        try:
            _flush_progress(args, results)
        finally:
            raise SystemExit(128 + int(signum))

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    if "step0" in done_labels:
        print("\n=== Evaluating step0 model ===")
        print("[resume] skip step0 (already done)")
    else:
        print("\n=== Evaluating step0 model ===")
        r0 = _run_one_model_subprocess(
            script_path=script_path,
            base_args=base_args,
            label="step0",
            model=args.step0_model,
        )
        r0_er = EvalResult(label="step0", step=None, model=args.step0_model, n=r0["n"], acc=r0["acc"])
        results.append(r0_er)
        _flush_progress(args, results)

        if wandb_run is not None:
            import wandb  # type: ignore
            wandb.log({
                f"eval/pass@{args.pass_k}/accuracy": r0_er.acc,
                f"eval/pass@{args.pass_k}/n_samples": r0_er.n,
                "eval/global_step": 0,
                "global_step": 0,
                "eval/label": r0_er.label,
                "eval/model": r0_er.model,
            }, step=0)

    print("\n=== Evaluating checkpoints ===")
    for c in ckpts:
        label = f"global_step_{c.step}"
        if label in done_labels:
            print(f"\n--- {label} ---")
            print(f"[resume] skip {label} (already done)")
            continue

        print(f"\n--- {label} ---")
        r = _run_one_model_subprocess(
            script_path=script_path,
            base_args=base_args,
            label=label,
            model=str(merged_dirs[c.step]),
        )
        er = EvalResult(label=label, step=c.step, model=str(merged_dirs[c.step]), n=r["n"], acc=r["acc"])
        results.append(er)
        _flush_progress(args, results)

        if wandb_run is not None:
            import wandb  # type: ignore
            gs = int(c.step)
            wandb.log({
                f"eval/pass@{args.pass_k}/accuracy": er.acc,
                f"eval/pass@{args.pass_k}/n_samples": er.n,
                "eval/global_step": gs,
                "global_step": gs,
                "eval/label": er.label,
                "eval/model": er.model,
            }, step=gs)

    # Print compact summary
    print("\n=== Summary (accuracy) ===")

    def _key(x: EvalResult):
        return (-1 if x.step is None else x.step)

    for r in sorted(results, key=_key):
        step_str = "step0" if r.step is None else str(r.step)
        print(f"{step_str:>6}  acc={r.acc:.6f}  n={r.n}  model={r.model}")


    # Finalize W&B logging (table + finish) if requested.
    if wandb_run is not None:
        try:
            import wandb  # type: ignore

            def _key(x: EvalResult):
                return (-1 if x.step is None else x.step)

            table_data = []
            for r in sorted(results, key=_key):
                step_str = "step0" if r.step is None else f"step_{r.step}"
                table_data.append([step_str, r.acc, r.n, r.model])

            table = wandb.Table(
                columns=["checkpoint", f"pass@{args.pass_k}_acc", "n_samples", "model_path"],
                data=table_data,
            )
            wandb.log({f"pass@{args.pass_k}_results": table})
        except Exception as e:
            print(f"\nWarning: W&B final table logging failed: {e}")
        finally:
            try:
                import wandb  # type: ignore
                wandb.finish()
            except Exception:
                pass
            print(f"\nLogged results to W&B project: {args.wandb_project}")
    # Final flush (we also flush after each checkpoint).
    _flush_progress(args, results)


if __name__ == "__main__":
    main()
