# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Rollout Benchmark — benchmark vLLM rollout performance with verl's standalone
init flow inside Ray. Sweeps over (gen_batch_size, rollout_n) configurations.

Usage:
    python3 tests/workers/rollout/perf/vllm_standalone_rollout_bench.py \\
        --model_path /path/to/model --tp_size 8 ...

Profile a single step:
    ... --profile_step 0 --profile_output /tmp/profile --enforce_eager
"""

import argparse
import asyncio
import json
import random
import statistics
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import ray


@dataclass
class RequestResult:
    success: bool = False
    prompt_len: int = 0
    output_tokens: int = 0
    e2e_latency: float = 0.0
    ttft: float = 0.0
    tpot: float = 0.0
    stop_reason: str = ""
    prompt_ids: list = field(default_factory=list)
    output_ids: list = field(default_factory=list)


def parse_args():
    parser = argparse.ArgumentParser(description="Rollout Benchmark")

    parser.add_argument("--model_path", type=str, required=True, help="Path to model weights")
    parser.add_argument("--tp_size", type=int, default=8, help="Tensor parallel size")
    parser.add_argument("--gpus_per_node", type=int, default=8, help="GPUs per node")
    parser.add_argument(
        "--quantization",
        type=str,
        default=None,
        help="vLLM quantization method (e.g. fp8, awq, gptq, experts_int8, torchao, None for no quantization)",
    )
    parser.add_argument("--load_format", type=str, default="auto", help="Weight load format")

    parser.add_argument("--gen_batch_sizes", type=str, default="128", help="Comma-separated batch sizes")
    parser.add_argument("--rollout_ns", type=str, default="1", help="Comma-separated rollout.n values")

    parser.add_argument("--data_source", type=str, default="random", help="'random' or path to parquet")
    parser.add_argument("--prompt_length", type=int, default=512, help="Max prompt token length")
    parser.add_argument("--response_length", type=int, default=3072, help="Max response token length")

    parser.add_argument("--num_warmup_steps", type=int, default=1, help="Warmup batches before timing")
    parser.add_argument("--num_bench_steps", type=int, default=2, help="Timed benchmark batches")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85, help="vLLM GPU memory fraction")
    parser.add_argument("--max_model_len", type=int, default=65536, help="Max model context length")
    parser.add_argument("--max_num_seqs", type=int, default=1024, help="Max concurrent sequences")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--enforce_eager", action="store_true", help="Disable CUDA graph")
    parser.add_argument(
        "--cudagraph_capture_sizes",
        type=str,
        default=None,
        help="Comma-separated CUDA graph capture batch sizes, e.g. 1,2,4,8,16,32",
    )
    parser.add_argument(
        "--cudagraph_mode",
        type=str,
        default="FULL_AND_PIECEWISE",
        help="CUDA graph compilation mode (e.g. FULL_AND_PIECEWISE, PIECEWISE, NONE)",
    )
    parser.add_argument(
        "--attention_backend",
        type=str,
        default=None,
        help="vLLM attention backend (e.g. FLASH_ATTN, FLASHINFER, FLASHMLA, FLASHINFER_MLA)",
    )
    parser.add_argument(
        "--kv_cache_dtype",
        type=str,
        default="auto",
        help="KV cache data type (auto, bfloat16, fp8, fp8_e4m3, fp8_e5m2, fp8_inc, fp8_ds_mla)",
    )
    parser.add_argument(
        "--calculate_kv_scales",
        action="store_true",
        help="Enable dynamic calculation of k_scale and v_scale for fp8 KV cache",
    )

    parser.add_argument(
        "--profile_step", type=int, default=None, help="Bench step index to profile (applies to every sweep config)"
    )
    parser.add_argument(
        "--profile_output",
        type=str,
        default="/tmp/bench_rollout_profile",
        help="Directory for torch.profiler trace output",
    )

    parser.add_argument("--output_json", type=str, default=None, help="Directory to write benchmark results")
    parser.add_argument(
        "--save_generations",
        action="store_true",
        help="Save model generations as JSONL alongside benchmark results (requires --output_json)",
    )

    return parser.parse_args()


def build_config(args):
    """Build verl config via Hydra compose with benchmark overrides."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    project_root = Path(__file__).resolve()
    while not (project_root / "verl").is_dir() and project_root.parent != project_root:
        project_root = project_root.parent
    config_dir = str(project_root / "verl" / "trainer" / "config")

    with initialize_config_dir(config_dir=config_dir, version_base=None):
        config = compose(config_name="ppo_trainer")

    config.actor_rollout_ref.model.path = args.model_path

    rollout = config.actor_rollout_ref.rollout
    rollout.name = "vllm"
    rollout.mode = "async"
    rollout.tensor_model_parallel_size = args.tp_size
    rollout.load_format = args.load_format
    rollout.gpu_memory_utilization = args.gpu_memory_utilization
    rollout.prompt_length = args.prompt_length
    rollout.response_length = args.response_length
    rollout.temperature = args.temperature
    rollout.enforce_eager = args.enforce_eager
    rollout.enable_chunked_prefill = True
    rollout.enable_prefix_caching = True
    rollout.skip_tokenizer_init = False
    rollout.free_cache_engine = False
    rollout.max_num_seqs = args.max_num_seqs
    rollout.disable_log_stats = False
    rollout.max_model_len = args.max_model_len

    # Quantization: bypass whitelist by passing through engine_kwargs.vllm
    # which gets **-unpacked last in launch_server() and overrides the args dict.
    quant_value = args.quantization if args.quantization else None
    OmegaConf.update(config, "actor_rollout_ref.rollout.quantization", None, force_add=True)
    OmegaConf.update(config, "actor_rollout_ref.rollout.enable_sleep_mode", False, force_add=True)

    if quant_value is not None:
        OmegaConf.update(
            config,
            "actor_rollout_ref.rollout.engine_kwargs.vllm.quantization",
            quant_value,
            force_add=True,
        )
        OmegaConf.update(
            config,
            "actor_rollout_ref.rollout.engine_kwargs.vllm.hf_overrides",
            {"quantization_config": {"quant_method": quant_value}},
            force_add=True,
        )

    compilation = {"cudagraph_mode": args.cudagraph_mode}
    if args.cudagraph_capture_sizes:
        compilation["cudagraph_capture_sizes"] = [int(x) for x in args.cudagraph_capture_sizes.split(",")]
    OmegaConf.update(
        config,
        "actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config",
        compilation,
        force_add=True,
    )

    if args.attention_backend:
        OmegaConf.update(
            config,
            "actor_rollout_ref.rollout.engine_kwargs.vllm.attention_config",
            {"backend": args.attention_backend},
            force_add=True,
        )

    if args.kv_cache_dtype and args.kv_cache_dtype != "auto":
        OmegaConf.update(
            config,
            "actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_dtype",
            args.kv_cache_dtype,
            force_add=True,
        )
    if args.calculate_kv_scales:
        OmegaConf.update(
            config,
            "actor_rollout_ref.rollout.engine_kwargs.vllm.calculate_kv_scales",
            True,
            force_add=True,
        )

    # Profiler config — VLLM_TORCH_PROFILER_DIR is propagated via ray.init() in async_main()
    if args.profile_step is not None:
        profiler_cfg = {
            "tool": "torch",
            "enable": True,
            "all_ranks": True,
            "ranks": [],
            "save_path": args.profile_output,
            "tool_config": {
                "torch": {
                    "contents": ["cuda", "cpu"],
                    "discrete": True,
                    "name": "torch",
                },
            },
        }
        OmegaConf.update(config, "actor_rollout_ref.rollout.profiler", profiler_cfg, force_add=True)

    config.trainer.n_gpus_per_node = args.gpus_per_node
    config.trainer.nnodes = 1

    config.data.max_prompt_length = args.prompt_length
    config.data.max_response_length = args.response_length
    config.data.train_batch_size = max(int(x) for x in args.gen_batch_sizes.split(","))

    return config


async def create_rollout_servers(config, args):
    """Create standalone vLLM rollout replicas via verl's init flow."""
    from verl.workers.rollout.replica import get_rollout_replica_class

    rollout_config = config.actor_rollout_ref.rollout
    model_config = config.actor_rollout_ref.model
    tp_size = rollout_config.tensor_model_parallel_size
    gpus_per_node = config.trainer.n_gpus_per_node
    num_replicas = gpus_per_node // tp_size

    print(f"Creating {num_replicas} standalone vLLM replica(s) (TP={tp_size}, GPUs={gpus_per_node})")

    rollout_server_class = get_rollout_replica_class("vllm")
    rollout_servers = [
        rollout_server_class(
            replica_rank=rank,
            config=rollout_config,
            model_config=model_config,
            gpus_per_node=gpus_per_node,
        )
        for rank in range(num_replicas)
    ]

    print("Initializing standalone rollout servers (loading weights + quantization)...")
    await asyncio.gather(*[server.init_standalone() for server in rollout_servers])

    server_handles = [server.server_handle for server in rollout_servers]
    server_addresses = [server.server_address for server in rollout_servers]
    print(f"Rollout servers ready at: {server_addresses}")
    return rollout_servers, server_handles


def generate_random_prompts(tokenizer, num_prompts, prompt_length):
    """Generate random token ID sequences, all exactly prompt_length tokens."""
    vocab_size = tokenizer.vocab_size
    min_token_id = 100
    max_token_id = min(vocab_size - 1, 150000)

    prompts = []
    for _ in range(num_prompts):
        prompt_ids = [random.randint(min_token_id, max_token_id) for _ in range(prompt_length)]
        prompts.append(prompt_ids)
    return prompts


def load_dataset_prompts(config, tokenizer, data_file, max_prompts):
    """Load prompts from a parquet dataset, tokenize with apply_chat_template."""
    import datasets as hf_datasets

    prompt_key = config.data.get("prompt_key", "prompt")
    max_prompt_length = config.data.max_prompt_length

    dataset = hf_datasets.load_dataset("parquet", data_files=data_file)["train"]
    print(f"Dataset has {len(dataset)} rows, columns: {dataset.column_names}")

    if prompt_key not in dataset.column_names:
        raise ValueError(f"Parquet file must contain a '{prompt_key}' column, got columns: {dataset.column_names}")

    all_prompt_ids = []
    for row in dataset:
        if len(all_prompt_ids) >= max_prompts:
            break
        try:
            messages = row[prompt_key]
            if isinstance(messages, list):
                prompt_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            elif isinstance(messages, str):
                prompt_ids = tokenizer.encode(messages)
            else:
                print(f"Warning: unexpected type for '{prompt_key}': {type(messages)}, skipping")
                continue
            prompt_ids = prompt_ids[:max_prompt_length]
            all_prompt_ids.append(prompt_ids)
        except Exception as e:
            print(f"Warning: failed to tokenize row: {e}")
            continue

    all_prompt_ids = all_prompt_ids[:max_prompts]
    if not all_prompt_ids:
        raise ValueError(f"No valid prompts loaded from {data_file}")
    print(
        f"Loaded {len(all_prompt_ids)} prompts from dataset "
        f"(avg length: {sum(len(p) for p in all_prompt_ids) / len(all_prompt_ids):.0f} tokens)"
    )
    return all_prompt_ids


def make_batches(prompts, batch_size):
    """Split prompts into batches, cycling if fewer prompts than batch_size."""
    batches = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        if len(batch) == batch_size:
            batches.append(batch)
    if not batches:
        cycled = [prompts[i % len(prompts)] for i in range(batch_size)]
        batches.append(cycled)
    return batches


async def _generate_one(handle, prompt_ids, sampling_params, request_id, save_ids=False):
    """Submit one generation request and return a RequestResult with timing."""
    t_start = time.perf_counter()
    token_output = await handle.generate.remote(
        prompt_ids=prompt_ids,
        sampling_params=dict(sampling_params),
        request_id=request_id,
    )
    t_end = time.perf_counter()

    e2e_latency = t_end - t_start
    output_tokens = len(token_output.token_ids)
    success = token_output.stop_reason != "aborted"

    extra = token_output.extra_info or {}
    ftl = extra.get("first_token_latency")
    ttft = ftl if ftl and ftl > 0 else 0.0

    first_ts = extra.get("first_token_ts")
    last_ts = extra.get("last_token_ts")
    tpot = 0.0
    if output_tokens > 1 and first_ts is not None and last_ts is not None and first_ts > 0 and last_ts > first_ts:
        tpot = (last_ts - first_ts) / (output_tokens - 1)

    return RequestResult(
        success=success,
        prompt_len=len(prompt_ids),
        output_tokens=output_tokens,
        e2e_latency=e2e_latency,
        ttft=ttft,
        tpot=tpot,
        stop_reason=token_output.stop_reason or "",
        prompt_ids=prompt_ids if save_ids else [],
        output_ids=list(token_output.token_ids) if save_ids else [],
    )


async def run_single_batch(server_handles, prompt_ids_batch, sampling_params, rollout_n, save_ids=False):
    """Submit gen_batch_size * rollout_n requests, round-robin across handles."""
    coros = []
    num_handles = len(server_handles)
    idx = 0

    for prompt_ids in prompt_ids_batch:
        for _ in range(rollout_n):
            request_id = uuid.uuid4().hex
            handle = server_handles[idx % num_handles]
            coros.append(_generate_one(handle, prompt_ids, sampling_params, request_id, save_ids))
            idx += 1

    results = await asyncio.gather(*coros)
    return list(results)


def _percentiles(values, pcts=(50, 90, 99)):
    if not values:
        return {f"p{p}": 0.0 for p in pcts}
    arr = np.array(values)
    return {f"p{p}": float(np.percentile(arr, p)) for p in pcts}


def _basic_stats(values):
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


async def run_benchmark_config(server_handles, rollout_servers, prompts, args, gen_batch_size, rollout_n):
    """Run benchmark for a single (gen_batch_size, rollout_n) configuration."""
    sampling_params = {
        "temperature": args.temperature,
        "top_k": -1,
        "top_p": 1.0,
        "max_tokens": args.response_length,
    }
    if args.data_source == "random":
        sampling_params["min_tokens"] = args.response_length
        sampling_params["ignore_eos"] = True

    # Warmup with small batch + short max_tokens
    warmup_size = 16
    warmup_batch = [prompts[i % len(prompts)] for i in range(warmup_size)]
    warmup_params = {**sampling_params, "max_tokens": 128, "min_tokens": 0, "ignore_eos": False}
    for warmup_idx in range(args.num_warmup_steps):
        print(f"  [WARMUP] step={warmup_idx}, requests={warmup_size * rollout_n}")
        await run_single_batch(server_handles, warmup_batch, warmup_params, rollout_n)

    # Clear KV cache so prefix cache from warmup doesn't skew the first bench step
    for server in rollout_servers:
        await server.clear_kv_cache()

    batches = make_batches(prompts, gen_batch_size)
    step_results = []
    all_request_results = []
    save_gens = args.save_generations

    for step_idx in range(args.num_bench_steps):
        batch = batches[step_idx % len(batches)]

        profiling_this_step = args.profile_step is not None and step_idx == args.profile_step
        if profiling_this_step:
            print(f"  [PROFILE] Starting torch.profiler for step {step_idx}")
            for server in rollout_servers:
                await server.start_profile()

        t_batch_start = time.perf_counter()
        request_results = await run_single_batch(server_handles, batch, sampling_params, rollout_n, save_gens)
        t_batch_end = time.perf_counter()

        if profiling_this_step:
            for server in rollout_servers:
                await server.stop_profile()
            print(f"  [PROFILE] Trace saved to {args.profile_output}")

        batch_elapsed = t_batch_end - t_batch_start

        num_requests = len(request_results)
        num_completed = sum(1 for r in request_results if r.success)
        num_aborted = num_requests - num_completed

        total_output_tokens = sum(r.output_tokens for r in request_results)
        total_input_tokens = sum(r.prompt_len for r in request_results)
        total_tokens = total_output_tokens + total_input_tokens

        output_tps = total_output_tokens / batch_elapsed if batch_elapsed > 0 else 0
        total_tps = total_tokens / batch_elapsed if batch_elapsed > 0 else 0
        rps = num_requests / batch_elapsed if batch_elapsed > 0 else 0

        completed = [r for r in request_results if r.success]
        e2e_latencies = [r.e2e_latency for r in completed]
        ttfts = [r.ttft for r in completed if r.ttft > 0]
        tpots = [r.tpot for r in completed if r.tpot > 0]

        has_engine_ttft = bool(ttfts)
        has_engine_tpot = bool(tpots)

        ttft_stats = _basic_stats(ttfts) if has_engine_ttft else None
        ttft_pcts = _percentiles(ttfts) if has_engine_ttft else None
        tpot_stats = _basic_stats(tpots) if has_engine_tpot else None
        tpot_pcts = _percentiles(tpots) if has_engine_tpot else None

        print(
            f"  [BENCH] step={step_idx}, "
            f"requests={num_requests}, "
            f"output_tokens={total_output_tokens}, "
            f"batch_time={batch_elapsed:.2f}s, "
            f"output_tok/s={output_tps:.0f}, "
            f"total_tok/s={total_tps:.0f}, "
            f"req/s={rps:.1f}, "
            f"completed={num_completed}, aborted={num_aborted}"
        )
        if has_engine_ttft:
            print(
                f"         TTFT(s): mean={ttft_stats['mean']:.4f}, "
                f"p50={ttft_pcts['p50']:.4f}, "
                f"p90={ttft_pcts['p90']:.4f}, "
                f"p99={ttft_pcts['p99']:.4f}"
            )
        if has_engine_tpot:
            print(
                f"         TPOT(s): mean={tpot_stats['mean']:.6f}, "
                f"p50={tpot_pcts['p50']:.6f}, "
                f"p90={tpot_pcts['p90']:.6f}, "
                f"p99={tpot_pcts['p99']:.6f}"
            )

        step_data = {
            "step": step_idx,
            "batch_elapsed_sec": batch_elapsed,
            "num_requests": num_requests,
            "num_completed": num_completed,
            "num_aborted": num_aborted,
            "total_output_tokens": total_output_tokens,
            "total_input_tokens": total_input_tokens,
            "output_tokens_per_sec": output_tps,
            "total_tokens_per_sec": total_tps,
            "requests_per_sec": rps,
            "e2e_latency": {**_basic_stats(e2e_latencies), **_percentiles(e2e_latencies)},
        }
        if has_engine_ttft:
            step_data["ttft"] = {**ttft_stats, **ttft_pcts}
        if has_engine_tpot:
            step_data["tpot"] = {**tpot_stats, **tpot_pcts}
        step_results.append(step_data)
        if save_gens:
            all_request_results.extend(request_results)

    return step_results, all_request_results


async def run_sweep(server_handles, rollout_servers, prompts, args):
    """Sweep over all (gen_batch_size, rollout_n) combinations."""
    gen_batch_sizes = [int(x) for x in args.gen_batch_sizes.split(",")]
    rollout_ns = [int(x) for x in args.rollout_ns.split(",")]

    all_results = []
    all_generations = []

    for gbs in gen_batch_sizes:
        for rn in rollout_ns:
            total_requests = gbs * rn
            print(f"\n{'=' * 70}")
            print(f"Config: gen_batch_size={gbs}, rollout_n={rn}, total_requests={total_requests}")
            print(f"{'=' * 70}")

            step_results, request_results = await run_benchmark_config(
                server_handles, rollout_servers, prompts, args, gbs, rn
            )
            all_generations.extend(request_results)

            if step_results:
                batch_times = [r["batch_elapsed_sec"] for r in step_results]
                output_tps_list = [r["output_tokens_per_sec"] for r in step_results]
                total_tps_list = [r["total_tokens_per_sec"] for r in step_results]
                rps_list = [r["requests_per_sec"] for r in step_results]
                output_tokens_list = [r["total_output_tokens"] for r in step_results]

                all_e2e = []
                all_ttft = []
                all_tpot = []
                for r in step_results:
                    all_e2e.append(r["e2e_latency"]["mean"])
                    if "ttft" in r:
                        all_ttft.append(r["ttft"]["mean"])
                    if "tpot" in r:
                        all_tpot.append(r["tpot"]["mean"])

                config_result = {
                    "gen_batch_size": gbs,
                    "rollout_n": rn,
                    "total_requests_per_step": total_requests,
                    "output_tokens": _basic_stats(output_tokens_list),
                    "output_tokens_per_sec": _basic_stats(output_tps_list),
                    "total_tokens_per_sec": _basic_stats(total_tps_list),
                    "requests_per_sec": _basic_stats(rps_list),
                    "batch_time_sec": _basic_stats(batch_times),
                    "e2e_latency_sec": _basic_stats(all_e2e),
                    "steps": step_results,
                }
                if all_ttft:
                    config_result["ttft_sec"] = _basic_stats(all_ttft)
                if all_tpot:
                    config_result["tpot_sec"] = _basic_stats(all_tpot)
                all_results.append(config_result)

    return all_results, all_generations


def print_summary(results, args):
    """Print a formatted summary table."""
    print(f"\n{'=' * 120}")
    print("Rollout Benchmark Summary")
    model_name = Path(args.model_path).name
    print(f"Model: {model_name} | TP={args.tp_size} | Quant={args.quantization} | KV={args.kv_cache_dtype}")
    print(f"Data: {args.data_source} | prompt_len={args.prompt_length} | response_len={args.response_length}")
    print(f"{'=' * 120}")

    has_ttft = any("ttft_sec" in r for r in results)
    has_tpot = any("tpot_sec" in r for r in results)

    header_parts = [
        f"{'batch':>6}",
        f"{'n':>4}",
        f"{'reqs':>6}",
        f"{'out_toks':>10}",
        f"{'batch_time':>10}",
        f"{'out_tok/s':>10}",
        f"{'tot_tok/s':>10}",
        f"{'req/s':>8}",
        f"{'e2e_lat':>8}",
    ]
    if has_ttft:
        header_parts.append(f"{'ttft':>8}")
    if has_tpot:
        header_parts.append(f"{'tpot':>10}")

    header = " | ".join(header_parts)
    print(header)
    print("-" * len(header))

    for r in results:
        row_parts = [
            f"{r['gen_batch_size']:>6}",
            f"{r['rollout_n']:>4}",
            f"{r['total_requests_per_step']:>6}",
            f"{r['output_tokens']['mean']:>10.0f}",
            f"{r['batch_time_sec']['mean']:>10.2f}",
            f"{r['output_tokens_per_sec']['mean']:>10.0f}",
            f"{r['total_tokens_per_sec']['mean']:>10.0f}",
            f"{r['requests_per_sec']['mean']:>8.1f}",
            f"{r['e2e_latency_sec']['mean']:>8.2f}",
        ]
        if has_ttft:
            ttft_val = r.get("ttft_sec", {}).get("mean")
            row_parts.append(f"{ttft_val:>8.4f}" if ttft_val is not None else f"{'N/A':>8}")
        if has_tpot:
            tpot_val = r.get("tpot_sec", {}).get("mean")
            row_parts.append(f"{tpot_val:>10.6f}" if tpot_val is not None else f"{'N/A':>10}")

        print(" | ".join(row_parts))

    print(f"{'=' * 120}")

    if results and results[-1].get("steps"):
        last = results[-1]
        last_step = last["steps"][-1]
        print(
            f"\nDetailed percentiles for last config "
            f"(batch={last['gen_batch_size']}, n={last['rollout_n']}, last step):"
        )
        e2e = last_step["e2e_latency"]
        print(
            f"  E2E latency (s): mean={e2e['mean']:.3f}, "
            f"p50={e2e['p50']:.3f}, p90={e2e['p90']:.3f}, p99={e2e['p99']:.3f}"
        )
        if "ttft" in last_step:
            t = last_step["ttft"]
            print(
                f"  TTFT (s):        mean={t['mean']:.4f}, p50={t['p50']:.4f}, p90={t['p90']:.4f}, p99={t['p99']:.4f}"
            )
        if "tpot" in last_step:
            t = last_step["tpot"]
            print(
                f"  TPOT (s):        mean={t['mean']:.6f}, p50={t['p50']:.6f}, p90={t['p90']:.6f}, p99={t['p99']:.6f}"
            )


def write_json_results(output_dir, results, args):
    output_path = Path(output_dir) / "bench_results.json"
    output = {
        "benchmark_config": {
            "model_path": args.model_path,
            "tp_size": args.tp_size,
            "gpus_per_node": args.gpus_per_node,
            "quantization": args.quantization,
            "kv_cache_dtype": args.kv_cache_dtype,
            "calculate_kv_scales": args.calculate_kv_scales,
            "load_format": args.load_format,
            "data_source": args.data_source,
            "prompt_length": args.prompt_length,
            "response_length": args.response_length,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": args.max_model_len,
            "max_num_seqs": args.max_num_seqs,
            "num_warmup_steps": args.num_warmup_steps,
            "num_bench_steps": args.num_bench_steps,
            "temperature": args.temperature,
        },
        "results": results,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults written to {output_path}")


def write_generations(output_dir, generations, tokenizer):
    """Write model generations to a JSONL file."""
    output_path = Path(output_dir) / "generations.jsonl"
    with open(output_path, "w") as f:
        for r in generations:
            entry = {
                "prompt": tokenizer.decode(r.prompt_ids, skip_special_tokens=False),
                "output_text": tokenizer.decode(r.output_ids, skip_special_tokens=False),
                "output_ids": r.output_ids,
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Generations written to {output_path} ({len(generations)} entries)")


async def async_main(args):
    if args.profile_step is not None and args.profile_step >= args.num_bench_steps:
        raise ValueError(
            f"--profile_step {args.profile_step} >= --num_bench_steps {args.num_bench_steps}, "
            f"profiling would never trigger. Use a step index in [0, {args.num_bench_steps - 1}]."
        )

    # Build output directory and profile path
    output_dir = None
    if args.output_json or args.profile_step is not None:
        from datetime import datetime

        model_name = Path(args.model_path).name
        quant_tag = args.quantization if args.quantization else "bf16"
        if args.kv_cache_dtype and args.kv_cache_dtype != "auto":
            quant_tag = f"{quant_tag}_kv{args.kv_cache_dtype}"
        run_tag = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{model_name}_{quant_tag}"

        if args.output_json:
            output_dir = Path(args.output_json) / run_tag
            output_dir.mkdir(parents=True, exist_ok=True)

        if args.profile_step is not None:
            args.profile_output = str(output_dir) if output_dir else str(Path(args.profile_output) / run_tag)

    config = build_config(args)

    # VLLM_TORCH_PROFILER_DIR is set inside the Actor process by
    # build_vllm_profiler_args() during engine launch, no need to
    # pass it via ray.init runtime_env.
    ray.init()
    print(f"Ray initialized: {ray.cluster_resources()}")

    try:
        rollout_servers, server_handles = await create_rollout_servers(config, args)

        from verl.utils import hf_tokenizer

        tokenizer = hf_tokenizer(args.model_path)

        max_batch_size = max(int(x) for x in args.gen_batch_sizes.split(","))
        num_prompts_needed = max_batch_size * 2

        if args.data_source == "random":
            print(f"\nGenerating {num_prompts_needed} random prompts (length ~{args.prompt_length} tokens)...")
            prompts = generate_random_prompts(tokenizer, num_prompts_needed, args.prompt_length)
        else:
            print(f"\nLoading prompts from dataset: {args.data_source}")
            prompts = load_dataset_prompts(config, tokenizer, args.data_source, num_prompts_needed)

        avg_prompt_len = sum(len(p) for p in prompts) / len(prompts)
        print(f"Prepared {len(prompts)} prompts (avg length: {avg_prompt_len:.0f} tokens)")

        results, generations = await run_sweep(server_handles, rollout_servers, prompts, args)
        print_summary(results, args)

        if output_dir:
            write_json_results(output_dir, results, args)
            if args.save_generations:
                write_generations(output_dir, generations, tokenizer)

    finally:
        ray.shutdown()
        print("Ray shutdown complete.")


def main():
    args = parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
