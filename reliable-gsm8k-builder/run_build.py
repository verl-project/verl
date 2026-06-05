#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from reliable_gsm8k.pipeline import build_run, detect_visible_gpu_ids, merge_run_shards, should_enable_multi_gpu


def _spawn_multi_gpu_workers(args: argparse.Namespace, *, script_path: Path, gpu_ids: list[str]) -> None:
    processes: list[tuple[int, str, subprocess.Popen[str]]] = []
    num_workers = len(gpu_ids)
    try:
        for worker_index, gpu_id in enumerate(gpu_ids):
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu_id
            cmd = [
                sys.executable,
                str(script_path),
                "--run-id",
                args.run_id,
                "--split",
                args.split,
                "--output-root",
                str(Path(args.output_root).expanduser()),
                "--generator-profile",
                args.generator_profile,
                "--inference-profile",
                args.inference_profile,
                "--judge-max-tokens",
                str(args.judge_max_tokens),
                "--incorrect-target-count",
                str(args.incorrect_target_count),
                "--incorrect-max-attempts",
                str(args.incorrect_max_attempts),
                "--correct-max-attempts",
                str(args.correct_max_attempts),
                "--seed",
                str(args.seed),
                "--worker-index",
                str(worker_index),
                "--num-workers",
                str(num_workers),
                "--gpu-id",
                str(gpu_id),
                "--disable-multi-gpu",
            ]
            if args.generator_model_path is not None:
                cmd.extend(["--generator-model-path", args.generator_model_path])
            if args.judge_profile is not None:
                cmd.extend(["--judge-profile", args.judge_profile])
            if args.use_judge:
                cmd.append("--use-judge")
            if args.effective_max_items is not None:
                cmd.extend(["--max-items", str(args.effective_max_items)])
            print(
                f"[reliable-build] launching worker={worker_index}/{num_workers} gpu={gpu_id} cmd={' '.join(cmd)}",
                flush=True,
            )
            processes.append((worker_index, gpu_id, subprocess.Popen(cmd, env=env)))

        failures: list[tuple[int, str, int]] = []
        for worker_index, gpu_id, process in processes:
            return_code = process.wait()
            if return_code != 0:
                failures.append((worker_index, gpu_id, return_code))
        if failures:
            raise RuntimeError(f"one or more multi-GPU workers failed: {failures}")
    finally:
        for _, _, process in processes:
            if process.poll() is None:
                process.terminate()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build simple reliable GSM8K OE/MC datasets.")
    parser.add_argument("--run-id", required=True, help="Run identifier.")
    parser.add_argument("--split", default="train", choices=["train", "test"], help="GSM8K split.")
    parser.add_argument("--max-items", type=int, default=None, help="Optional item cap for smoke runs.")
    parser.add_argument("--num-samples", type=int, default=None, help="Alias for --max-items when you want to run only a small sample.")
    parser.add_argument("--output-root", default=str(SCRIPT_DIR / "runs"), help="Directory that will contain run outputs.")
    parser.add_argument("--generator-profile", default="qwen25_3b", help="Built-in generator profile name.")
    parser.add_argument(
        "--generator-model-path",
        default=None,
        help="Optional model/checkpoint path overriding the selected generator profile's model/tokenizer.",
    )
    parser.add_argument("--inference-profile", default="greedy", help="Built-in inference profile name.")
    parser.add_argument("--judge-profile", default="judgelm_7b", help="Built-in judge profile name if --use-judge is enabled.")
    parser.add_argument("--use-judge", action="store_true", help="Use a judge model to verify generated numeric answers. Disabled by default.")
    parser.add_argument(
        "--generator-temperature",
        type=float,
        default=None,
        help="Deprecated: decoding is now controlled by --inference-profile only.",
    )
    parser.add_argument(
        "--generator-max-tokens",
        type=int,
        default=None,
        help="Deprecated: decoding is now controlled by --inference-profile only.",
    )
    parser.add_argument("--judge-max-tokens", type=int, default=192, help="Max judge tokens.")
    parser.add_argument("--incorrect-target-count", type=int, default=4, help="Required accepted incorrect solutions.")
    parser.add_argument("--incorrect-max-attempts", type=int, default=24, help="Max generation attempts for incorrect solutions.")
    parser.add_argument("--correct-max-attempts", type=int, default=8, help="Max generation attempts for the correct solution.")
    parser.add_argument("--seed", type=int, default=7, help="Base random seed for option shuffling and generation.")
    parser.add_argument("--gpu-ids", default=None, help="Optional comma-separated GPU ids to use for auto multi-GPU sharding.")
    parser.add_argument("--disable-multi-gpu", action="store_true", help="Force single-process execution.")
    parser.add_argument("--worker-index", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--num-workers", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--gpu-id", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.max_items is not None and args.num_samples is not None and args.max_items != args.num_samples:
        parser.error("--max-items and --num-samples were both set with different values")
    if args.generator_temperature is not None or args.generator_max_tokens is not None:
        parser.error(
            "--generator-temperature/--generator-max-tokens are disabled. "
            "Use --inference-profile to control decoding."
        )
    args.effective_max_items = args.num_samples if args.num_samples is not None else args.max_items

    output_root = Path(args.output_root).expanduser()
    is_worker = args.worker_index is not None
    visible_gpu_ids = detect_visible_gpu_ids(args.gpu_ids)

    if not is_worker and not args.disable_multi_gpu and should_enable_multi_gpu(
        generator_profile_name=args.generator_profile,
        judge_profile_name=args.judge_profile,
        gpu_ids=visible_gpu_ids,
        use_judge=args.use_judge,
    ):
        print(
            f"[reliable-build] multi-gpu mode enabled with gpu_ids={visible_gpu_ids}",
            flush=True,
        )
        _spawn_multi_gpu_workers(args, script_path=Path(__file__).resolve(), gpu_ids=visible_gpu_ids)
        manifest = merge_run_shards(
            run_id=args.run_id,
            split=args.split,
            output_root=output_root,
            generator_profile_name=args.generator_profile,
            inference_profile_name=args.inference_profile,
            judge_profile_name=args.judge_profile,
            use_judge=args.use_judge,
            num_workers=len(visible_gpu_ids),
            gpu_ids=visible_gpu_ids,
            max_items=args.effective_max_items,
        )
        print(
            f"[reliable-build] merged {len(visible_gpu_ids)} shard(s) into {output_root / args.run_id} "
            f"record_counts={manifest.get('record_counts', {})}",
            flush=True,
        )
        return

    build_run(
        run_id=args.run_id,
        split=args.split,
        max_items=args.effective_max_items,
        output_root=output_root,
        generator_profile_name=args.generator_profile,
        generator_model_path=args.generator_model_path,
        inference_profile_name=args.inference_profile,
        judge_profile_name=args.judge_profile,
        judge_max_tokens=args.judge_max_tokens,
        incorrect_target_count=args.incorrect_target_count,
        incorrect_max_attempts=args.incorrect_max_attempts,
        correct_max_attempts=args.correct_max_attempts,
        seed=args.seed,
        use_judge=args.use_judge,
        worker_index=args.worker_index,
        num_workers=args.num_workers,
        gpu_id=args.gpu_id,
    )


if __name__ == "__main__":
    main()
