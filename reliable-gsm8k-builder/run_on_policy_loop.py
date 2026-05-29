#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent


def _run(cmd: list[str], *, cwd: Path, env: dict[str, str], dry_run: bool) -> None:
    print("[on-policy] " + " ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def _gpu_count(gpu_ids: str | None) -> int:
    if gpu_ids:
        return len([value for value in gpu_ids.split(",") if value.strip()])
    return int(os.environ.get("GPUS", "1"))


def _load_manifest(run_dir: Path) -> dict:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing generation manifest: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _dataset_key(name: str) -> str:
    mapping = {
        "oe": "oe",
        "mc_onecorrect": "mc_onecorrect",
        "mc_allwrong": "mc_allwrong",
    }
    return mapping[name]


def _generated_train_file(*, run_dir: Path, train_dataset: str) -> Path:
    manifest = _load_manifest(run_dir)
    parquet = ((manifest.get("artifacts") or {}).get("parquet") or {})
    path = parquet.get(_dataset_key(train_dataset))
    if path is None:
        raise FileNotFoundError(
            f"manifest has no parquet artifact for train_dataset={train_dataset!r}; "
            f"available={sorted(parquet)}"
        )
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"generated train parquet does not exist: {resolved}")
    return resolved


def _candidate_hf_paths(checkpoint_dir: Path) -> list[Path]:
    paths: list[Path] = []
    paths.append(checkpoint_dir / "best" / "actor" / "huggingface")
    global_steps = sorted(
        checkpoint_dir.glob("global_step_*"),
        key=lambda path: int(path.name.rsplit("_", 1)[-1]) if path.name.rsplit("_", 1)[-1].isdigit() else -1,
    )
    for step_dir in reversed(global_steps):
        paths.append(step_dir / "actor" / "huggingface")
    return paths


def _find_next_model_path(checkpoint_dir: Path) -> Path:
    for path in _candidate_hf_paths(checkpoint_dir):
        if (path / "config.json").exists():
            return path
    checked = "\n".join(str(path) for path in _candidate_hf_paths(checkpoint_dir))
    raise FileNotFoundError(f"could not find an HF actor checkpoint under {checkpoint_dir}; checked:\n{checked}")


def _train_command(
    *,
    model_path: Path | str,
    train_file: Path,
    val_file: Path,
    project_name: str,
    experiment_name: str,
    checkpoint_dir: Path,
    gpus: int,
    total_epochs: int,
    save_freq: int,
    train_batch_size: int,
    ppo_mini_batch_size: int,
    actor_micro_batch_size: int,
    rollout_n: int,
    learning_rate: float,
    test_freq: int,
    train_dataset: str,
) -> list[str]:
    train_reward_path = "verl/utils/reward_score/gsm8k.py" if train_dataset == "oe" else "verl/utils/reward_score/gsm8k_mc.py"
    return [
        sys.executable,
        "-m",
        "verl.trainer.main_ppo",
        "algorithm.adv_estimator=grpo",
        f"data.train_files={train_file}",
        f"data.val_files={val_file}",
        f"data.train_batch_size={train_batch_size}",
        "data.max_prompt_length=512",
        "data.max_response_length=512",
        f"actor_rollout_ref.model.path={model_path}",
        f"actor_rollout_ref.actor.optim.lr={learning_rate}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={ppo_mini_batch_size}",
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={actor_micro_batch_size}",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=80",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.4",
        f"actor_rollout_ref.rollout.n={rollout_n}",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=20",
        "algorithm.kl_ctrl.kl_coef=0.001",
        "trainer.val_before_train=False",
        f"trainer.n_gpus_per_node={gpus}",
        "trainer.nnodes=1",
        "trainer.resume_mode=disable",
        f"trainer.save_freq={save_freq}",
        f"trainer.test_freq={test_freq}",
        'trainer.logger=["console","wandb"]',
        "+trainer.best_ckpt_metric=val-aux/openai/gsm8k/score/mean@1",
        "+trainer.best_ckpt_mode=max",
        "+trainer.best_ckpt_keep_only=true",
        "trainer.early_stop_metric=val-aux/openai/gsm8k/score/mean@1",
        "trainer.early_stop_patience=10",
        "trainer.early_stop_mode=max",
        'actor_rollout_ref.actor.checkpoint.save_contents=["model","optimizer","extra","hf_model"]',
        f"trainer.default_local_dir={checkpoint_dir}",
        f"trainer.total_epochs={total_epochs}",
        f"trainer.project_name={project_name}",
        f"trainer.experiment_name={experiment_name}",
        f"custom_reward_function.path={train_reward_path}",
        "custom_reward_function.name=compute_score",
        "val_custom_reward_function.path=verl/utils/reward_score/gsm8k.py",
        "val_custom_reward_function.name=compute_score",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run generate -> train -> roll out checkpoint iterations.")
    parser.add_argument("--run-prefix", required=True, help="Prefix for iteration run IDs.")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--num-samples", type=int, required=True, help="Number of GSM8K questions per generation step.")
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-3B-Instruct", help="Initial model path/name.")
    parser.add_argument("--generator-profile", default="qwen25_3b")
    parser.add_argument("--inference-profile", default="sample_balanced")
    parser.add_argument("--train-dataset", choices=["mc_onecorrect", "mc_allwrong", "oe"], default="mc_onecorrect")
    parser.add_argument("--output-root", default=str(SCRIPT_DIR / "runs_on_policy"))
    parser.add_argument("--checkpoint-root", default=str(REPO_ROOT / "checkpoints" / "on_policy"))
    parser.add_argument("--val-file", default=str(Path.home() / "data/gsm8k/test.parquet"))
    parser.add_argument("--gpu-ids", default=None)
    parser.add_argument("--project-name", default="multiple_choice_question_study")
    parser.add_argument("--total-epochs", type=int, default=1)
    parser.add_argument("--save-freq", type=int, default=999999, help="Positive value; VERL also saves on last step.")
    parser.add_argument("--test-freq", type=int, default=1)
    parser.add_argument("--train-batch-size", type=int, default=1024)
    parser.add_argument("--ppo-mini-batch-size", type=int, default=1024)
    parser.add_argument("--actor-micro-batch-size", type=int, default=8)
    parser.add_argument("--rollout-n", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--use-judge", action="store_true")
    parser.add_argument("--judge-profile", default="judgelm_7b")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.iterations < 1:
        parser.error("--iterations must be >= 1")
    if args.num_samples < 1:
        parser.error("--num-samples must be >= 1")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "1")
    if args.gpu_ids:
        env["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    gpus = _gpu_count(args.gpu_ids)

    output_root = Path(args.output_root).expanduser().resolve()
    checkpoint_root = Path(args.checkpoint_root).expanduser().resolve()
    val_file = Path(args.val_file).expanduser().resolve()
    current_model: str | Path = args.base_model

    for iteration in range(args.iterations):
        iter_tag = f"{args.run_prefix}-iter{iteration:02d}"
        generation_run_id = f"{iter_tag}-data"
        generation_run_dir = output_root / generation_run_id
        experiment_name = f"{iter_tag}-train"
        checkpoint_dir = checkpoint_root / experiment_name

        build_cmd = [
            sys.executable,
            str(SCRIPT_DIR / "run_build.py"),
            "--run-id",
            generation_run_id,
            "--split",
            args.split,
            "--num-samples",
            str(args.num_samples),
            "--output-root",
            str(output_root),
            "--generator-profile",
            args.generator_profile,
            "--generator-model-path",
            str(current_model),
            "--inference-profile",
            args.inference_profile,
        ]
        if args.gpu_ids:
            build_cmd.extend(["--gpu-ids", args.gpu_ids])
        if args.use_judge:
            build_cmd.extend(["--use-judge", "--judge-profile", args.judge_profile])

        _run(build_cmd, cwd=SCRIPT_DIR, env=env, dry_run=args.dry_run)
        train_file = generation_run_dir / "datasets" / f"gsm8k_{args.train_dataset}_{args.generator_profile}" / f"{args.split}.parquet"
        if not args.dry_run:
            train_file = _generated_train_file(run_dir=generation_run_dir, train_dataset=args.train_dataset)

        train_cmd = _train_command(
            model_path=current_model,
            train_file=train_file,
            val_file=val_file,
            project_name=args.project_name,
            experiment_name=experiment_name,
            checkpoint_dir=checkpoint_dir,
            gpus=gpus,
            total_epochs=args.total_epochs,
            save_freq=args.save_freq,
            train_batch_size=args.train_batch_size,
            ppo_mini_batch_size=args.ppo_mini_batch_size,
            actor_micro_batch_size=args.actor_micro_batch_size,
            rollout_n=args.rollout_n,
            learning_rate=args.learning_rate,
            test_freq=args.test_freq,
            train_dataset=args.train_dataset,
        )
        _run(train_cmd, cwd=REPO_ROOT, env=env, dry_run=args.dry_run)
        if args.dry_run:
            current_model = checkpoint_dir / "best" / "actor" / "huggingface"
        else:
            current_model = _find_next_model_path(checkpoint_dir)
        print(f"[on-policy] iteration={iteration} next_model={current_model}", flush=True)


if __name__ == "__main__":
    main()
