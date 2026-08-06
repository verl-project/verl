#!/usr/bin/env python3
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
E2E determinism verification: run PPO training twice, verify reward curves bitwise aligned.

Uses colocate mode: RM shares the same GPU pool with actor/ref/rollout,
so RM scoring goes through the ``_compute_reward_colocate`` path with
sleep/wake GPU memory management.

Usage:
  python tests/experimental/reward_loop/run_determinism_e2e_with_rm.py \
    --policy_model ~/models/Qwen/Qwen2.5-0.5B-Instruct \
    --rm_model ~/models/Skywork/Skywork-Reward-V2-Llama-3.2-1B \
    --train_files ~/data/gsm8k/train.parquet \
    --val_files ~/data/gsm8k/test.parquet \
    --n_gpus 2 --n_steps 5

Defaults to 2 GPUs: all shared by actor/ref/rollout/RM in colocate mode.

After both runs complete, verifies bitwise alignment of reward curves and
saves a comparison plot to output_dir/determinism_reward_curves.png.
"""

import argparse
import json
import os
import subprocess
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

# Seed used for all determinism configs — same across both runs.
SEED = 42


def run_training(
    policy_model,
    rm_model,
    n_gpus,
    n_steps,
    run_name,
    output_dir,
    rollout_tp=1,
    rm_tp=1,
    train_files=None,
    val_files=None,
):
    """Run one PPO training session via main_ppo using Hydra overrides.

    Uses colocate mode (RM shares the same GPU pool with actor/ref/rollout)
    so that RM scoring goes through the well-tested ``_compute_reward_colocate``
    path with sleep/wake GPU memory management.

    Args:
        n_gpus: Total GPUs visible to the trainer.  All are shared by
            actor/ref/rollout/RM in colocate mode.
        rollout_tp: Rollout TP per replica.
        rm_tp: RM TP per replica.  In colocate mode, RM replicas share the
            same GPUs as rollout; ``n_gpus // rollout_tp`` rollout replicas and
            ``n_gpus // rm_tp`` RM replicas are created.
    """
    env = os.environ.copy()
    jsonl_path = os.path.join(output_dir, f"{run_name}.jsonl")
    env = os.environ.copy()
    env["VLLM_DISABLE_COMPILE_CACHE"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "true"
    env["NCCL_DEBUG"] = "WARN"
    env["VLLM_LOGGING_LEVEL"] = "WARN"
    env["VERL_FILE_LOGGER_PATH"] = jsonl_path
    # PYTHONHASHSEED must be set before Python starts; this ensures deterministic
    # hash() across all processes spawned by this training run (driver, Ray actors,
    # NaiveRouter subprocess, etc.).
    env["PYTHONHASHSEED"] = str(SEED)
    env["VERL_DETERMINISM_DEBUG"] = "1"

    cmd = [
        sys.executable,
        "-m",
        "verl.trainer.main_ppo",
        # Rollout
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.mode=async",
        "actor_rollout_ref.rollout.full_determinism=true",
        f"actor_rollout_ref.rollout.seed={SEED}",
        "actor_rollout_ref.rollout.scheduling_policy=priority",
        "actor_rollout_ref.rollout.enforce_eager=true",
        "actor_rollout_ref.rollout.temperature=0.7",
        "actor_rollout_ref.rollout.calculate_log_probs=true",
        "actor_rollout_ref.rollout.n=1",
        "actor_rollout_ref.rollout.prompt_length=64",
        "actor_rollout_ref.rollout.response_length=64",
        "actor_rollout_ref.rollout.max_model_len=128",
        "actor_rollout_ref.rollout.max_num_seqs=4",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={rollout_tp}",
        "actor_rollout_ref.rollout.agent.num_workers=1",
        "actor_rollout_ref.rollout.disable_log_stats=true",
        # Actor
        "actor_rollout_ref.actor.fsdp_config.full_determinism=true",
        f"actor_rollout_ref.actor.fsdp_config.seed={SEED}",
        "actor_rollout_ref.actor.fsdp_config.param_offload=true",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=true",
        "actor_rollout_ref.actor.fsdp_config.fsdp_size=1",
        "actor_rollout_ref.actor.use_dynamic_bsz=true",
        "actor_rollout_ref.actor.ppo_mini_batch_size=4",
        # Ref
        "actor_rollout_ref.ref.fsdp_config.full_determinism=true",
        f"actor_rollout_ref.ref.fsdp_config.seed={SEED}",
        # Model
        f"actor_rollout_ref.model.path={policy_model}",
        "actor_rollout_ref.model.trust_remote_code=true",
        # RM — colocate mode: shares GPU pool with actor/ref/rollout.
        # RM scoring goes through _compute_reward_colocate path with
        # sleep/wake GPU memory management (not agent-loop HTTP path).
        "reward.reward_model.enable=true",
        f"reward.reward_model.model_path={rm_model}",
        "reward.reward_model.rollout.name=vllm",
        "reward.reward_model.rollout.full_determinism=true",
        f"reward.reward_model.rollout.seed={SEED}",
        "reward.reward_model.rollout.enforce_eager=true",
        "reward.reward_model.rollout.gpu_memory_utilization=0.4",
        f"reward.reward_model.rollout.tensor_model_parallel_size={rm_tp}",
        "reward.reward_model.rollout.max_model_len=512",
        # Serialize RM inference for determinism: max_num_seqs=1 ensures
        # each RM forward pass processes one request at a time.
        "reward.reward_model.rollout.max_num_seqs=1",
        "reward.reward_model.rollout.skip_tokenizer_init=false",
        "reward.reward_model.rollout.disable_log_stats=true",
        "reward.reward_manager.name=dapo",
        # Algorithm
        "algorithm.adv_estimator=grpo",
        "critic.enable=false",
        # Trainer
        f"trainer.n_gpus_per_node={n_gpus}",
        "trainer.nnodes=1",
        f"trainer.total_training_steps={n_steps}",
        "trainer.total_epochs=1",
        "data.train_batch_size=8",
        f"trainer.experiment_name={run_name}",
        "trainer.logger=[console,file]",
        # Data
        "data.shuffle=true",
        f"data.seed={SEED}",
        "data.max_prompt_length=64",
        "data.max_response_length=64",
        f"+exp_name={run_name}",
        # Limit data to avoid long iterations
        "data.train_max_samples=64",
        "data.val_max_samples=64",
    ]
    if train_files:
        cmd.append(f"data.train_files={train_files}")
    if val_files:
        cmd.append(f"data.val_files={val_files}")

    print(f"\n{'=' * 60}")
    print(f"Starting {run_name}")
    print(f"{'=' * 60}\n")

    subprocess.run(["ray", "stop"], env=env, capture_output=True)
    time.sleep(3)

    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        print(f"ERROR: {run_name} failed (return code {result.returncode})")
        # Print the JSONL file if it exists — it may contain partial metrics useful for debugging.
        jsonl_path = os.path.join(output_dir, f"{run_name}.jsonl")
        if os.path.exists(jsonl_path):
            print(f"Partial metrics in {jsonl_path}:")
            with open(jsonl_path) as f:
                for line in f:
                    print(line.rstrip())
        sys.exit(1)

    print(f"\n{run_name} completed.")


def read_metrics_from_jsonl(output_dir, run_name):
    """Read per-step metrics from FileLogger JSONL output."""
    filepath = os.path.join(output_dir, f"{run_name}.jsonl")
    if not os.path.exists(filepath):
        print(f"ERROR: Metrics file not found at {filepath}")
        sys.exit(1)

    steps = {}
    with open(filepath) as f:
        for line in f:
            entry = json.loads(line)
            step = entry["step"]
            steps[step] = entry["data"]
    return steps


def _pick_reward_key(data: dict) -> str | None:
    """Pick the best reward key from a single step's data dict.

    Training steps use ``critic/rewards/mean``; validation (step 0) uses
    ``val-core/.../reward/mean@1``.  Returns None if nothing matches.
    """
    # Training-step keys (preferred order)
    for key in ["critic/rewards/mean", "reward/mean", "train/reward/mean"]:
        if key in data:
            return key
    # Validation-step keys
    for key in ["val-core/unknown/reward/mean@1", "val-core/reward/mean"]:
        if key in data:
            return key
    # Fallback: any key containing both "reward" and "mean"
    for key in data:
        if "reward" in key.lower() and "mean" in key.lower():
            return key
    return None


def extract_reward_curve(metrics_by_step):
    """Extract mean reward per step from metrics dict.

    Step 0 (validation) uses a different key namespace than training steps,
    so we pick the best key for each step independently.
    """
    steps = sorted(metrics_by_step.keys())
    rewards = []
    for s in steps:
        key = _pick_reward_key(metrics_by_step[s])
        if key is None:
            print(f"ERROR: No reward key found in step {s}")
            print(f"  Available keys: {list(metrics_by_step[s].keys())}")
            sys.exit(1)
        rewards.append(metrics_by_step[s][key])

    train_key = _pick_reward_key(metrics_by_step.get(1, {}))
    val_key = _pick_reward_key(metrics_by_step.get(0, {}))
    print(f"Using training reward key: {train_key}, validation reward key: {val_key}")

    return steps, rewards


def verify_bitwise_alignment(rewards1, rewards2):
    """Verify two reward curves are bitwise aligned (float32)."""
    r1 = np.array(rewards1, dtype=np.float32)
    r2 = np.array(rewards2, dtype=np.float32)

    assert len(r1) == len(r2), f"Length mismatch: {len(r1)} vs {len(r2)}"

    aligned = np.all(r1 == r2)
    max_diff = float(np.max(np.abs(r1 - r2))) if not aligned else 0.0

    print(f"\n{'=' * 60}")
    print("Bitwise alignment check (float32)")
    print(f"  Steps: {len(r1)}")
    print(f"  Aligned: {aligned}")
    print(f"  Max diff: {max_diff}")
    print(f"  Run 1: {r1.tolist()}")
    print(f"  Run 2: {r2.tolist()}")
    if not aligned:
        for i in range(len(r1)):
            if r1[i] != r2[i]:
                print(f"  Step {i + 1} DIFF: {float(r1[i])} vs {float(r2[i])}, diff={abs(float(r1[i]) - float(r2[i]))}")
    print(f"{'=' * 60}")

    return aligned


def plot_reward_curves(steps1, rewards1, rewards2, output_path):
    """Plot two reward curves overlaid for visual comparison."""
    r1 = [float(r) for r in rewards1]
    r2 = [float(r) for r in rewards2]
    steps = [s for s in steps1]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(steps, r1, "o-", label="Run 1", linewidth=2.5, markersize=10, color="#2563eb", zorder=3)
    ax.plot(steps, r2, "s-", label="Run 2", linewidth=2.5, markersize=10, color="#dc2626", zorder=2)

    ax.set_xlabel("Training Step", fontsize=14)
    ax.set_ylabel("Mean Reward", fontsize=14)
    ax.set_title("E2E Determinism: Reward Curve Comparison", fontsize=16, fontweight="bold")
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    aligned = np.all(np.array(r1, dtype=np.float32) == np.array(r2, dtype=np.float32))
    color = "green" if aligned else "red"
    status = "✓ BITWISE ALIGNED" if aligned else "✗ NOT ALIGNED"
    ax.annotate(
        status,
        xy=(0.5, 0.95),
        xycoords="axes fraction",
        fontsize=16,
        ha="center",
        fontweight="bold",
        color=color,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=color, alpha=0.9),
    )

    if aligned:
        ax.annotate(
            "max_diff = 0 (float32)", xy=(0.5, 0.88), xycoords="axes fraction", fontsize=12, ha="center", color="green"
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="E2E determinism verification: run twice, compare reward curves")
    parser.add_argument("--policy_model", default="~/models/Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--rm_model", default="~/models/Skywork/Skywork-Reward-V2-Llama-3.2-1B")
    # All GPUs shared by actor/ref/rollout/RM in colocate mode.
    parser.add_argument("--n_gpus", type=int, default=2)
    parser.add_argument("--rollout_tp", type=int, default=1)
    parser.add_argument("--rm_tp", type=int, default=1)
    parser.add_argument("--n_steps", type=int, default=2)
    parser.add_argument("--train_files", default=None, help="Path to training data parquet")
    parser.add_argument("--val_files", default=None, help="Path to validation data parquet")
    parser.add_argument("--output_dir", default="/tmp/determinism_e2e")
    parser.add_argument("--project_name", default="determinism_e2e")
    args = parser.parse_args()

    policy_model = os.path.expanduser(args.policy_model)
    rm_model = os.path.expanduser(args.rm_model)
    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    common_kwargs = dict(
        policy_model=policy_model,
        rm_model=rm_model,
        n_gpus=args.n_gpus,
        n_steps=args.n_steps,
        output_dir=output_dir,
        rollout_tp=args.rollout_tp,
        rm_tp=args.rm_tp,
        train_files=args.train_files,
        val_files=args.val_files,
    )

    # ── Run 1 ──
    run_training(run_name="run1", **common_kwargs)

    # ── Run 2 ──
    run_training(run_name="run2", **common_kwargs)

    # ── Read metrics ──
    metrics1 = read_metrics_from_jsonl(output_dir, "run1")
    metrics2 = read_metrics_from_jsonl(output_dir, "run2")

    steps1, rewards1 = extract_reward_curve(metrics1)
    steps2, rewards2 = extract_reward_curve(metrics2)

    # ── Verify ──
    aligned = verify_bitwise_alignment(rewards1, rewards2)

    # ── Plot ──
    plot_path = os.path.join(output_dir, "determinism_reward_curves.png")
    plot_reward_curves(steps1, rewards1, rewards2, plot_path)

    if aligned:
        print("\n🎉 SUCCESS: E2E determinism verified — reward curves are bitwise aligned!")
    else:
        print("\n❌ FAILURE: Reward curves are NOT bitwise aligned.")
        print(
            "Check: actor/critic full_determinism, rollout seed + priority"
            " + batch_invariant, RM full_determinism, data shuffle seed"
        )


if __name__ == "__main__":
    main()
