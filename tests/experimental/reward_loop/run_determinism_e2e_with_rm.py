#!/usr/bin/env python3
# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
E2E determinism gate: run PPO twice with identical seeds, verify the per-step
reward curves are bitwise aligned in float32.

Reward curves are parsed from the training subprocess stdout (the ``console``
logger prints ``step:N - key:V - ...`` per step), so no intermediate files are
needed. ``--save_metrics`` also writes a per-run JSONL; ``--plot`` also saves a
comparison PNG (needs matplotlib). Exit code is the verdict: 0 = aligned,
1 = not aligned / a run crashed.

Colocate mode (RM shares the actor/ref/rollout GPU pool). Replica count is
``n_gpus // *_tp``, so the defaults launch 2 rollout + 2 RM replicas — this
exercises both the deterministic load-balancer's hash routing (rollout) and
``NaiveRouter`` crc32 routing (RM). Uses the V0 trainer backend (V1's TQ
collects outputs in completion order, breaking bitwise reproducibility).

RM ``max_num_seqs`` is pinned to 1 in the Hydra command: this script uses a
discriminative RM (no custom reward fn), whose ``/classify`` pooling path is
not covered by ``VLLM_BATCH_INVARIANT`` and would drift under co-batching.
Generative RMs (custom reward fn) are not forced and may run ``>1``. See
``RewardLoopManager`` for the runtime force-to-1 guard.

Usage:
  python tests/experimental/reward_loop/run_determinism_e2e_with_rm.py \\
    --policy_model ~/models/Qwen/Qwen2.5-0.5B-Instruct \\
    --rm_model ~/models/Skywork/Skywork-Reward-V2-Llama-3.2-1B \\
    --train_files ~/data/gsm8k/train.parquet \\
    --val_files ~/data/gsm8k/test.parquet \\
    --n_gpus 2 --n_steps 2
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time

import numpy as np

# Seed used for all determinism configs — same across both runs.
SEED = 42

# Matches a console-logger line of the form
#   step:0 - critic/rewards/mean:0.123 - other/key:1.0
# Captures the step index and the named metric value (as a string token).
_STEP_LINE_RE = re.compile(r"(?<![\w:])step\s*:\s*(\d+)\s*-?\s*(.*)$")


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
    temperature=0.7,
    max_num_seqs=4,
    save_metrics=False,
):
    """Run one PPO training session via main_ppo using Hydra overrides.

    Captures stdout so the caller can parse the per-step reward curve. No JSONL
    is written unless ``save_metrics`` is set.

    Colocate mode: RM shares the actor/ref/rollout GPU pool, so ``n_gpus // *_tp``
    rollout and RM replicas are created.
    """
    env = os.environ.copy()
    env["VLLM_DISABLE_COMPILE_CACHE"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "true"
    env["NCCL_DEBUG"] = "WARN"
    env["VLLM_LOGGING_LEVEL"] = "WARN"
    # PYTHONHASHSEED must be set before Python starts for deterministic hash().
    env["PYTHONHASHSEED"] = str(SEED)

    # Default: console only. --save_metrics adds the file logger.
    loggers = ["console"]
    jsonl_path = os.path.join(output_dir, f"{run_name}.jsonl")
    if save_metrics:
        loggers = ["console", "file"]
        env["VERL_FILE_LOGGER_PATH"] = jsonl_path

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
        f"actor_rollout_ref.rollout.temperature={temperature}",
        "actor_rollout_ref.rollout.calculate_log_probs=true",
        "actor_rollout_ref.rollout.n=1",
        "actor_rollout_ref.rollout.prompt_length=64",
        "actor_rollout_ref.rollout.response_length=64",
        "actor_rollout_ref.rollout.max_model_len=128",
        f"actor_rollout_ref.rollout.max_num_seqs={max_num_seqs}",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={rollout_tp}",
        "actor_rollout_ref.rollout.agent.num_workers=1",
        "actor_rollout_ref.rollout.disable_log_stats=true",
        # Actor
        "actor_rollout_ref.actor.fsdp_config.full_determinism=true",
        f"actor_rollout_ref.actor.fsdp_config.seed={SEED}",
        "actor_rollout_ref.actor.fsdp_config.param_offload=false",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=false",
        "actor_rollout_ref.actor.fsdp_config.fsdp_size=1",
        "actor_rollout_ref.actor.use_dynamic_bsz=true",
        "actor_rollout_ref.actor.ppo_mini_batch_size=4",
        "actor_rollout_ref.actor.shuffle=false",
        # Ref
        "actor_rollout_ref.ref.fsdp_config.full_determinism=true",
        f"actor_rollout_ref.ref.fsdp_config.seed={SEED}",
        # Model
        f"actor_rollout_ref.model.path={policy_model}",
        "actor_rollout_ref.model.trust_remote_code=true",
        # RM — colocate (shares GPU pool with actor/ref/rollout; _compute_reward_colocate path).
        "reward.reward_model.enable=true",
        f"reward.reward_model.model_path={rm_model}",
        "reward.reward_model.rollout.name=vllm",
        "reward.reward_model.rollout.full_determinism=true",
        f"reward.reward_model.rollout.seed={SEED}",
        "reward.reward_model.rollout.enforce_eager=true",
        "reward.reward_model.rollout.gpu_memory_utilization=0.3",
        f"reward.reward_model.rollout.tensor_model_parallel_size={rm_tp}",
        "reward.reward_model.rollout.max_model_len=512",
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
        # V0 backend (V1's TQ collects outputs in completion order → nondeterministic concat).
        "trainer.use_v1=false",
        f"trainer.total_training_steps={n_steps}",
        "trainer.total_epochs=1",
        "data.train_batch_size=8",
        f"trainer.experiment_name={run_name}",
        f"trainer.logger={list(loggers)}".replace(" ", ""),  # e.g. ['console','file']
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
    print(f"Starting {run_name} (metrics: {', '.join(loggers)})")
    print(f"{'=' * 60}\n")

    subprocess.run(["ray", "stop"], env=env, capture_output=True)
    time.sleep(3)

    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    stdout_lines = []
    assert proc.stdout is not None
    for line in proc.stdout:
        stdout_lines.append(line)
        sys.stdout.write(line)
        sys.stdout.flush()
    returncode = proc.wait()
    if returncode != 0:
        print(f"ERROR: {run_name} failed (return code {returncode})")
        if save_metrics and os.path.exists(jsonl_path):
            print(f"Partial metrics in {jsonl_path}:")
            with open(jsonl_path) as f:
                for line in f:
                    print(line.rstrip())
        sys.exit(1)

    print(f"\n{run_name} completed.")
    return "".join(stdout_lines)


_NP_WRAP = re.compile(r"np\.(?:float|int|uint|bool)\d*\((.+)\)")
_NUM_RE = re.compile(r"(?<![A-Za-z0-9._])[-+]?(?:\d+\.\d+|\d+\.|\.\d+|\d+)(?:[eE][-+]?\d+)?")


def _parse_float(token: str) -> float:
    """Parse a metric value to float (handles ``np.float64(x)``, ``np.int32(n)``, plain floats)."""
    token = token.strip()
    m = _NP_WRAP.match(token)
    if m:
        token = m.group(1).strip()
    try:
        return float(token)
    except ValueError:
        m = _NUM_RE.search(token)
        return float(m.group(0)) if m else float("nan")


def parse_reward_curve_from_stdout(stdout: str):
    """Parse per-step reward metrics from training stdout.

    Console logger lines look like ``step:1 - critic/rewards/mean:0.5 - ...``.
    Returns ``(steps, rewards)`` using the reward key picked per step, or
    ``(None, None)`` if no step line with a reward key was found.
    """
    metrics_by_step = {}
    for line in stdout.splitlines():
        m = _STEP_LINE_RE.search(line)
        if not m:
            continue
        step = int(m.group(1))
        rest = m.group(2)
        data = {}
        # Values may contain ':'; split on " - " first, then on the FIRST ':' only.
        for chunk in rest.split(" - "):
            chunk = chunk.strip()
            if not chunk or ":" not in chunk:
                continue
            key, _, value = chunk.partition(":")
            data[key.strip()] = value.strip()
        if data:
            metrics_by_step[step] = data

    if not metrics_by_step:
        return None, None

    steps = sorted(metrics_by_step)
    rewards = []
    for s in steps:
        key = _pick_reward_key(metrics_by_step[s])
        if key is None:
            print(f"ERROR: No reward key found in step {s}")
            print(f"  Available keys: {list(metrics_by_step[s].keys())}")
            return None, None
        rewards.append(_parse_float(metrics_by_step[s][key]))

    train_key = _pick_reward_key(metrics_by_step.get(1, {}))
    val_key = _pick_reward_key(metrics_by_step.get(0, {}))
    print(f"Using training reward key: {train_key}, validation reward key: {val_key}")

    return steps, rewards


def read_metrics_from_jsonl(output_dir, run_name):
    """Read per-step metrics from FileLogger JSONL output (opt-in path)."""
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
    """Pick the reward key from a step's data dict (training or validation)."""
    for key in ["critic/rewards/mean", "reward/mean", "train/reward/mean"]:
        if key in data:
            return key
    for key in ["val-core/unknown/reward/mean@1", "val-core/reward/mean"]:
        if key in data:
            return key
    for key in data:  # fallback: any key with "reward" and "mean"
        if "reward" in key.lower() and "mean" in key.lower():
            return key
    return None


def extract_reward_curve(metrics_by_step):
    """Extract mean reward per step from a metrics dict (JSONL path)."""
    steps = sorted(metrics_by_step.keys())
    rewards = []
    for s in steps:
        key = _pick_reward_key(metrics_by_step[s])
        if key is None:
            print(f"ERROR: No reward key found in step {s}")
            print(f"  Available keys: {list(metrics_by_step[s].keys())}")
            sys.exit(1)
        rewards.append(float(metrics_by_step[s][key]))

    train_key = _pick_reward_key(metrics_by_step.get(1, {}))
    val_key = _pick_reward_key(metrics_by_step.get(0, {}))
    print(f"Using training reward key: {train_key}, validation reward key: {val_key}")

    return steps, rewards


def verify_bitwise_alignment(rewards1, rewards2):
    """Verify two reward curves are bitwise aligned (float32).

    Returns ``(aligned, reason)`` so the caller surfaces a concrete failure cause.
    """
    r1 = np.array(rewards1, dtype=np.float32)
    r2 = np.array(rewards2, dtype=np.float32)

    if len(r1) != len(r2):
        return (
            False,
            f"length mismatch: run1 has {len(r1)} steps, run2 has {len(r2)} steps",
        )

    aligned = bool(np.all(r1 == r2))
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
                print(f"  Step {i} DIFF: {float(r1[i])} vs {float(r2[i])}, diff={abs(float(r1[i]) - float(r2[i]))}")
    print(f"{'=' * 60}")

    if aligned:
        return True, "reward curves are bitwise aligned"
    return False, f"reward curves differ (max_diff={max_diff:.3e})"


def print_reward_table(steps, rewards1, rewards2, aligned):
    """Print a compact side-by-side reward comparison to stdout."""
    print(f"\n{'Step':>6} | {'Run 1':>16} | {'Run 2':>16} | {'Diff':>12} | {'Match':>5}")
    print("-" * 64)
    for i, s in enumerate(steps):
        r1 = float(rewards1[i])
        r2 = float(rewards2[i])
        diff = abs(r1 - r2)
        match = "✓" if r1 == r2 else "✗"
        print(f"{s:>6} | {r1:>16.8f} | {r2:>16.8f} | {diff:>12.3e} | {match:>5}")
    print("-" * 64)
    status = "✓ BITWISE ALIGNED" if aligned else "✗ NOT ALIGNED"
    print(f"Verdict: {status}\n")


def plot_reward_curves(steps1, rewards1, rewards2, output_path):
    """Plot two reward curves overlaid for visual comparison."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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
    parser = argparse.ArgumentParser(
        description="E2E determinism gate: run PPO twice, verify reward curves bitwise aligned.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--policy_model", default="~/models/Qwen/Qwen2.5-0.5B-Instruct", help="Policy (actor) model path."
    )
    parser.add_argument(
        "--rm_model",
        default="~/models/Skywork/Skywork-Reward-V2-Llama-3.2-1B",
        help="Reward model path (discriminative → max_num_seqs forced to 1).",
    )
    parser.add_argument("--n_gpus", type=int, default=2, help="Total GPUs (single node); replicas = n_gpus // *_tp.")
    parser.add_argument("--rollout_tp", type=int, default=1, help="Rollout tensor-parallel size per replica.")
    parser.add_argument("--rm_tp", type=int, default=1, help="RM tensor-parallel size per replica.")
    parser.add_argument("--n_steps", type=int, default=2, help="PPO steps per run (raise to 10 for a longer curve).")
    parser.add_argument("--train_files", default=None, help="Training parquet (omitted → synthetic dataset).")
    parser.add_argument("--val_files", default=None, help="Validation parquet (omitted → reuse train set).")
    parser.add_argument(
        "--output_dir", default="/tmp/determinism_e2e", help="Where tables / plots / optional JSONL go."
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Rollout temperature (0 = greedy, deterministic without a seed)."
    )
    parser.add_argument(
        "--max_num_seqs",
        type=int,
        default=4,
        help="Actor ROLLOUT max_num_seqs (the RM's is forced to 1). Lower to 1 to serialize if the policy "
        "model is not batch-invariant.",
    )
    parser.add_argument("--project_name", default="determinism_e2e", help="Tracking project name.")
    parser.add_argument("--save_metrics", action="store_true", help="Also write {output_dir}/{run_name}.jsonl per run.")
    parser.add_argument("--plot", action="store_true", help="Also save a comparison PNG (needs matplotlib).")
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
        temperature=args.temperature,
        max_num_seqs=args.max_num_seqs,
        train_files=args.train_files,
        val_files=args.val_files,
        save_metrics=args.save_metrics,
    )

    # ── Run 1 ──
    stdout1 = run_training(run_name="run1", **common_kwargs)

    # ── Run 2 ──
    stdout2 = run_training(run_name="run2", **common_kwargs)

    # ── Extract reward curves ──
    if args.save_metrics:
        metrics1 = read_metrics_from_jsonl(output_dir, "run1")
        metrics2 = read_metrics_from_jsonl(output_dir, "run2")
        steps1, rewards1 = extract_reward_curve(metrics1)
        steps2, rewards2 = extract_reward_curve(metrics2)
    else:
        steps1, rewards1 = parse_reward_curve_from_stdout(stdout1)
        steps2, rewards2 = parse_reward_curve_from_stdout(stdout2)
        if steps1 is None or steps2 is None:
            print("\n❌ FAILURE: could not parse reward metrics from stdout.")
            sys.exit(1)

    # ── Verify ──
    aligned, reason = verify_bitwise_alignment(rewards1, rewards2)

    # ── Show results ──
    if steps1 == steps2:
        print_reward_table(steps1, rewards1, rewards2, aligned)
    else:  # length mismatch already reported; dump both raw curves
        print("\nRun 1 reward curve:")
        for s, r in zip(steps1, rewards1, strict=False):
            print(f"  step {s}: {float(r):.8f}")
        print("Run 2 reward curve:")
        for s, r in zip(steps2, rewards2, strict=False):
            print(f"  step {s}: {float(r):.8f}")

    # ── Plot (opt-in) ──
    if args.plot:
        try:
            plot_path = os.path.join(output_dir, "determinism_reward_curves.png")
            plot_steps = steps1 if steps1 == steps2 else list(range(min(len(rewards1), len(rewards2))))
            plot_reward_curves(plot_steps, rewards1[: len(plot_steps)], rewards2[: len(plot_steps)], plot_path)
        except ImportError:
            print("matplotlib not installed; skipping plot.")

    # ── Exit code is the verdict (CI gate) ──
    if aligned:
        print("\n🎉 SUCCESS: reward curves are bitwise aligned!")
        sys.exit(0)
    else:
        print(f"\n❌ FAILURE: {reason}")
        sys.exit(1)


if __name__ == "__main__":
    main()
