# Copyright (c) 2026 Google LLC. All rights reserved.
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
"""Verify the log of a TPU CI test.

Tests (see tests/special_tpu/run_tpu_e2e_ci.sh):
    smoke       - tests/special_tpu/tpu_smoke_test.py (1 chip)
    unit        - the verl unit tests of TPU_UNIT_TESTS, one pytest process per file (1 chip)
    sft         - examples/tpu/sft/run_qwen3_0_6b_torchtitan.sh, SMOKE_TEST=1
    sft_resume  - the same script resumed from the checkpoint of the sft run (trainer.resume_mode=auto)
    grpo        - examples/tpu/grpo/run_qwen3_torchtitan.sh MODEL_SIZE=0.6B (5-step smoke test)
    grpo_4b     - examples/tpu/grpo/run_qwen3_torchtitan.sh MODEL_SIZE=4B (nightly, with validation)

The GRPO tests run with the tpu-sync parity checks on (VERIFY_PARITY=all); their verifiers also require the checks'
results in the log (see verify_tpu_sync_parity).

Usage:
    python3 tests/special_tpu/verify_tpu_e2e_log.py <smoke|unit|sft|sft_resume|grpo|grpo_4b> <log_file> [--steps N]
"""

import argparse
import math
import re
import sys
from statistics import mean

NUMBER = r"(?:np\.float\d*\()?([-+]?(?:[0-9.]+(?:[eE][-+]?[0-9]+)?|nan|inf))"


def _fail(message: str) -> None:
    sys.exit(f"[TPU CI] ERROR: {message}")


def _step_metrics(text: str) -> dict[int, dict[str, float]]:
    """Return {step: {metric: value}} from the trainers' `step:N - key:value - ...` console lines.

    Step 0 only exists with `trainer.val_before_train=True` and only carries validation metrics.
    """
    steps: dict[int, dict[str, float]] = {}
    for line in text.splitlines():
        match = re.search(r"\bstep:(\d+) - (.*)", line)
        if not match:
            continue
        metrics = steps.setdefault(int(match.group(1)), {})
        for key, value in re.findall(r"([\w@/().\-]+):" + NUMBER, match.group(2)):
            metrics[key] = float(value)
    return steps


def _training_steps(text: str, want_steps: int) -> dict[int, dict[str, float]]:
    """Return the metrics of the training steps (>= 1) and check that `want_steps` of them were logged."""
    steps = {s: m for s, m in _step_metrics(text).items() if s >= 1}
    if not steps:
        _fail("no training step metrics (`step:N - ...`) in the job log.")
    if want_steps and max(steps) < want_steps:
        _fail(f"only {max(steps)} of {want_steps} training steps were logged.")
    return steps


def _series(steps: dict[int, dict[str, float]], key: str) -> list[float]:
    return [steps[s][key] for s in sorted(steps) if key in steps[s]]


def _check_finite(steps: dict[int, dict[str, float]], keys: list[str]) -> None:
    for key in keys:
        bad = {s: m[key] for s, m in sorted(steps.items()) if key in m and not math.isfinite(m[key])}
        if bad:
            _fail(f"non-finite {key}: {bad}")


def verify_smoke(text: str, args: argparse.Namespace) -> None:
    for marker, what in [
        ("[TPU smoke] PASSED", "did not report PASSED"),
        ('"device": "tpu:0"', "did not run on a tpu device"),
        ("[TPU smoke] TPU vs CPU max |diff|", "did not run the TPU-vs-CPU check"),
        ("[TPU smoke] verl TPU platform:", "did not check the verl TPU platform (verl-hardware-plugin)"),
    ]:
        if marker not in text:
            _fail(f"TPU platform test {what}.")
    print("[TPU CI] TPU platform test verified.")


def verify_unit(text: str, args: argparse.Namespace) -> None:
    """The verl unit tests that test_tpu_platform runs with run_tpu_unit_tests.sh (one pytest process per file)."""
    ran = re.findall(r"^\[TPU CI\] pytest (\S+): exit code (\d+)$", text, re.MULTILINE)
    if args.files and len(ran) != args.files:
        _fail(f"{len(ran)} of {args.files} unit test files ran.")
    failed = [path for path, code in ran if code != "0"]
    if failed:
        _fail(f"unit tests failed on TPU: {', '.join(failed)} (see their pytest output above).")
    counts = []
    for path, _ in ran:
        output = text.split(f"[TPU CI] pytest {path}\n", 1)[-1].split(f"[TPU CI] pytest {path}: exit code", 1)[0]
        passed = len(re.findall(r"^PASSED ", output, re.MULTILINE))
        if not passed:
            _fail(f"no test of {path} passed (all skipped or not collected).")
        counts.append(f"{path.rsplit('/', 1)[-1]} {passed}")
    print(f"[TPU CI] verl unit tests on TPU: {len(ran)} files passed ({', '.join(counts)} tests)")


def verify_sft(text: str, args: argparse.Namespace) -> None:
    steps = _training_steps(text, args.steps)
    _check_finite(steps, ["train/loss", "train/grad_norm", "val/loss"])
    losses, val_losses = _series(steps, "train/loss"), _series(steps, "val/loss")
    if not losses or not val_losses:
        _fail("missing train/loss or val/loss in the SFT output.")
    if not losses[-1] < losses[0]:
        _fail(f"SFT train/loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}")
    if len(val_losses) >= 2 and not val_losses[-1] < val_losses[0]:
        _fail(f"SFT val/loss did not decrease: {val_losses[0]:.4f} -> {val_losses[-1]:.4f}")
    saved = ""
    if args.saved_step:
        # verl.utils.checkpoint.checkpoint_handler.CheckpointHandler.save_checkpoint
        for marker in (f"Saving checkpoint to: \\S*global_step_{args.saved_step}\\b", r"Updated checkpoint tracker: "):
            if not re.search(marker, text):
                _fail(f"no checkpoint of step {args.saved_step} was saved (missing {marker!r}).")
        saved = f", checkpoint saved at step {args.saved_step}"
    print(
        f"[TPU CI] SFT verified: {max(steps)} steps, train/loss {losses[0]:.4f} -> {losses[-1]:.4f}, "
        f"val/loss {' -> '.join(f'{v:.4f}' for v in val_losses)}{saved}"
    )


def verify_sft_resume(text: str, args: argparse.Namespace) -> None:
    """An SFT run that resumed from the checkpoint of a previous run (trainer.resume_mode=auto)."""
    found = re.search(r"Found latest checkpoint: \S*global_step_(\d+)", text)
    if not found:
        _fail("the run did not resume from a checkpoint (no 'Found latest checkpoint').")
    resumed = int(found.group(1))
    if args.resumed_step and resumed != args.resumed_step:
        _fail(f"resumed from step {resumed} instead of {args.resumed_step}.")
    steps = _training_steps(text, args.steps)
    _check_finite(steps, ["train/loss", "train/grad_norm"])
    if min(steps) != resumed + 1:
        _fail(f"the first logged training step is {min(steps)}, not {resumed + 1}: the step counter was not restored.")
    losses = _series(steps, "train/loss")
    if not losses:
        _fail("missing train/loss in the SFT output.")
    if args.max_first_loss and not losses[0] < args.max_first_loss:
        # A model that did not get the checkpoint's weights starts over at the loss of the base model.
        _fail(f"the first resumed train/loss {losses[0]:.4f} is not below {args.max_first_loss}: weights not restored?")
    print(
        f"[TPU CI] SFT resume verified: resumed from step {resumed}, steps {min(steps)}..{max(steps)}, "
        f"train/loss {' -> '.join(f'{v:.4f}' for v in losses)}"
    )


def _rollout_vs_trainer(steps: dict[int, dict[str, float]], name: str) -> list[float]:
    """Series of a rollout-correction metric; logged as `actor/rollout_corr/*` in bypass mode."""
    return _series(steps, f"rollout_corr/{name}") or _series(steps, f"actor/rollout_corr/{name}")


def verify_tpu_sync_parity(text: str) -> str:
    """Check that the tpu-sync parity checks (VERIFY_PARITY=all) ran on every weight sync and passed.

    verl-hardware-plugin logs the results of its parity checks; with fail-on-mismatch it also stops the run.
    Requiring them here makes sure the checks actually ran: a config or logging change that silently turned them
    off would otherwise pass CI.
    """
    if "TPU-SYNC PARITY MISMATCH" in text or "TPUSyncParityError" in text:
        _fail("a tpu-sync parity check failed (see the [TPU-SYNC PARITY ...] lines in the log).")
    synced = {int(s) for s in re.findall(r"\[TPU-SYNC TELEMETRY \| Orchestrator\] Step (\d+) Completed", text)}
    if 0 not in synced:
        _fail("no tpu-sync step-0 weight sync ([TPU-SYNC TELEMETRY | Orchestrator] Step 0) in the log.")
    if not re.search(r"\[TPU-SYNC PARITY EXACT \| Step 0\] .* 0 mismatched, 0 unresolved", text):
        _fail("the tpu-sync step-0 exact parity check did not run or did not report 0 mismatched, 0 unresolved.")
    verified = {int(s) for s in re.findall(r"\[TPU-SYNC PARITY VERIFIED \| Step (\d+)", text)}
    unverified = sorted(synced - {0} - verified)
    if unverified:
        _fail(f"the tpu-sync syncs at steps {unverified} have no [TPU-SYNC PARITY VERIFIED] norm check.")
    if not re.search(r"timing_s/tpu-sync/total_sync[:\s]+[0-9.eE+-]+", text):
        _fail("no timing_s/tpu-sync/total_sync metric in the log.")
    return f"tpu-sync exact parity at step 0, norm parity at steps {sorted(verified)}"


def _verify_grpo_common(text: str, args: argparse.Namespace) -> tuple[dict[int, dict[str, float]], str]:
    steps = _training_steps(text, args.steps)
    parity_summary = verify_tpu_sync_parity(text)
    _check_finite(steps, ["actor/grad_norm", "actor/pg_loss", "critic/rewards/mean"])
    grad_norms = _series(steps, "actor/grad_norm")
    if not grad_norms or max(grad_norms) <= 0.0:
        _fail(f"actor/grad_norm was 0 or missing on every step (no gradient flowed): {grad_norms}")
    # Rollout (vLLM) vs trainer log-probs of the same sampled tokens. A healthy run has KL ~1e-3 and a
    # perplexity ratio ~1.001; a weight sync that corrupts or skips weights, or engines that compute
    # different log-probs, push the KL up by orders of magnitude (ppl ratio ~20 for a corrupted sync).
    kls, ppl_ratios = _rollout_vs_trainer(steps, "kl"), _rollout_vs_trainer(steps, "ppl_ratio")
    if len(kls) < len(steps) or len(ppl_ratios) < len(steps):
        _fail(f"rollout_corr/kl and rollout_corr/ppl_ratio must be logged on all {len(steps)} steps.")
    if max(kls) > 0.02 or max(abs(r - 1.0) for r in ppl_ratios) > 0.05:
        _fail(f"rollout and trainer log-probs disagree: kl {kls}, ppl_ratio {ppl_ratios}")
    corrs = _series(steps, "training/rollout_actor_probs_pearson_corr")
    if corrs and min(corrs) < 0.9:
        _fail(f"rollout vs trainer log-prob Pearson correlation below 0.9: {[round(c, 4) for c in corrs]}")
    summary = f"max rollout-vs-trainer KL {max(kls):.5f}" + (f", min Pearson {min(corrs):.4f}" if corrs else "")
    return steps, f"{summary}, {parity_summary}"


def verify_grpo(text: str, args: argparse.Namespace) -> None:
    steps, summary = _verify_grpo_common(text, args)
    rewards = _series(steps, "critic/rewards/mean")
    if not rewards or max(rewards) <= 0.0:
        _fail(f"GRPO training reward was 0 on every step (no correct GSM8K answer was sampled): {rewards}")
    val_accs = _series(steps, "val-core/openai/gsm8k/acc/mean@1")
    if not val_accs:
        _fail("no val-core/openai/gsm8k/acc/mean@1 in the GRPO output (validation did not run).")
    # A 5-step run at lr 1e-6 does not move GSM8K accuracy, so the smoke test only requires that it ran.
    print(
        f"[TPU CI] GRPO verified: {max(steps)} steps, rewards {[round(r, 4) for r in rewards]}, "
        f"val acc {val_accs}, {summary}, max grad_norm {max(_series(steps, 'actor/grad_norm')):.4f}"
    )


def verify_grpo_4b(text: str, args: argparse.Namespace) -> None:
    steps, summary = _verify_grpo_common(text, args)
    # Validation before training is logged as step 0, the last one with the last training step.
    val_accs = _series(_step_metrics(text), "val-core/openai/gsm8k/acc/mean@1")
    if len(val_accs) < 2:
        _fail(f"need a validation before and after training, got GSM8K val acc {val_accs}.")
    first, last = val_accs[0], val_accs[-1]
    if not (last > first and last >= args.min_final_val_acc):
        _fail(f"GSM8K val acc did not improve to >= {args.min_final_val_acc}: {val_accs}")
    step_times = _series({s: m for s, m in steps.items() if s > 1}, "timing_s/step")
    rewards = _series(steps, "critic/rewards/mean")
    print(
        f"[TPU CI] GRPO 4B verified: {max(steps)} steps, val acc {' -> '.join(f'{v:.4f}' for v in val_accs)}, "
        f"reward {rewards[0]:.4f} -> {rewards[-1]:.4f} (max {max(rewards):.4f}), {summary}, "
        f"mean step time (steps 2+) {mean(step_times) if step_times else float('nan'):.1f} s"
    )


VERIFIERS = {
    "smoke": verify_smoke,
    "unit": verify_unit,
    "sft": verify_sft,
    "sft_resume": verify_sft_resume,
    "grpo": verify_grpo,
    "grpo_4b": verify_grpo_4b,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("test", choices=sorted(VERIFIERS))
    parser.add_argument("log_file")
    parser.add_argument("--steps", type=int, default=0, help="number of training steps the run must log")
    parser.add_argument("--min-final-val-acc", type=float, default=0.45, help="grpo_4b: final GSM8K val acc")
    parser.add_argument("--files", type=int, default=0, help="unit: number of test files that must have run")
    parser.add_argument("--saved-step", type=int, default=0, help="sft: step whose checkpoint the run must save")
    parser.add_argument(
        "--resumed-step", type=int, default=0, help="sft_resume: checkpoint step the run must resume from"
    )
    parser.add_argument(
        "--max-first-loss", type=float, default=0.0, help="sft_resume: bound on the first resumed train/loss"
    )
    args = parser.parse_args()
    with open(args.log_file, encoding="utf-8", errors="replace") as f:
        text = f.read()
    VERIFIERS[args.test](text, args)


if __name__ == "__main__":
    main()
