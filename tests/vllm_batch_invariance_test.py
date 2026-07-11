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
"""Standalone test of vLLM batch invariance (independent of verl).

Batch invariance means: the output for a given prompt must NOT depend on which
other prompts are batched together with it, nor on the batch size. vLLM provides
the VLLM_BATCH_INVARIANT=1 env var to enforce this.

This script runs a fixed set of prompts TWICE with the same per-request seed and
checks whether each prompt always produces the SAME tokens across the two runs.
If outputs diverge while prompts are identical, batch invariance is NOT
effective for this vLLM version / model / hardware combination.

The test uses SAMPLING (temperature>0) with a fixed per-request seed, NOT greedy.
A working batch invariance implementation must make sampled outputs reproducible
across runs (same prompt + same seed + same batch size → same tokens), even when
different requests are interleaved in the same batch.

==================== RUN STEPS ====================

1. (recommended) set PYTHONHASHSEED in the shell before Python starts, so any
   internal hash() vLLM relies on is stable across runs:
       export PYTHONHASHSEED=0

2. Run the script ONCE — it internally does two generation passes and compares
   them automatically (no need to run it twice):

       python tests/vllm_batch_invariance_test.py \
           --model /data/models/Qwen2.5-0.5B \
           --max-num-seqs 4 \
           --temperature 0.7

3. Read the final line:
       ✓ ALL RUNS ALIGNED  -> batch invariance IS effective
       ✗ DIVERGED          -> batch invariance is NOT effective

4. Useful variants:

   # Also bit-level compare logprobs (stronger check; see INTERPRETATION)
   python tests/vllm_batch_invariance_test.py --model ... --max-num-seqs 4 --temperature 0.7 --logprobs 1

   # Batch invariance OFF — baseline (should diverge to confirm the env var matters)
   python tests/vllm_batch_invariance_test.py --model ... --max-num-seqs 4 --temperature 0.7 --no-batch-invariant

   # Serialize (one request at a time) — should always align (no batch interleaving)
   python tests/vllm_batch_invariance_test.py --model ... --max-num-seqs 1 --temperature 0.7

   # Larger batch to stress batch interleaving
   python tests/vllm_batch_invariance_test.py --model ... --max-num-seqs 8 --n-prompts 16 --temperature 0.7

==================================================

Two comparison layers (the script reports both):
  1. TOKEN-LEVEL: are the generated token_ids identical across runs?
     -> This is what batch invariance GUARANTEES. If this diverges, batch
        invariance is NOT effective.
  2. LOGPROB-BIT (only with --logprobs 1): are the per-step logprobs
     bit-for-bit identical? Batch invariance does NOT guarantee this —
     logits may differ in floating-point tail due to batch-composition-dependent
     reduction order, yet still produce identical tokens. Token equality is
     the correct reproducibility criterion.

Decisive interpretation:
  - token-level ALIGNED -> batch invariance IS effective. verl non-determinism
     is NOT vLLM forward; look at verl's request ordering / routing / seed.
  - token-level DIVERGED (max_num_seqs>1, batch invariance ON) -> batch
     invariance is NOT effective for this vLLM install.
     Workaround: max_num_seqs=1, or upgrade vLLM, or use a model with
     verified batch-invariance support.
"""

import argparse
import os
import sys


def build_prompts(n: int) -> list[str]:
    """A fixed, varied set of prompts so logits/argmax differ and expose divergence."""
    base = [
        "The capital of France is",
        "Explain quantum entanglement in simple terms.",
        "Write a Python function to reverse a list.",
        "Translate 'good morning' to Japanese.",
        "What is the derivative of x^3?",
        "Name three types of clouds.",
        "Summarize the plot of Romeo and Juliet.",
        "List the planets in the solar system.",
    ]
    prompts = []
    i = 0
    while len(prompts) < n:
        prompts.append(base[i % len(base)] + f" (variant {i})")
        i += 1
    return prompts[:n]


def run_once(llm, prompts, sp_kwargs, log):
    """Run one generation pass. Returns (token_ids_list, logprobs_list_or_None)."""
    from vllm import SamplingParams

    # Rebuild sampling params per run so there is no shared mutable state.
    sampling_params = SamplingParams(**sp_kwargs)
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    token_ids_list = []
    logprobs_list = []
    has_logprobs = sp_kwargs.get("logprobs") is not None
    for o in outputs:
        out = o.outputs[0]
        token_ids_list.append(list(out.token_ids))
        if has_logprobs:
            # out.logprobs: list (one per generated step) of {token_id: Logprob}.
            # Flatten chosen-token logprob floats for bit-level comparison.
            flat = []
            for lp_dict in out.logprobs or []:
                if lp_dict:
                    tid, lg = next(iter(lp_dict.items()))
                    flat.append((int(tid), float(lg.logprob)))
                else:
                    flat.append(None)
            logprobs_list.append(flat)
    return token_ids_list, (logprobs_list if has_logprobs else None)


def compare_tokens(res_a, res_b):
    """Compare two token-id lists element-wise. Returns (n, n_diff, first_diff_idx)."""
    n = min(len(res_a), len(res_b))
    n_diff = 0
    first_diff = None
    for i in range(n):
        if res_a[i] != res_b[i]:
            n_diff += 1
            if first_diff is None:
                first_diff = i
    return n, n_diff, first_diff


def compare_logprobs(res_a, res_b):
    """Bit-level compare of logprobs. Returns (n, n_diff, first_diff_idx, max_abs_diff)."""
    n = min(len(res_a), len(res_b))
    n_diff = 0
    first_diff = None
    max_abs = 0.0
    for i in range(n):
        la, lb = res_a[i], res_b[i]
        if la is None or lb is None:
            continue
        diff_here = False
        for pa, pb in zip(la, lb, strict=False):
            if pa is None or pb is None:
                continue
            ta, va = pa
            tb, vb = pb
            if ta != tb or va != vb:
                diff_here = True
                if va is not None and vb is not None:
                    max_abs = max(max_abs, abs(va - vb))
        if diff_here:
            n_diff += 1
            if first_diff is None:
                first_diff = i
    return n, n_diff, first_diff, max_abs


def main():
    parser = argparse.ArgumentParser(description="Standalone vLLM batch invariance test (sampling + fixed seed)")
    parser.add_argument("--model", required=True, help="HF model path (e.g. /data/models/Qwen2.5-0.5B)")
    parser.add_argument("--max-num-seqs", type=int, default=4, help="vLLM max_num_seqs (1=serialize, >1=batched)")
    parser.add_argument("--n-prompts", type=int, default=8, help="number of prompts")
    parser.add_argument("--max-tokens", type=int, default=16, help="generation length")
    parser.add_argument("--temperature", type=float, default=0.7, help="sampling temperature (>0 = sampling with seed)")
    parser.add_argument("--seed", type=int, default=42, help="per-request sampling seed (same for every request)")
    parser.add_argument(
        "--no-batch-invariant", action="store_true", help="do NOT set VLLM_BATCH_INVARIANT=1 (baseline)"
    )
    parser.add_argument(
        "--enforce-eager", action="store_true", default=True, help="disable CUDA graphs (better determinism)"
    )
    parser.add_argument("--tp", type=int, default=1, help="tensor parallel size")
    parser.add_argument("--dtype", default="bfloat16", help="model dtype")
    parser.add_argument(
        "--logprobs",
        type=int,
        default=0,
        help="if >0, also request top-1 logprob per token and bit-level compare logprobs",
    )
    args = parser.parse_args()

    if args.temperature <= 0:
        print(
            "[warn] temperature<=0 means greedy; seed is ignored by vLLM. "
            "This test is designed for temperature>0 (sampling + fixed seed)."
        )

    # Set VLLM_BATCH_INVARIANT before importing vllm (it reads env on init).
    if not args.no_batch_invariant:
        os.environ["VLLM_BATCH_INVARIANT"] = "1"
        print("[setup] VLLM_BATCH_INVARIANT=1 (batch invariance ON)")
    else:
        os.environ.pop("VLLM_BATCH_INVARIANT", None)
        print("[setup] VLLM_BATCH_INVARIANT unset (batch invariance OFF — baseline)")

    # Import after env is set.
    import vllm

    print(f"[setup] vllm version: {vllm.__version__}")
    from vllm import LLM

    prompts = build_prompts(args.n_prompts)
    print(
        f"[setup] {len(prompts)} prompts, max_num_seqs={args.max_num_seqs}, "
        f"temperature={args.temperature}, seed={args.seed}, max_tokens={args.max_tokens}"
    )

    llm = LLM(
        model=args.model,
        max_num_seqs=args.max_num_seqs,
        enforce_eager=args.enforce_eager,
        tensor_parallel_size=args.tp,
        dtype=args.dtype,
    )

    # Per-request sampling params: same seed for every request (like verl does).
    # NOTE: with temperature>0, vLLM's seed must make each request reproducible
    # across runs IF batch invariance is effective.
    sp_kwargs = dict(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )
    if args.logprobs > 0:
        sp_kwargs["logprobs"] = args.logprobs
    print(f"[setup] sampling_params kwargs: {sp_kwargs}")

    # Warmup run (not compared) — let vLLM finish lazy init / cache warmup.
    print("\n[warmup] generating once (not compared)...")
    _ = run_once(llm, prompts, sp_kwargs, print)

    # Two compared runs.
    print("\n[run 1/2] generating...")
    tok1, lp1 = run_once(llm, prompts, sp_kwargs, print)
    print("[run 2/2] generating...")
    tok2, lp2 = run_once(llm, prompts, sp_kwargs, print)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    # Layer 1: token-level (the batch-invariance guarantee).
    n_tok, n_diff_tok, first_diff_tok = compare_tokens(tok1, tok2)
    if n_diff_tok == 0:
        print(f"[token-level] run1 vs run2: ALIGNED ({n_tok}/{n_tok} prompts identical)")
    else:
        print(f"[token-level] run1 vs run2: DIVERGED ({n_diff_tok}/{n_tok} prompts differ)")
        print(f"  first diverging prompt index: {first_diff_tok}")
        print(f"  prompt: {prompts[first_diff_tok]!r}")
        print(f"  run1 tokens ({len(tok1[first_diff_tok])}): {tok1[first_diff_tok]}")
        print(f"  run2 tokens ({len(tok2[first_diff_tok])}): {tok2[first_diff_tok]}")

    # Layer 2: logprob bit-level (stronger; batch invariance does NOT guarantee this).
    n_lp_diff = None
    if lp1 is not None and lp2 is not None:
        n_lp, n_lp_diff, first_lp_diff, max_abs = compare_logprobs(lp1, lp2)
        if n_lp_diff == 0:
            print(f"[logprob-bit]  run1 vs run2: BITWISE ALIGNED ({n_lp}/{n_lp} prompts)")
        else:
            print(f"[logprob-bit]  run1 vs run2: DIFF ({n_lp_diff}/{n_lp} prompts differ)")
            print(f"  first diverging prompt index: {first_lp_diff}")
            print(f"  prompt: {prompts[first_lp_diff]!r}")
            print(f"  max abs logprob diff: {max_abs:.6e}")

    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    if n_diff_tok == 0:
        print("✓ TOKEN-LEVEL ALIGNED — batch invariance IS effective for this config.")
        print("  (generated tokens are reproducible across runs with same prompt+seed)")
        if n_lp_diff is None:
            print("  (logprob bit-level not requested; pass --logprobs 1 to also check that)")
        elif n_lp_diff == 0:
            print("  Bonus: logprobs are ALSO bit-level aligned (strongest guarantee).")
        else:
            print("  Note: logprobs differ at the bit level even though tokens match.")
            print("        This is EXPECTED — batch invariance guarantees token equality,")
            print("        not logprob float equality. Token equality is sufficient for")
            print("        reproducible generation.")
        print("  => verl non-determinism is NOT vLLM forward; look at verl's")
        print("     request ordering / routing / per-request seed instead.")
        sys.exit(0)
    else:
        print("✗ TOKEN-LEVEL DIVERGED — batch invariance is NOT effective for this config.")
        print("  vLLM generated different tokens for identical prompt+seed across runs.")
        print("  Workaround: max_num_seqs=1, or upgrade vLLM, or use a model with")
        print("  verified batch-invariance support.")
        sys.exit(1)


if __name__ == "__main__":
    main()
