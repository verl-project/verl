#!/usr/bin/env python3
"""
rebuttal_significance.py — paired bootstrap + cross-seed mean±std for rebuttal.

=== TWO USE CASES ===

USE CASE A: You already have eval JSON files (single-seed runs).
  Extract per-sample BERTScore from existing JSONs, run bootstrap NOW.
  No new training needed.

  python rebuttal_significance.py --mode existing \
    --mednli   eval_04_student_checker_no_triage_s93.json \
    --gptnli   eval_2.5-7b-triage-guarded-29-12-03-step200.json \
    --baseline eval_01_baseline_zeroshot.json

USE CASE B: After running new seeds, aggregate mean±std + bootstrap.
  Each seed produces one eval JSON per config; put them in seed dirs:
    seeds/seed0/eval_mednli.json  seeds/seed1/eval_mednli.json  seeds/seed2/eval_mednli.json
    seeds/seed0/eval_gptnli.json  seeds/seed1/eval_gptnli.json  seeds/seed2/eval_gptnli.json

  python rebuttal_significance.py --mode seeds \
    --seed-dirs seeds/seed0 seeds/seed1 seeds/seed2 \
    --mednli-name  eval_mednli.json \
    --gptnli-name  eval_gptnli.json \
    --likelihood-name eval_likelihood.json   # optional

=== WHAT EACH STEP DOES (so you can explain it to reviewers) ===

Q1 — SAMPLE NOISE (paired bootstrap):
  "Is the 0.009 BERTScore gap real or just lucky sampling?"
  - For each of the 1479 test samples, we have a pair:
      (MedNLI-Cls score, GPT-NLI score)
  - Subtract → 1479 difference values (the "paired" part eliminates
    question-difficulty noise that affects both configs equally)
  - Bootstrap: resample these 1479 differences 10,000 times WITH replacement,
    compute mean each time → 10k means = uncertainty distribution
  - 95% CI: 2.5th–97.5th percentile of those 10k means
  - If CI excludes 0 → gap is unlikely to be sampling noise
  - If CI includes 0 → acknowledge gap is within noise, pivot to cascade argument

Q2 — TRAINING NOISE (cross-seed mean±std):
  "Does a different random seed flip the ordering?"
  - Run same config 3× with different seeds
  - Report mean ± std across seeds for BERTScore, F1, Support%
  - Key question: is the sign (MedNLI ≥ GPT-NLI) stable across all 3 seeds?
  - If yes: qualitative ordering is robust even if absolute gap is small

REBUTTAL STRATEGY: If BERTScore gap is NOT significant (likely!):
  → That's fine. Reframe finding (ii) as:
    "Moderate signal achieves COMPARABLE quality while avoiding cascade
    and requiring no GPT dependency — a Pareto advantage."
  → Cascade signals (length 394→130, 98% zero-search, 64% Chinese) are
    large-effect behavioral differences that seed variance cannot flip.
    These are your real, robust finding (ii).
"""

import argparse
import json
import os
import sys
import numpy as np
from typing import Dict, List, Optional, Tuple

# ── BERTScore model (same as compute_bertscore_all.py) ─────────────────────
BERT_MODEL  = "microsoft/deberta-xlarge-mnli"
BATCH_SIZE  = 8
HF_HOME     = "/ocean/projects/med230010p/yji3/.cache/huggingface"

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HF_HOME",               HF_HOME)
os.environ.setdefault("HF_HUB_CACHE",          os.path.join(HF_HOME, "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE",     os.path.join(HF_HOME, "transformers"))
os.environ.setdefault("HF_HUB_OFFLINE",        "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE",  "1")


# ════════════════════════════════════════════════════════════════════════════
# Core helpers
# ════════════════════════════════════════════════════════════════════════════

def load_results(path: str) -> List[Dict]:
    with open(path) as f:
        d = json.load(f)
    return d.get("results", [])


def compute_per_sample_bertscore(results: List[Dict],
                                  cache_path: Optional[str] = None) -> np.ndarray:
    """
    Return array of shape (n,) with per-sample BERTScore F1.
    Saves to cache_path if given (so you don't recompute next run).
    """
    if cache_path and os.path.exists(cache_path):
        arr = np.load(cache_path)
        print(f"  [cache] loaded {arr.shape[0]} scores from {cache_path}")
        return arr

    preds = [r.get("model_answer") or r.get("assistant_response") or "" for r in results]
    refs  = [r["golden_answers"][0] if r.get("golden_answers") else "" for r in results]

    from bert_score import score as bscore
    _, _, F = bscore(
        preds, refs,
        lang="en",
        model_type=BERT_MODEL,
        device="cpu",
        batch_size=BATCH_SIZE,
        verbose=False,
    )
    arr = F.numpy()

    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        np.save(cache_path, arr)
        print(f"  [saved] {cache_path}")

    return arr


def extract_scalar_metrics(results: List[Dict]) -> Dict[str, float]:
    """Non-BERTScore metrics computed from result fields."""
    n = len(results)
    f1          = np.mean([r.get("f1", 0) or 0 for r in results])
    support     = np.mean([r.get("checker_support_rate", 0) or 0 for r in results])
    tag_rate    = np.mean([1.0 if r.get("has_answer_tag") else 0.0 for r in results])
    avg_search  = np.mean([r.get("num_searches", 0) or 0 for r in results])
    ans_len     = np.mean([len(r.get("model_answer") or "") for r in results])
    return dict(n=n, f1=f1, support=support, tag_rate=tag_rate,
                avg_search=avg_search, ans_len=ans_len)


# ════════════════════════════════════════════════════════════════════════════
# Q1: Paired bootstrap (sample noise)
# ════════════════════════════════════════════════════════════════════════════

def paired_bootstrap(scores_a: np.ndarray, scores_b: np.ndarray,
                     n_boot: int = 10_000,
                     alpha: float = 0.05,
                     rng_seed: int = 42) -> Dict:
    """
    Test H0: mean(A - B) = 0 via paired bootstrap.

    Returns dict with:
      observed_diff  — the actual mean(A) - mean(B)
      ci_lo, ci_hi   — (1-alpha) confidence interval
      p_value        — fraction of bootstrap samples where diff ≤ 0 (one-sided)
                       or where |diff| ≥ |observed| (two-sided, reported)
      significant    — True if CI excludes 0
    """
    assert len(scores_a) == len(scores_b), "Arrays must be same length (paired)"
    rng   = np.random.default_rng(rng_seed)
    n     = len(scores_a)
    diffs = scores_a - scores_b                     # paired differences
    obs   = diffs.mean()

    # Bootstrap distribution of the mean difference
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)            # resample WITH replacement
        boot_means[i] = diffs[idx].mean()

    lo = np.percentile(boot_means, 100 * alpha / 2)
    hi = np.percentile(boot_means, 100 * (1 - alpha / 2))

    # Two-sided p-value: fraction where |boot| >= |obs| under H0 centered at 0
    boot_centered  = boot_means - boot_means.mean()
    p_val = (np.abs(boot_centered) >= np.abs(obs)).mean()

    return dict(
        observed_diff=float(obs),
        ci_lo=float(lo), ci_hi=float(hi),
        p_value=float(p_val),
        significant=(lo > 0 or hi < 0),        # CI excludes 0
        n=n,
    )


# ════════════════════════════════════════════════════════════════════════════
# Q2: Cross-seed aggregation (training noise)
# ════════════════════════════════════════════════════════════════════════════

def cross_seed_stats(score_arrays: List[np.ndarray]) -> Dict:
    """
    Given per-sample score arrays from k seeds, return mean±std across seeds.
    Each array has shape (n,); we first compute the per-seed mean, then
    aggregate across seeds.
    """
    per_seed_means = np.array([a.mean() for a in score_arrays])
    return dict(
        mean=float(per_seed_means.mean()),
        std=float(per_seed_means.std(ddof=1)),   # unbiased std
        n_seeds=len(score_arrays),
        per_seed=per_seed_means.tolist(),
    )


# ════════════════════════════════════════════════════════════════════════════
# Reporting
# ════════════════════════════════════════════════════════════════════════════

def print_bootstrap_result(name_a: str, name_b: str, res: Dict) -> None:
    sign = ">" if res["observed_diff"] > 0 else "<"
    print(f"\n{'─'*65}")
    print(f"  Paired bootstrap: {name_a} vs. {name_b}")
    print(f"  n = {res['n']} samples")
    print(f"  Observed diff: {res['observed_diff']:+.4f}  "
          f"({name_a} {sign} {name_b})")
    print(f"  95% CI:  [{res['ci_lo']:+.4f}, {res['ci_hi']:+.4f}]")
    print(f"  p-value: {res['p_value']:.4f}")
    if res["significant"]:
        print(f"  ✓ SIGNIFICANT — CI excludes 0.")
        print(f"    Rebuttal: 'Paired bootstrap on {res['n']} samples gives")
        print(f"    95% CI [{res['ci_lo']:+.4f}, {res['ci_hi']:+.4f}] (p={res['p_value']:.3f}),")
        print(f"    confirming the quality gap is not sampling noise.'")
    else:
        print(f"  ✗ NOT SIGNIFICANT — CI includes 0.")
        print(f"    Rebuttal: 'We acknowledge the {abs(res['observed_diff']):.3f} BERTScore gap")
        print(f"    is within sampling noise (95% CI [{res['ci_lo']:+.4f},{res['ci_hi']:+.4f}],")
        print(f"    p={res['p_value']:.3f}). We have accordingly reframed finding (ii):")
        print(f"    the moderate-signal checker achieves comparable quality while")
        print(f"    avoiding the reward-hacking cascade — a Pareto advantage over")
        print(f"    strong-signal GPT-NLI. The cascade evidence (answer length")
        print(f"    394→130 chars, ~98% zero-search, 64% non-English) represents")
        print(f"    large behavioral effects that seed variance cannot flip.'")
    print(f"{'─'*65}")


def print_seed_table(config_stats: Dict[str, Dict]) -> None:
    print(f"\n{'═'*70}")
    print(f"  Cross-seed BERTScore (mean ± std across seeds)")
    print(f"  {'Config':<20} {'Seeds':>5} {'Mean':>7} {'±Std':>7} {'Per-seed values'}")
    print(f"  {'─'*60}")
    for name, s in config_stats.items():
        per = "  ".join(f"{v:.4f}" for v in s["per_seed"])
        print(f"  {name:<20} {s['n_seeds']:>5} {s['mean']:>7.4f} {s['std']:>7.4f}   {per}")
    print(f"{'═'*70}")

    # Check sign stability
    names = list(config_stats.keys())
    if len(names) >= 2:
        a_vals = np.array(config_stats[names[0]]["per_seed"])
        b_vals = np.array(config_stats[names[1]]["per_seed"])
        stable = np.all(a_vals >= b_vals) or np.all(a_vals <= b_vals)
        direction = names[0] if a_vals.mean() > b_vals.mean() else names[1]
        print(f"\n  Sign stability ({names[0]} ≥ {names[1]}): "
              f"{'STABLE across all seeds ✓' if stable else 'NOT stable — order flips ✗'}")
        print(f"  Dominant: {direction}")
        if stable:
            print(f"  Rebuttal: 'The qualitative ordering is stable across all {a_vals.shape[0]}")
            print(f"  seeds, confirming that our single-seed finding reflects a real trend.'")
        else:
            print(f"  Rebuttal: 'The ordering flips across seeds, so we report only")
            print(f"  the behavioral cascade signals as our robust finding (ii).'")


# ════════════════════════════════════════════════════════════════════════════
# Mode A: Existing single-seed JSONs
# ════════════════════════════════════════════════════════════════════════════

def mode_existing(args):
    print("\n=== MODE A: Existing single-seed JSONs ===")
    print("(No new training needed — extracting per-sample BERTScore now)\n")

    configs = {}
    if args.mednli:
        configs["MedNLI-Cls"] = args.mednli
    if args.gptnli:
        configs["GPT-NLI"] = args.gptnli
    if args.likelihood:
        configs["Likelihood-NLI"] = args.likelihood
    if args.baseline:
        configs["Zero-shot"] = args.baseline

    if len(configs) < 2:
        print("ERROR: Need at least 2 configs (--mednli and --gptnli)")
        sys.exit(1)

    os.makedirs("rebuttal_scores", exist_ok=True)

    # Load per-sample BERTScores
    bs_arrays  = {}
    scalar_met = {}
    for name, path in configs.items():
        print(f"Loading: {name} from {path}")
        results = load_results(path)
        cache = f"rebuttal_scores/{name.lower().replace('-','_')}_seed0_bert.npy"
        bs_arrays[name]  = compute_per_sample_bertscore(results, cache_path=cache)
        scalar_met[name] = extract_scalar_metrics(results)

    # Print summary
    print(f"\n{'═'*70}")
    print(f"  Single-seed summary")
    print(f"  {'Config':<20} {'n':>5} {'BERT':>7} {'F1':>7} {'Supp%':>7} {'Tag%':>7} {'Search':>7} {'Len':>6}")
    print(f"  {'─'*65}")
    for name in configs:
        s  = scalar_met[name]
        bs = bs_arrays[name].mean()
        print(f"  {name:<20} {s['n']:>5} {bs:>7.4f} {s['f1']:>7.4f} "
              f"{s['support']:>7.1%} {s['tag_rate']:>7.1%} "
              f"{s['avg_search']:>7.3f} {s['ans_len']:>6.0f}")
    print(f"{'═'*70}")

    # Paired bootstrap: MedNLI-Cls vs GPT-NLI
    if "MedNLI-Cls" in bs_arrays and "GPT-NLI" in bs_arrays:
        res = paired_bootstrap(bs_arrays["MedNLI-Cls"], bs_arrays["GPT-NLI"])
        print_bootstrap_result("MedNLI-Cls", "GPT-NLI", res)

    # Also vs zero-shot baseline if provided
    if "Zero-shot" in bs_arrays and "MedNLI-Cls" in bs_arrays:
        res2 = paired_bootstrap(bs_arrays["MedNLI-Cls"], bs_arrays["Zero-shot"])
        print_bootstrap_result("MedNLI-Cls", "Zero-shot", res2)

    print("\n[Done] Per-sample scores saved to rebuttal_scores/*.npy")
    print("       Run with --mode seeds after retraining to add cross-seed stats.")


# ════════════════════════════════════════════════════════════════════════════
# Mode B: Multiple seed directories
# ════════════════════════════════════════════════════════════════════════════

def mode_seeds(args):
    print("\n=== MODE B: Cross-seed aggregation ===\n")

    if not args.seed_dirs:
        print("ERROR: Provide --seed-dirs dir0 dir1 dir2")
        sys.exit(1)

    name_map = {
        "MedNLI-Cls":      args.mednli_name,
        "GPT-NLI":         args.gptnli_name,
        "Likelihood-NLI":  args.likelihood_name,
    }
    name_map = {k: v for k, v in name_map.items() if v}

    os.makedirs("rebuttal_scores", exist_ok=True)

    seed_bs: Dict[str, List[np.ndarray]] = {name: [] for name in name_map}
    seed_sc: Dict[str, List[Dict]]       = {name: [] for name in name_map}

    for seed_i, seed_dir in enumerate(args.seed_dirs):
        for name, fname in name_map.items():
            path = os.path.join(seed_dir, fname)
            if not os.path.exists(path):
                print(f"  [skip] {path} not found")
                continue
            print(f"  seed{seed_i} {name}: {path}")
            results = load_results(path)
            cache = f"rebuttal_scores/{name.lower().replace('-','_')}_seed{seed_i}_bert.npy"
            arr = compute_per_sample_bertscore(results, cache_path=cache)
            seed_bs[name].append(arr)
            seed_sc[name].append(extract_scalar_metrics(results))

    # Cross-seed stats
    cs_bert = {}
    cs_f1   = {}
    cs_sup  = {}
    for name in name_map:
        if seed_bs[name]:
            cs_bert[name] = cross_seed_stats(seed_bs[name])
            cs_f1[name]   = cross_seed_stats(
                [np.full(len(s.get("f1", [0])), sc["f1"]) for sc in seed_sc[name]])
            # F1 is already a scalar per seed, use a simpler approach
            f1_per_seed = np.array([sc["f1"] for sc in seed_sc[name]])
            cs_f1[name]  = dict(mean=float(f1_per_seed.mean()),
                                 std=float(f1_per_seed.std(ddof=1)),
                                 n_seeds=len(f1_per_seed),
                                 per_seed=f1_per_seed.tolist())
            sup_per_seed = np.array([sc["support"] for sc in seed_sc[name]])
            cs_sup[name] = dict(mean=float(sup_per_seed.mean()),
                                 std=float(sup_per_seed.std(ddof=1)),
                                 n_seeds=len(sup_per_seed),
                                 per_seed=sup_per_seed.tolist())

    # Print rebuttal table
    print(f"\n{'═'*80}")
    print("  REBUTTAL TABLE: Cross-seed results (paste into response)")
    print(f"  {'Config':<20} {'BERTScore':>18} {'F1':>18} {'Support%':>16}")
    print(f"  {'─'*75}")
    for name in name_map:
        if name in cs_bert:
            b = cs_bert[name]
            f = cs_f1[name]
            s = cs_sup[name]
            print(f"  {name:<20} {b['mean']:.3f} ± {b['std']:.3f}    "
                  f"{f['mean']:.3f} ± {f['std']:.3f}    "
                  f"{s['mean']:.1%} ± {s['std']:.1%}")
    print(f"{'═'*80}")

    # Paired bootstrap on per-sample arrays (use mean across seeds for pooled test)
    if "MedNLI-Cls" in seed_bs and "GPT-NLI" in seed_bs:
        if seed_bs["MedNLI-Cls"] and seed_bs["GPT-NLI"]:
            # Pool all seeds together for maximum power
            pooled_mednli = np.concatenate(seed_bs["MedNLI-Cls"])
            pooled_gptnli = np.concatenate(seed_bs["GPT-NLI"])

            # If same samples evaluated per seed, take mean across seeds instead
            if len(seed_bs["MedNLI-Cls"]) == len(seed_bs["GPT-NLI"]):
                arr_m = np.stack(seed_bs["MedNLI-Cls"], axis=0).mean(axis=0)
                arr_g = np.stack(seed_bs["GPT-NLI"],    axis=0).mean(axis=0)
                res = paired_bootstrap(arr_m, arr_g)
                print_bootstrap_result("MedNLI-Cls (mean of seeds)",
                                       "GPT-NLI (mean of seeds)", res)

    # Sign stability
    print_seed_table(cs_bert)

    # Copy-paste block for rebuttal
    print("\n" + "═"*70)
    print("  COPY-PASTE BLOCK FOR REBUTTAL (fill in [] with actual values):")
    print("─"*70)
    for name in name_map:
        if name in cs_bert:
            b = cs_bert[name]
            print(f"  {name}: BERTScore {b['mean']:.3f} ± {b['std']:.3f} "
                  f"(seeds: {', '.join(f'{v:.3f}' for v in b['per_seed'])})")


# ════════════════════════════════════════════════════════════════════════════
# Argument parsing
# ════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--mode", choices=["existing", "seeds"], default="existing",
                    help="existing: use current single-seed JSONs; "
                         "seeds: aggregate across multiple seed dirs")

    # Mode A args
    ap.add_argument("--mednli",   help="Path to MedNLI-Cls eval JSON")
    ap.add_argument("--gptnli",   help="Path to GPT-NLI eval JSON")
    ap.add_argument("--likelihood", help="Path to Likelihood-NLI eval JSON (optional)")
    ap.add_argument("--baseline", help="Path to zero-shot baseline eval JSON (optional)")

    # Mode B args
    ap.add_argument("--seed-dirs", nargs="+",
                    help="Directories containing per-seed eval JSONs")
    ap.add_argument("--mednli-name",     default="eval_mednli.json")
    ap.add_argument("--gptnli-name",     default="eval_gptnli.json")
    ap.add_argument("--likelihood-name", default=None)

    args = ap.parse_args()

    if args.mode == "existing":
        mode_existing(args)
    else:
        mode_seeds(args)


if __name__ == "__main__":
    main()
