#!/usr/bin/env python3
"""
# 复制到服务器上
cp compute_bertscore_all.py \
   /ocean/projects/med230010p/yji3/BrowseCamp/verl/

# 运行（约15-20分钟，CPU跑所有文件）
cd /ocean/projects/med230010p/yji3/BrowseCamp/verl
export CUDA_VISIBLE_DEVICES=""
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/ocean/projects/med230010p/yji3/.cache/huggingface

python compute_bertscore_all.py 2>&1 | tee bertscore_all.log

脚本会做什么

自动找所有 eval_*.json 文件
计算每个文件的 ROUGE-L 和 BERTScore
写回每个 JSON 文件（metrics.bert_score 和 metrics.rouge_l）
打印完整对比表格
保存 bertscore_summary.csv，直接可以用来填论文表格


compute_bertscore_all.py
计算所有 eval JSON 文件的 BERTScore，填补 '--' 的空缺。

Usage:
    cd /ocean/projects/med230010p/yji3/BrowseCamp/verl
    export CUDA_VISIBLE_DEVICES=""
    export HF_HOME=/ocean/projects/med230010p/yji3/.cache/huggingface
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    python compute_bertscore_all.py


    Format+search bonus: BERTScore = 0.5761

Triage+GPT English: BERTScore = 0.6007

(vllm07) [yji3@w002 verl]这里是 当前的结果 还有 

===============================================================================================
Config                                            F1    R-L   BERT   Tag%  Search  Supp%
===============================================================================================
01_baseline_zeroshot                           0.191  0.106  0.538 100.0%   0.000   0.0%
02_search_only                                 0.190  0.122  0.537  58.6%   0.655   0.0%
03_search_triage                               0.116  0.088  0.507   5.7%   0.000   0.0%
04_student_checker_no_triage_s93               0.210  0.136  0.595  98.9%   0.046   0.0%
05_student_checker_triage_s93                  0.219  0.150  0.593 100.0%   0.000   0.0%
06_student_explicit_no_triage_s188             0.198  0.135  0.589  96.6%   0.172   0.0%
07_student_explicit_triage_s188                0.205  0.143  0.602 100.0%   0.092   0.0%
08_diag_evidence_fix_768tok                    0.190  0.129  0.589  93.1%   0.299  35.7%
09_diag_hybrid_gpt_extract_meditron_nli        0.194  0.133  0.591  95.4%   0.207  37.5%
Llama GPT checker                              0.154  0.035  0.324  13.8%   0.000   0.0%
10_diag_hybrid_triage                          0.204  0.139  0.600  98.9%   0.069  17.9%
11_gpt_checker_no_penalty_s377                 0.168  0.114  0.531  75.9%   1.023  65.6%
12_gpt_checker_no_penalty_v2_s377              0.160  0.112  0.553  82.8%   0.782  77.5%
Format penalty only                            0.174  0.107  0.504  59.8%   0.483   0.0%
Format+GPT (BEST)                              0.194  0.130  0.589  98.9%   0.023  75.0%
Format+GPT+bonus                               0.171  0.116  0.576 100.0%   0.966  71.1%
Triage+GPT (Chinese issue)                     0.078  0.053  0.455  97.7%   0.724  68.9%
Triage+GPT (English, no bonus)                 0.203  0.141  0.601 100.0%   0.034 100.0%
Triage+GPT+bonus (BEST)                        0.212  0.133  0.565  90.8%   0.621  87.2%
checker_guarded_step751_explicit               0.150  0.076  0.406  42.5%   0.471  29.4%
Student checker (no triage, s93)               0.210  0.136  0.595  98.9%   0.046   0.0%
Student checker + triage (s93)                 0.219  0.150  0.593 100.0%   0.000   0.0%
Baseline (zero-shot)                           0.191  0.106  0.538 100.0%   0.000   0.0%
combined_search_checker_triage_step93          0.199  0.131  0.573  95.4%   0.000   0.0%
debug_sample                                   0.083  0.068  0.454 100.0%   0.000   0.0%
debug_validate                                 0.083  0.068  0.454 100.0%   0.000   0.0%
force_search                                   0.164  0.111  0.565  71.3%   0.333   0.0%
hybrid_checker_guarded_step377                 0.169  0.119  0.543  86.2%   0.264  66.7%
hybrid_checker_no_triage                       0.210  0.136  0.596  97.7%   0.000   0.0%
hybrid_checker_triage                          0.216  0.145  0.590  98.9%   0.023  50.0%
Hybrid no triage (s188)                        0.194  0.133  0.591  95.4%   0.207  37.5%
Hybrid triage (s188)                           0.204  0.139  0.600  98.9%   0.069  17.9%
Llama student checker                          0.218  0.149  0.600 100.0%   0.000   0.0%
llama-3.1-8b-checker-guarded-31-11-45-step200  0.154  0.034  0.324  12.6%   0.023   0.0%
model_06_student_explicit_no_triage_s188_hotpotqa  0.070  0.066  0.465  68.0%   0.020   0.0%
model_07_student_explicit_triage_s188_hotpotqa  0.071  0.068  0.486 100.0%   0.020 100.0%
Evidence fix (768 tok)                         0.190  0.129  0.589  93.1%   0.299  35.7%
no_triage_explicitcheck_after_fix              0.200  0.137  0.589  95.4%   0.253   0.0%
Student explicit no triage (s188)              0.198  0.135  0.589  96.6%   0.172   0.0%
no_triage_explicitcheck_step188                0.197  0.136  0.587  95.4%   0.207   0.0%
GPT checker old (s377)                         0.168  0.114  0.531  75.9%   1.023  65.6%
GPT checker new (s377)                         0.160  0.112  0.553  82.8%   0.782  77.5%
qwen2.5-7b-checker-guarded-ablation-27-16-40-step200  0.194  0.130  0.589  98.9%   0.023  75.0%
qwen2.5-7b-checker-guarded-ablation-28-11-20-step200  0.171  0.116  0.576 100.0%   0.966  71.1%
qwen2.5-7b-checker-guarded-ablation-28-13-06-step200  0.196  0.122  0.528  65.5%   0.621  84.6%
qwen2.5-7b-no_checker-guarded-ablation-27-14-24-step200  0.174  0.107  0.504  59.8%   0.483   0.0%
Search only                                    0.190  0.122  0.537  58.6%   0.655   0.0%
Search + triage                                0.116  0.088  0.507   5.7%   0.000   0.0%
Student explicit triage (s188)                 0.205  0.143  0.602 100.0%   0.092   0.0%
triage_explicitcheck_step188                   0.203  0.137  0.597  98.9%   0.207   0.0%

Summary saved to: bertscore_summary.csv
All BERTScores written back to eval JSON files.
(vllm07) [yji3@w002 verl]$ 
(vllm07) [yji3@w002 verl]$ 
"""

import json, os, glob, sys

# ── Config ───────────────────────────────────────────────────
EVAL_DIR    = "."          # directory containing eval_*.json
BERT_MODEL  = "microsoft/deberta-xlarge-mnli"
BATCH_SIZE  = 8
DEVICE      = "cpu"        # CPU to avoid OOM

os.environ["CUDA_VISIBLE_DEVICES"]  = ""
os.environ["HF_HOME"]               = "/ocean/projects/med230010p/yji3/.cache/huggingface"
os.environ["HF_HUB_CACHE"]          = "/ocean/projects/med230010p/yji3/.cache/huggingface/hub"
os.environ["TRANSFORMERS_CACHE"]    = "/ocean/projects/med230010p/yji3/.cache/huggingface/transformers"
os.environ["HF_HUB_OFFLINE"]        = "1"
os.environ["TRANSFORMERS_OFFLINE"]  = "1"

# ── Map eval file → human-readable label (for the summary table) ──
LABEL_MAP = {
    "eval_clean.json":                                              "Baseline (zero-shot)",
    "eval_search_only_no_triage.json":                             "Search only",
    "eval_search_only_triage.json":                                "Search + triage",
    "eval_checker_no_triage.json":                                 "Student checker (no triage, s93)",
    "eval_checker_triage_after_fix.json":                          "Student checker + triage (s93)",
    "eval_no_triage_explicitcheck_after_fix_v2.json":              "Student explicit no triage (s188)",
    "eval_triage_explicitcheck_after_fix_v2.json":                 "Student explicit triage (s188)",
    "eval_no_triage_explicitcheck_after_evidence_fix.json":        "Evidence fix (768 tok)",
    "eval_hybrid_no_triage_explicit_step188.json":                 "Hybrid no triage (s188)",
    "eval_hybrid_triage_explicit_step188.json":                    "Hybrid triage (s188)",
    "eval_openai_checker_guarded_step377.json":                    "GPT checker old (s377)",
    "eval_qwen2.5-7b-checker-guarded-ablation-26-19-48-step377.json": "GPT checker new (s377)",
    "eval_13_format_penalty_only_s200.json":                       "Format penalty only",
    "eval_14_gpt_checker_format_penalty_s200.json":                "Format+GPT (BEST)",
    "eval_15_gpt_checker_format_search_bonus_s200.json":           "Format+GPT+bonus",
    "eval_qwen2_5-7b-no_checker-guarded-ablation-27-14-24-step200.json": "Format penalty only (27)",
    "eval_qwen2_5-7b-checker-guarded-ablation-27-16-40-step200.json":    "Format+GPT (27)",
    "eval_qwen2_5-7b-checker-guarded-ablation-28-11-20-step200.json":    "Format+GPT+bonus(len50)",
    "eval_qwen2_5-7b-checker-guarded-ablation-28-13-06-step200.json":    "Format+GPT+bonus(len120)",
    "eval_2.5-7b-triage-guarded-28-15-49-step200.json":            "Triage+GPT (Chinese issue)",
    "eval_2.5-7b-triage-guarded-29-12-03-step200.json":            "Triage+GPT (English, no bonus)",
    "eval_2.5-7b-triage-guarded-29-15-02-step200.json":            "Triage+GPT+bonus (BEST)",
    "eval_llama-3.1-8b-checker-guarded-30-22-50-step200.json":     "Llama student checker",
    "eval_0e9e39f249a16976918f-checker-guarded-31-11-45-step200.json": "Llama GPT checker",
}

def load_bert_scorer():
    from bert_score import score as bscore
    print(f"BERTScore model: {BERT_MODEL} (device={DEVICE})")
    return bscore

def rouge_l(ref, pred):
    def lcs(x, y):
        m, n = len(x), len(y)
        if not m or not n: return 0
        prev = [0] * (n + 1)
        for i in range(1, m + 1):
            curr = [0] * (n + 1)
            for j in range(1, n + 1):
                curr[j] = prev[j-1] + 1 if x[i-1] == y[j-1] else max(curr[j-1], prev[j])
            prev = curr
        return prev[n]
    rt, pt = ref.lower().split(), pred.lower().split()
    if not rt or not pt: return 0.0
    l = lcs(rt, pt)
    p, r = l / len(pt), l / len(rt)
    return 2 * p * r / (p + r) if p + r else 0.0

def process_file(fpath, bscore_fn):
    with open(fpath) as f:
        d = json.load(f)

    m       = d.get("metrics", {})
    results = d.get("results", [])
    if not results:
        return None

    preds = [r.get("model_answer") or "" for r in results]
    refs  = [r["golden_answers"][0] if r.get("golden_answers") else "" for r in results]

    # Skip if BERTScore already computed (non-zero)
    existing_bs = m.get("bert_score", 0.0)

    # Compute ROUGE-L
    rl = sum(rouge_l(ref, pred) for ref, pred in zip(refs, preds)) / len(preds)

    # Compute BERTScore
    valid = [(p, r) for p, r in zip(preds, refs) if p and r]
    bs = 0.0
    if valid:
        vp, vr = zip(*valid)
        try:
            _, _, F = bscore_fn(
                list(vp), list(vr),
                lang="en",
                model_type=BERT_MODEL,
                device=DEVICE,
                batch_size=BATCH_SIZE,
                verbose=False,
            )
            bs = F.mean().item()
        except Exception as e:
            print(f"  BERTScore failed: {e}")

    # Save back
    m["rouge_l"]    = round(rl, 4)
    m["bert_score"] = round(bs, 4)
    d["metrics"] = m
    with open(fpath, "w") as f:
        json.dump(d, f, indent=2)

    return {
        "f1":        m.get("f1_mean", 0),
        "rouge_l":   rl,
        "bert_score": bs,
        "tag":        m.get("has_answer_tag_rate", 0),
        "search":     m.get("avg_searches", 0),
        "support":    m.get("avg_checker_support_rate", 0),
        "n":          len(results),
    }

def main():
    # Find all eval JSON files
    all_files = sorted(glob.glob(os.path.join(EVAL_DIR, "eval_*.json")))
    if not all_files:
        print(f"No eval_*.json found in {EVAL_DIR}")
        sys.exit(1)

    print(f"Found {len(all_files)} eval files\n")

    # Load BERTScore
    bscore_fn = load_bert_scorer()

    # Process each file
    results_table = []
    for fpath in all_files:
        fname = os.path.basename(fpath)
        label = LABEL_MAP.get(fname, fname.replace("eval_", "").replace(".json", ""))
        print(f"Processing: {label} ...", flush=True)

        try:
            res = process_file(fpath, bscore_fn)
            if res is None:
                print(f"  SKIP (empty results)")
                continue
            results_table.append((label, res))
            print(f"  F1={res['f1']:.4f}  R-L={res['rouge_l']:.4f}  BERT={res['bert_score']:.4f}  "
                  f"Tag={res['tag']:.1%}  Search={res['search']:.3f}  Support={res['support']:.1%}")
        except Exception as e:
            print(f"  ERROR: {e}")

    # Print final summary table
    print("\n" + "=" * 95)
    print(f"{'Config':<45} {'F1':>6} {'R-L':>6} {'BERT':>6} {'Tag%':>6} {'Search':>7} {'Supp%':>6}")
    print("=" * 95)
    for label, res in results_table:
        print(f"{label:<45} {res['f1']:>6.3f} {res['rouge_l']:>6.3f} {res['bert_score']:>6.3f} "
              f"{res['tag']:>6.1%} {res['search']:>7.3f} {res['support']:>6.1%}")

    # Save summary CSV
    csv_path = "bertscore_summary.csv"
    with open(csv_path, "w") as f:
        f.write("Config,F1,ROUGE-L,BERTScore,Tag%,AvgSearch,Support%\n")
        for label, res in results_table:
            f.write(f"{label},{res['f1']:.4f},{res['rouge_l']:.4f},{res['bert_score']:.4f},"
                    f"{res['tag']:.4f},{res['search']:.4f},{res['support']:.4f}\n")
    print(f"\nSummary saved to: {csv_path}")
    print("All BERTScores written back to eval JSON files.")

if __name__ == "__main__":
    main()
