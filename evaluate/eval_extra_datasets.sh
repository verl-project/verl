#!/bin/bash
# ============================================================
# eval_extra_datasets.sh
# Batch eval of 5 core configs on all extra eval datasets.
#
# Usage:
#   bash evaluate/eval_extra_datasets.sh
# ============================================================
set -euo pipefail

VERL_ROOT="/ocean/projects/med230010p/yji3/BrowseCamp/verl"
EXTRA_DIR="$VERL_ROOT/searchr1_data/extra_eval"

# 5 core configs: (experiment_name, step)
declare -A CONFIGS
CONFIGS=(
    ["model_06_student_explicit_no_triage_s188"]="188"
    ["model_14_gpt_checker_format_penalty_BEST_s200"]="200"
    ["merged_2.5-7b-triage-guarded-29-15-02-step200"]="200"
    ["merged_llama-3.1-8b-checker-guarded-30-22-50-step200"]="200"
)

# Extra datasets: (parquet_path, suffix)
DATASETS=(
    "$EXTRA_DIR/medicationqa_test.parquet _medicationqa"
    "$EXTRA_DIR/bioasq_test.parquet _bioasq"
    "$EXTRA_DIR/medquad_full_test.parquet _medquad"
)

echo "Starting batch eval: ${#CONFIGS[@]} configs × ${#DATASETS[@]} datasets"
echo "Total runs: $(( ${#CONFIGS[@]} * ${#DATASETS[@]} ))"
echo ""

for exp in "${!CONFIGS[@]}"; do
    step="${CONFIGS[$exp]}"
    for dataset_entry in "${DATASETS[@]}"; do
        path=$(echo "$dataset_entry" | awk '{print $1}')
        suffix=$(echo "$dataset_entry" | awk '{print $2}')
        ds_name=$(basename "$path" .parquet)

        echo "============================================================"
        echo "Running: $exp (step $step) on $ds_name"
        echo "============================================================"

        SKIP_MERGE=1 bash evaluate/eval.sh \
            "$exp" "$step" "$path" "$suffix" \
            2>&1 | tee "eval_log_${exp}_${ds_name}.txt"

        echo "Done: $exp on $ds_name"
        echo ""
    done
done

echo "All evals complete. Collecting results..."

python3 << 'PYEOF'
import json, glob, os
from pathlib import Path

results = []
for f in sorted(glob.glob("eval_*_medicationqa.json") +
                glob.glob("eval_*_bioasq.json") +
                glob.glob("eval_*_medquad.json")):
    try:
        with open(f) as fp:
            d = json.load(fp)
        m = d['metrics']
        results.append({
            'file':          f,
            'n':             len(d['results']),
            'f1':            m.get('f1_mean', 0),
            'rouge_l':       m.get('rouge_l', 0),
            'bert_score':    m.get('bert_score', 0),
            'support_rate':  m.get('avg_checker_support_rate', 0),
            'tag_rate':      m.get('has_answer_tag_rate', 0),
            'faithfulness':  m.get('rag_faithfulness', 0),
            'claim_prec':    m.get('rag_claim_precision', 0),
        })
    except Exception as e:
        print(f"Failed to read {f}: {e}")

print(f"\n{'Config':<50} {'Dataset':<12} {'F1':>6} {'BERT':>6} {'Supp%':>6} {'Faith':>6}")
print("─" * 95)
for r in results:
    name = r['file'].replace('eval_', '').replace('.json', '')
    parts = name.split('_')
    ds = parts[-1] if parts[-1] in ['medicationqa','bioasq','medquad'] else 'unknown'
    exp = name.replace('_' + ds, '')
    exp_short = exp[-40:] if len(exp) > 40 else exp
    print(f"{exp_short:<50} {ds:<12} {r['f1']:>6.3f} {r['bert_score']:>6.3f} {r['support_rate']:>6.1%} {r['faithfulness']:>6.3f}")

# Save summary
with open("extra_eval_summary.json", 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSummary saved to extra_eval_summary.json")
PYEOF
