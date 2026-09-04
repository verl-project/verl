#!/bin/bash
# ============================================================
# eval_batch_gpt5mini.sh
# Same as eval_batch.sh but uses gpt-5-mini checker (port 8005)
# Results saved with _gpt5mini suffix to avoid overwriting existing results
#
# Usage:
#   export CUDA_VISIBLE_DEVICES=3
#   bash evaluate/eval_batch_gpt5mini.sh 2>&1 | tee -a eval_batch_gpt5mini.log
# ============================================================
set -euo pipefail

VERL_ROOT="/ocean/projects/med230010p/yji3/BrowseCamp/verl"
EXTRA_DIR="$VERL_ROOT/searchr1_data/extra_eval"
MEDICAL_TEST="/ocean/projects/med230010p/yji3/MedicalRagChecker/verl/searchr1_data/combined__medical/test.parquet"
RESULTS_CSV="eval_all_results_gpt5mini.csv"

# Override checker URL to port 8005
export CHECKER_URL="http://127.0.0.1:8005"

# Write CSV header if needed
if [[ ! -f "$RESULTS_CSV" ]]; then
    echo "experiment,dataset,n,f1,rouge_l,bert_score,tag_pct,search,support_pct,faithfulness,claim_prec,checker_model" \
        > "$RESULTS_CSV"
fi

EXTRA_DATASETS=(
    "$EXTRA_DIR/medicationqa_test.parquet _medicationqa"
    "$EXTRA_DIR/bioasq_test.parquet _bioasq"
    "$EXTRA_DIR/medquad_full_test.parquet _medquad"
    "$EXTRA_DIR/mediqa_test.parquet _mediqa"
)

# 5 core models only (enough for paper comparison)
ALL_MODELS=(
    "merged_2.5-7b-triage-guarded-29-15-02-step200 200"
    "model_06_student_explicit_no_triage_s188 188"
    "model_14_gpt_checker_format_penalty_BEST_s200 200"
    "merged_llama-3.1-8b-checker-guarded-30-22-50-step200 200"
    "model_09_hybrid_gpt_extract_meditron_nli_s377 377"
)

append_to_csv() {
    local eval_out="$1"
    local dataset_name="$2"
    python3 - "$eval_out" "$dataset_name" "$RESULTS_CSV" << 'PYEOF'
import json, sys, csv, os
eval_out, dataset, csv_path = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    with open(eval_out) as f: d = json.load(f)
    m = d['metrics']
    exp = os.path.basename(eval_out).replace('eval_','').replace(f'_{dataset}.json','').replace('.json','').replace('_gpt5mini','')
    row = {
        'experiment':    exp,
        'dataset':       dataset,
        'n':             len(d['results']),
        'f1':            round(m.get('f1_mean',0),4),
        'rouge_l':       round(m.get('rouge_l',0),4),
        'bert_score':    round(m.get('bert_score',0),4),
        'tag_pct':       round(m.get('has_answer_tag_rate',0)*100,1),
        'search':        round(m.get('avg_searches',0),3),
        'support_pct':   round(m.get('avg_checker_support_rate',0)*100,1),
        'faithfulness':  round(m.get('rag_faithfulness',0),3),
        'claim_prec':    round(m.get('rag_claim_precision',0),3),
        'checker_model': 'gpt-5-mini',
    }
    already = False
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            for line in f:
                if row['experiment'] in line and row['dataset'] in line:
                    already = True; break
    if not already:
        with open(csv_path,'a',newline='') as f:
            csv.DictWriter(f, fieldnames=list(row.keys())).writerow(row)
        print(f"  → CSV: {row['experiment']}/{dataset} F1={row['f1']} Faith={row['faithfulness']}")
except Exception as e:
    print(f"  CSV error: {e}")
PYEOF
}

run_eval() {
    local exp="$1" step="$2" test_file="$3" suffix="$4"
    # Use _gpt5mini suffix to keep results separate
    local out="eval_${exp}${suffix}_gpt5mini.json"
    local ds_name="${suffix#_}"

    if [[ -f "$out" ]]; then
        echo "  SKIP: $out"
        append_to_csv "$out" "$ds_name"
        return 0
    fi
    if [[ ! -f "$test_file" ]]; then return 0; fi

    echo "  Running [gpt-5-mini]: $exp | $ds_name"
    if SKIP_MERGE=1 \
       CHECKER_URL="http://127.0.0.1:8005" \
       bash evaluate/eval.sh "$exp" "$step" "$test_file" "${suffix}_gpt5mini" \
       > "eval_log_${exp}${suffix}_gpt5mini.txt" 2>&1; then
        echo "  ✓ $out"
        append_to_csv "$out" "$ds_name"
    else
        echo "  ✗ FAILED: eval_log_${exp}${suffix}_gpt5mini.txt"
    fi
}

echo "============================================================"
echo "Batch eval with gpt-5-mini checker (port 8005)"
echo "Results: $RESULTS_CSV"
echo "============================================================"

# Verify checker is running
if ! curl -s http://127.0.0.1:8005/health > /dev/null 2>&1; then
    echo "ERROR: gpt-5-mini checker not running on port 8005"
    echo "Start it with:"
    echo "  nohup python search_r1_preprocess/checker_gpt5mini.py --mode openai --port 8005 &"
    exit 1
fi
echo "Checker status: $(curl -s http://127.0.0.1:8005/health)"

for model_entry in "${ALL_MODELS[@]}"; do
    exp=$(echo "$model_entry" | awk '{print $1}')
    step=$(echo "$model_entry" | awk '{print $2}')
    echo ""
    echo "── $exp ──"
    run_eval "$exp" "$step" "$MEDICAL_TEST" ""
    for ds_entry in "${EXTRA_DATASETS[@]}"; do
        ds_path=$(echo "$ds_entry" | awk '{print $1}')
        ds_suffix=$(echo "$ds_entry" | awk '{print $2}')
        run_eval "$exp" "$step" "$ds_path" "$ds_suffix"
    done
done

echo ""
echo "Done. Results in: $RESULTS_CSV"

# Compare gpt-4o-mini vs gpt-5-mini
python3 << 'PYEOF'
import csv, os

def read_csv(path):
    if not os.path.exists(path): return {}
    rows = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            rows[(r['experiment'], r['dataset'])] = r
    return rows

old = read_csv("eval_all_results.csv")
new = read_csv("eval_all_results_gpt5mini.csv")

common = set(old.keys()) & set(new.keys())
if not common:
    print("No common results to compare yet.")
else:
    print(f"\n{'Experiment':<45} {'Dataset':<12} {'Faith(4o)':>9} {'Faith(5)':>8} {'Diff':>6}")
    print("─" * 85)
    for k in sorted(common):
        o, n = old[k], new[k]
        fo = float(o.get('faithfulness', 0))
        fn = float(n.get('faithfulness', 0))
        exp_s = k[0][-43:] if len(k[0]) > 43 else k[0]
        print(f"{exp_s:<45} {k[1]:<12} {fo:>9.3f} {fn:>8.3f} {fn-fo:>+6.3f}")
PYEOF
