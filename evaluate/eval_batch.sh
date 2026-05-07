#!/bin/bash
# ============================================================
# eval_batch.sh — priority-ordered batch eval
# All results go to eval_results/ subdirectory.
# Completed JSONs are skipped automatically.
#
# Usage:
#   export CUDA_VISIBLE_DEVICES=3
#   bash evaluate/eval_batch.sh 2>&1 | tee -a eval_batch.log
# ============================================================
set -euo pipefail

VERL_ROOT="/ocean/projects/med230010p/yji3/BrowseCamp/verl"
EXTRA_DIR="$VERL_ROOT/searchr1_data/extra_eval"
MEDICAL_TEST="/ocean/projects/med230010p/yji3/MedicalRagChecker/verl/searchr1_data/combined__medical/test.parquet"
SKIP_EXTRA="${SKIP_EXTRA:-0}"

# ── All outputs go here ───────────────────────────────────────
export EVAL_DIR="$VERL_ROOT/eval_results"
mkdir -p "$EVAL_DIR"

RESULTS_CSV="$EVAL_DIR/eval_all_results.csv"

# Write header if needed
if [[ ! -f "$RESULTS_CSV" ]]; then
    echo "experiment,dataset,n,f1,rouge_l,bert_score,tag_pct,search,support_pct,faithfulness,claim_prec,grounded_rate,retrieval_util" \
        > "$RESULTS_CSV"
    echo "Created: $RESULTS_CSV"
fi

EXTRA_DATASETS=(
    "$EXTRA_DIR/medicationqa_test.parquet _medicationqa"
    "$EXTRA_DIR/bioasq_test.parquet _bioasq"
    "$EXTRA_DIR/medquad_full_test.parquet _medquad"
    "$EXTRA_DIR/mediqa_test.parquet _mediqa"
)

ALL_MODELS=(
    # ★ Core story — run first
    "merged_2.5-7b-triage-guarded-29-15-02-step200 200"
    "model_06_student_explicit_no_triage_s188 188"
    "model_14_gpt_checker_format_penalty_BEST_s200 200"
    "merged_llama-3.1-8b-checker-guarded-30-22-50-step200 200"
    "model_09_hybrid_gpt_extract_meditron_nli_s377 377"
    # Supporting
    "model_07_student_explicit_triage_s188 188"
    "model_11_gpt_checker_no_penalty_s377 377"
    "model_12_gpt_checker_no_penalty_v2_s377c 377"
    "model_13_format_penalty_only_s200 200"
    "model_15a_gpt_checker_format_searchbonus_len50_s200 200"
    "model_15b_gpt_checker_format_searchbonus_len120_s200 200"
    "model_16_triage_guarded_gpt_checker_s200 200"
    "merged_2.5-7b-triage-guarded-29-12-03-step200 200"
    "merged_llama-3.1-8b-checker-guarded-31-11-45-step200 200"
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
    basename = os.path.basename(eval_out).replace('eval_','').replace('.json','')
    # Strip dataset suffix from experiment name
    exp = basename
    if dataset and basename.endswith('_' + dataset):
        exp = basename[:-len('_' + dataset)]

    row = {
        'experiment':    exp,
        'dataset':       dataset if dataset else 'medical',
        'n':             len(d['results']),
        'f1':            round(m.get('f1_mean',0), 4),
        'rouge_l':       round(m.get('rouge_l',0), 4),
        'bert_score':    round(m.get('bert_score',0), 4),
        'tag_pct':       round(m.get('has_answer_tag_rate',0)*100, 1),
        'search':        round(m.get('avg_searches',0), 3),
        'support_pct':   round(m.get('avg_checker_support_rate',0)*100, 1),
        'faithfulness':  round(m.get('rag_faithfulness',0), 3),
        'claim_prec':    round(m.get('rag_claim_precision',0), 3),
        'grounded_rate': round(m.get('rag_grounded_rate',0), 3),
        'retrieval_util':round(m.get('rag_retrieval_util',0), 3),
    }
    # Avoid duplicates
    already = False
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            for line in f:
                if row['experiment'] in line and row['dataset'] in line:
                    already = True; break
    if not already:
        with open(csv_path, 'a', newline='') as f:
            csv.DictWriter(f, fieldnames=list(row.keys())).writerow(row)
        print(f"  → CSV: {row['experiment']}/{row['dataset']} F1={row['f1']} Faith={row['faithfulness']}")
    else:
        print(f"  → Already in CSV: {row['experiment']}/{row['dataset']}")
except Exception as e:
    print(f"  CSV append error: {e}")
PYEOF
}

run_eval() {
    local exp="$1" step="$2" test_file="$3" suffix="$4"
    local out="$EVAL_DIR/eval_${exp}${suffix}.json"
    local ds_name="${suffix#_}"

    if [[ -f "$out" ]]; then
        echo "  SKIP (exists): $(basename $out)"
        append_to_csv "$out" "$ds_name"
        return 0
    fi
    if [[ ! -f "$test_file" ]]; then
        echo "  SKIP (no dataset): $(basename $test_file)"
        return 0
    fi

    echo "  Running: $exp | dataset=${ds_name:-medical}"
    if SKIP_MERGE=1 EVAL_DIR="$EVAL_DIR" \
       bash evaluate/eval.sh "$exp" "$step" "$test_file" "$suffix" \
       > "$EVAL_DIR/log_${exp}${suffix}.txt" 2>&1; then
        echo "  ✓ Done: $(basename $out)"
        append_to_csv "$out" "$ds_name"
    else
        echo "  ✗ FAILED — see $EVAL_DIR/log_${exp}${suffix}.txt"
    fi
}

echo "============================================================"
echo "Batch eval | $(date)"
echo "Output dir: $EVAL_DIR"
echo "CSV:        $RESULTS_CSV"
echo "============================================================"

for model_entry in "${ALL_MODELS[@]}"; do
    exp=$(echo  "$model_entry" | awk '{print $1}')
    step=$(echo "$model_entry" | awk '{print $2}')
    echo ""
    echo "── $exp (step $step) ──"
    run_eval "$exp" "$step" "$MEDICAL_TEST" ""
    if [[ "$SKIP_EXTRA" != "1" ]]; then
        for ds_entry in "${EXTRA_DATASETS[@]}"; do
            ds_path=$(echo   "$ds_entry" | awk '{print $1}')
            ds_suffix=$(echo "$ds_entry" | awk '{print $2}')
            run_eval "$exp" "$step" "$ds_path" "$ds_suffix"
        done
    fi
done

echo ""
echo "============================================================"
echo "Done | $(date)"
echo "============================================================"

# Print summary table
python3 << 'PYEOF'
import csv, os

csv_path = os.environ.get('EVAL_DIR', 'eval_results') + '/eval_all_results.csv'
try:
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    print(f"\n{'Experiment':<50} {'Dataset':<12} {'F1':>6} {'BERT':>6} {'Tag%':>5} {'Sup%':>5} {'Faith':>6}")
    print("─" * 100)
    for r in rows:
        exp = r['experiment'][-48:] if len(r['experiment']) > 48 else r['experiment']
        print(f"{exp:<50} {r['dataset']:<12} {float(r['f1']):>6.3f} "
              f"{float(r['bert_score']):>6.3f} {float(r['tag_pct']):>5.1f} "
              f"{float(r['support_pct']):>5.1f} {float(r['faithfulness']):>6.3f}")
    print(f"\nTotal: {len(rows)} rows | {csv_path}")
except Exception as e:
    print(f"Could not print table: {e}")
PYEOF
