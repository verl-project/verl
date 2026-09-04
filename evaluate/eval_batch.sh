#!/bin/bash
# ============================================================
# eval_batch.sh — priority-ordered batch eval
# All results go to eval_results/ subdirectory.
# Completed JSONs are skipped automatically.
#
# Usage:
# export CUDA_VISIBLE_DEVICES=3
# export EVAL_DIR=/ocean/projects/med230010p/yji3/BrowseCamp/verl/eval_results
# bash evaluate/eval_batch.sh 2>&1 | tee -a eval_batch.log
# ============================================================
set -euo pipefail

VERL_ROOT="/ocean/projects/med230010p/yji3/BrowseCamp/verl"
EXTRA_DIR="$VERL_ROOT/searchr1_data/extra_eval"
MEDICAL_TEST="/ocean/projects/med230010p/yji3/MedicalRagChecker/verl/searchr1_data/combined__medical/test.parquet"
SKIP_EXTRA="${SKIP_EXTRA:-0}"

export EVAL_DIR="${EVAL_DIR:-$VERL_ROOT/eval_results}"
mkdir -p "$EVAL_DIR"

RESULTS_CSV="$EVAL_DIR/eval_all_results.csv"

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
    # ★ P1 Core story
    "merged_2.5-7b-triage-guarded-29-15-02-step200 200"
    "model_06_student_explicit_no_triage_s188 188"
    "model_14_gpt_checker_format_penalty_BEST_s200 200"
    "merged_llama-3.1-8b-checker-guarded-30-22-50-step200 200"
    "model_09_hybrid_gpt_extract_meditron_nli_s377 377"
    # ★ P2 Supporting
    "model_07_student_explicit_triage_s188 188"
    "model_11_gpt_checker_no_penalty_s377 377"
    "model_12_gpt_checker_no_penalty_v2_s377c 377"
    "model_13_format_penalty_only_s200 200"
    "model_15a_gpt_checker_format_searchbonus_len50_s200 200"
    "model_15b_gpt_checker_format_searchbonus_len120_s200 200"
    "model_16_triage_guarded_gpt_checker_s200 200"
    "merged_2.5-7b-triage-guarded-29-12-03-step200 200"
    "merged_llama-3.1-8b-checker-guarded-31-11-45-step200 200"
    # ★ P3 New models
    "merged_qwen3-4b-checker-guarded-18-18-39-step200 200"
    "model_old_checker_guarded_s751 751"
    # ★ G-group: Qwen2.5-7B + PubMedBERT NLI checker
    "merged_G1_pubmedbert_mednli_explicit_only_seed42_23-15-57-step100 100"
    "merged_G2_pubmedbert_mednli_guarded_seed42_23-11-50-step200 200"
    # ★ Qwen3-4B + PubMedBERT NLI checker (for Table 1 cross-model rows)
    "merged_qwen3_4b_G1_explicit_only_step200 200"
    "merged_qwen3_4b_G2_guarded_step200 200"
    "merged_qwen3_4b_G1_explicit_only_step377 377"
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
    results = d['results']
    basename = os.path.basename(eval_out).replace('eval_','').replace('.json','')
    exp = basename
    if dataset and basename.endswith('_' + dataset):
        exp = basename[:-len('_' + dataset)]

    # Recompute faithfulness from per-sample fields (correct field names)
    fc = tw = 0
    cs_list, ct_list = [], []
    for r in results:
        cs = r.get('checker_total_supports',      0) or 0
        cc = r.get('checker_total_contradictions',0) or 0
        cn = r.get('checker_total_neutrals',      0) or 0
        total = cs + cc + cn
        if total > 0:
            tw += 1
            cs_list.append(cs)
            ct_list.append(total)
            if cc == 0: fc += 1
    faith = fc / tw if tw > 0 else 0.0
    cp    = sum(cs_list) / sum(ct_list) if sum(ct_list) > 0 else 0.0
    grounded = sum(1 for r in results
                   if r.get('num_searches',0)>0 and r.get('checker_total_supports',0)>0)
    sc = sum(1 for r in results
             if r.get('num_searches',0)>0 and r.get('num_checks',0)>0)

    row = {
        'experiment':    exp,
        'dataset':       dataset if dataset else 'medical',
        'n':             len(results),
        'f1':            round(m.get('f1_mean',0), 4),
        'rouge_l':       round(m.get('rouge_l',0), 4),
        'bert_score':    round(m.get('bert_score',0), 4),
        'tag_pct':       round(m.get('has_answer_tag_rate',0)*100, 1),
        'search':        round(m.get('avg_searches',0), 3),
        'support_pct':   round(m.get('avg_checker_support_rate',0)*100, 1),
        'faithfulness':  round(faith, 4),
        'claim_prec':    round(cp, 4),
        'grounded_rate': round(grounded/len(results) if results else 0, 4),
        'retrieval_util':round(sc/len(results) if results else 0, 4),
    }
    already = False
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            for line in f:
                if row['experiment'] in line and row['dataset'] in line:
                    already = True; break
    if not already:
        with open(csv_path, 'a', newline='') as f:
            csv.DictWriter(f, fieldnames=list(row.keys())).writerow(row)
        print(f"  → {row['experiment']}/{row['dataset']} F1={row['f1']} Faith={row['faithfulness']}")
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
        echo "  SKIP: $(basename $out)"
        append_to_csv "$out" "$ds_name"
        return 0
    fi
    if [[ ! -f "$test_file" ]]; then
        echo "  SKIP (no dataset): $(basename $test_file)"
        return 0
    fi

    echo "  Running: $exp | ${ds_name:-medical}"
    if SKIP_MERGE=1 EVAL_DIR="$EVAL_DIR" \
       bash evaluate/eval.sh "$exp" "$step" "$test_file" "$suffix" \
       > "$EVAL_DIR/log_${exp}${suffix}.txt" 2>&1; then
        echo "  ✓ $(basename $out)"
        append_to_csv "$out" "$ds_name"
    else
        echo "  ✗ FAILED — $EVAL_DIR/log_${exp}${suffix}.txt"
    fi
}

echo "============================================================"
echo "Batch eval | $(date)"
echo "Output: $EVAL_DIR"
echo "Models: ${#ALL_MODELS[@]}"
echo "============================================================"

for model_entry in "${ALL_MODELS[@]}"; do
    exp=$(echo  "$model_entry" | awk '{print $1}')
    step=$(echo "$model_entry" | awk '{print $2}')
    echo ""
    echo "── $exp ──"
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

python3 << 'PYEOF'
import csv, os
csv_path = os.environ.get('EVAL_DIR','eval_results') + '/eval_all_results.csv'
try:
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    print(f"\n{'Experiment':<50} {'DS':<12} {'F1':>6} {'BERT':>6} {'Sup%':>5} {'Faith':>6}")
    print("─"*95)
    for r in rows:
        exp = r['experiment'][-48:] if len(r['experiment'])>48 else r['experiment']
        print(f"{exp:<50} {r['dataset']:<12} {float(r['f1']):>6.3f} "
              f"{float(r['bert_score']):>6.3f} {float(r['support_pct']):>5.1f} "
              f"{float(r['faithfulness']):>6.3f}")
    print(f"\nTotal: {len(rows)} | {csv_path}")
except Exception as e:
    print(f"Table error: {e}")
PYEOF