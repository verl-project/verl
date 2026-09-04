#!/bin/bash
# ============================================================
# eval.sh — merge + evaluate + ROUGE + BERTScore + RAGChecker
#
# Usage:
#   bash eval.sh <experiment_name> <step> [test_file] [output_suffix]
#
# experiment_name can be:
#   - the raw experiment name (will prepend merged_ and append -stepN)
#     e.g.  2.5-7b-triage-guarded-29-15-02
#   - an already-merged directory name (used as-is)
#     e.g.  merged_2.5-7b-triage-guarded-29-15-02-step200
#          model_06_student_explicit_no_triage_s188
#          model_14_gpt_checker_format_penalty_BEST_s200
#
# Examples:
#   SKIP_MERGE=1 bash eval.sh merged_2.5-7b-triage-guarded-29-15-02-step200 200
#   SKIP_MERGE=1 bash eval.sh model_14_gpt_checker_format_penalty_BEST_s200 200
#   SKIP_MERGE=1 bash eval.sh merged_2.5-7b-triage-guarded-29-15-02-step200 200 \
#       searchr1_data/extra_eval/medicationqa_test.parquet _medicationqa
# ============================================================
set -euo pipefail

EXPERIMENT="${1:-}"
STEP="${2:-}"
TEST_FILE="${3:-/ocean/projects/med230010p/yji3/MedicalRagChecker/verl/searchr1_data/combined__medical/test.parquet}"
OUTPUT_SUFFIX="${4:-}"

if [[ -z "$EXPERIMENT" || -z "$STEP" ]]; then
    echo "Usage: bash eval.sh <experiment_name> <step> [test_file] [output_suffix]"
    exit 1
fi

SKIP_MERGE="${SKIP_MERGE:-0}"

# ── Paths ────────────────────────────────────────────────────
VERL_ROOT="/ocean/projects/med230010p/yji3/BrowseCamp/verl"
MODELS_DIR="$VERL_ROOT/merged_models"
CKPT_DIR="$VERL_ROOT/checkpoints/search_r1_like_async_rl/${EXPERIMENT}/global_step_${STEP}/actor"
TOOL_CONFIG="$VERL_ROOT/examples/sglang_multiturn/config/tool_config/medical_search_checker_tool_config.yaml"

# ── Resolve MERGED_DIR ───────────────────────────────────────
# If experiment name already exists as a directory in merged_models, use it directly.
# This handles:
#   merged_2.5-7b-triage-guarded-29-15-02-step200  → use as-is
#   model_06_student_explicit_no_triage_s188        → use as-is
#   2.5-7b-triage-guarded-29-15-02                  → prepend merged_ and append -stepN
if [[ -d "$MODELS_DIR/$EXPERIMENT" ]]; then
    MERGED_DIR="$MODELS_DIR/$EXPERIMENT"
else
    MERGED_DIR="$MODELS_DIR/merged_${EXPERIMENT}-step${STEP}"
fi

EVAL_OUT="eval_${EXPERIMENT}${OUTPUT_SUFFIX}.json"

# ── Env ──────────────────────────────────────────────────────
unset ROCR_VISIBLE_DEVICES
module load cuda
export PYTHONPATH=$VERL_ROOT:${PYTHONPATH:-}
export CUDA_HOME=/opt/packages/cuda/v12.6.1
export CUDA_PATH=/opt/packages/cuda/v12.6.1
export LD_LIBRARY_PATH=/opt/packages/cuda/v12.6.1/lib64:/opt/packages/cuda/v12.6.1/nvvm/lib64:/opt/packages/cuda/v12.6.1/extras/CUPTI/lib64:$LD_LIBRARY_PATH
export PATH=/opt/packages/cuda/v12.6.1/bin:$PATH
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/ocean/projects/med230010p/yji3/.cache/huggingface
export HF_DATASETS_CACHE=/ocean/projects/med230010p/yji3/.cache/huggingface/datasets
export TRANSFORMERS_CACHE=/ocean/projects/med230010p/yji3/.cache/huggingface/transformers
export HF_HUB_CACHE=/ocean/projects/med230010p/yji3/.cache/huggingface/hub

# ── Auto-detect Llama ────────────────────────────────────────
TEMPLATE_ARG=""
if echo "$EXPERIMENT" | grep -qi "llama"; then
    TEMPLATE_ARG="--use_inference_chat_template"
    echo "Detected Llama model: using inference chat template"
elif echo "$EXPERIMENT" | grep -qE "^[0-9a-f]{20}"; then
    TEMPLATE_ARG="--use_inference_chat_template"
    echo "Detected snapshot hash (likely Llama): using inference chat template"
fi

GPU_MEM_UTIL="0.4"
if [[ -n "$TEMPLATE_ARG" ]]; then
    GPU_MEM_UTIL="0.35"
fi

# ── Step 1: Merge ────────────────────────────────────────────
if [[ "$SKIP_MERGE" == "1" ]]; then
    echo "Skipping merge. Using: $MERGED_DIR"
    if [[ ! -d "$MERGED_DIR" ]]; then
        echo "ERROR: merged model not found: $MERGED_DIR"
        echo "Available models:"
        ls "$MODELS_DIR/"
        exit 1
    fi
else
    echo "============================================================"
    echo "STEP 1/4  MERGE"
    echo "  from: $CKPT_DIR"
    echo "  to:   $MERGED_DIR"
    echo "============================================================"
    if [[ ! -d "$CKPT_DIR" ]]; then
        echo "ERROR: checkpoint not found: $CKPT_DIR"
        exit 1
    fi
    CUDA_VISIBLE_DEVICES=3 python -m verl.model_merger merge \
        --backend fsdp \
        --local_dir "$CKPT_DIR" \
        --target_dir "$MERGED_DIR"
    echo "Merged to: $MERGED_DIR"
fi

# ── Step 2: Eval ─────────────────────────────────────────────
echo ""
echo "============================================================"
echo "STEP 2/4  EVAL"
echo "  model:    $MERGED_DIR"
echo "  data:     $TEST_FILE"
echo "  output:   $EVAL_OUT"
echo "  template: ${TEMPLATE_ARG:-default (Qwen)}"
echo "============================================================"

CUDA_VISIBLE_DEVICES=3 python evaluate/evaluate_search_r1.py \
    --repo_root "$VERL_ROOT" \
    --model_path "$MERGED_DIR" \
    --test_file "$TEST_FILE" \
    --max_samples 100 \
    --eval_batch_size 4 \
    --output_file "$EVAL_OUT" \
    --tool_count_mode both \
    --tag_style auto \
    --prompt_mode explicit_check \
    --tool_config_path "$TOOL_CONFIG" \
    --multi_turn_format search_r1_with_checker \
    --tensor_parallel_size 1 --nnodes 1 --n_gpus_per_node 1 \
    --gpu_memory_utilization $GPU_MEM_UTIL \
    --max_model_len 8000 \
    --max_prompt_length 3072 \
    --max_response_length 2000 \
    --max_assistant_turns 5 \
    --max_tool_response_length 768 \
    $TEMPLATE_ARG

# ── Step 3: ROUGE + BERTScore ────────────────────────────────
echo ""
echo "============================================================"
echo "STEP 3/4  ROUGE + BERTScore"
echo "============================================================"
export CUDA_VISIBLE_DEVICES=""

python3 << PYEOF
import json, os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HOME"] = "/ocean/projects/med230010p/yji3/.cache/huggingface"
os.environ["HF_HUB_CACHE"] = "/ocean/projects/med230010p/yji3/.cache/huggingface/hub"
os.environ["TRANSFORMERS_CACHE"] = "/ocean/projects/med230010p/yji3/.cache/huggingface/transformers"

with open("$EVAL_OUT") as f:
    d = json.load(f)
m       = d['metrics']
results = d['results']
preds   = [r.get('model_answer') or '' for r in results]
refs    = [r['golden_answers'][0] if r.get('golden_answers') else '' for r in results]

# ROUGE
r1 = r2 = rl = 0.0
try:
    from rouge_score import rouge_scorer
    rouge = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)
    scores = [rouge.score(ref, pred) for ref, pred in zip(refs, preds) if ref and pred]
    if scores:
        r1 = sum(s['rouge1'].fmeasure for s in scores) / len(scores)
        r2 = sum(s['rouge2'].fmeasure for s in scores) / len(scores)
        rl = sum(s['rougeL'].fmeasure for s in scores) / len(scores)
    print(f"ROUGE-1: {r1:.4f}  ROUGE-2: {r2:.4f}  ROUGE-L: {rl:.4f}")
except Exception as e:
    print(f"ROUGE failed: {e}")

# BERTScore
bs = 0.0
print("Computing BERTScore (~3 min)...")
try:
    from bert_score import score as bscore
    valid = [(p, r) for p, r in zip(preds, refs) if p and r]
    if valid:
        vp, vr = zip(*valid)
        _, _, F = bscore(list(vp), list(vr), lang='en',
                         model_type='microsoft/deberta-xlarge-mnli',
                         device='cpu', batch_size=8, verbose=False)
        bs = F.mean().item()
    print(f"BERTScore: {bs:.4f}")
except Exception as e:
    print(f"BERTScore failed: {e}")

m['rouge1'] = round(r1, 4)
m['rouge2'] = round(r2, 4)
m['rouge_l'] = round(rl, 4)
m['bert_score'] = round(bs, 4)
with open("$EVAL_OUT", 'w') as f:
    json.dump(d, f, indent=2)
PYEOF

# ── Step 4: RAGChecker-style metrics ─────────────────────────
echo ""
echo "============================================================"
echo "STEP 4/4  RAGChecker Faithfulness Metrics"
echo "============================================================"

python3 << PYEOF
import json
with open("$EVAL_OUT") as f:
    d = json.load(f)
m = d['metrics']
results = d['results']

faithful_count = total_with_checker = 0
claim_supports = []
claim_totals = []
for r in results:
    cs = r.get('checker_total_supports', 0) or 0
    cc = r.get('checker_total_contradictions', 0) or 0
    cn = r.get('checker_total_neutrals', 0) or 0
    total = cs + cc + cn
    if total > 0:
        total_with_checker += 1
        claim_supports.append(cs)
        claim_totals.append(total)
        if cc == 0:
            faithful_count += 1

faithfulness   = faithful_count / total_with_checker if total_with_checker > 0 else 0.0
claim_prec     = sum(claim_supports) / sum(claim_totals) if sum(claim_totals) > 0 else 0.0
grounded       = sum(1 for r in results if r.get('num_searches',0)>0 and r.get('checker_supports',0)>0)
grounded_rate  = grounded / len(results) if results else 0.0
search_check   = sum(1 for r in results if r.get('num_searches',0)>0 and r.get('num_checks',0)>0)
retrieval_util = search_check / len(results) if results else 0.0
length_ratios  = []
for r in results:
    pred = r.get('model_answer') or ''
    ref  = (r.get('golden_answers') or [''])[0]
    if ref and pred:
        length_ratios.append(min(len(pred.split()) / max(len(ref.split()),1), 2.0))
completeness = sum(length_ratios)/len(length_ratios) if length_ratios else 0.0

m['rag_faithfulness']    = round(faithfulness,   4)
m['rag_claim_precision'] = round(claim_prec,     4)
m['rag_grounded_rate']   = round(grounded_rate,  4)
m['rag_retrieval_util']  = round(retrieval_util, 4)
m['rag_completeness']    = round(completeness,   4)

with open("$EVAL_OUT", 'w') as f:
    json.dump(d, f, indent=2)

f1       = m.get('f1_mean', 0)
rl       = m.get('rouge_l', 0)
bs       = m.get('bert_score', 0)
tag_rate = m.get('has_answer_tag_rate', 0)
sup_rate = m.get('avg_checker_support_rate', 0)
avg_s    = m.get('avg_searches', 0)

print("\n============================================================")
print("FINAL SUMMARY")
print("============================================================")
print(f"Experiment:       $EXPERIMENT")
print(f"Dataset:          $(basename $TEST_FILE)")
print(f"N samples:        {len(results)}")
print(f"--- Answer Quality ---")
print(f"Token F1:         {f1:.4f}   {'✓' if f1>0.19 else '✗'}")
print(f"ROUGE-L:          {rl:.4f}")
print(f"BERTScore:        {bs:.4f}   {'✓' if bs>0.58 else '✗'}")
print(f"--- RAGChecker ---")
print(f"Faithfulness:     {faithfulness:.3f}")
print(f"Claim precision:  {claim_prec:.3f}")
print(f"Grounded rate:    {grounded_rate:.3f}")
print(f"--- Behavior ---")
print(f"Answer tag rate:  {tag_rate:.1%}   {'✓' if tag_rate>0.90 else '✗'}")
print(f"Support rate:     {sup_rate:.1%}   {'✓' if sup_rate>0.4 else '✗ collapse'}")
print(f"Avg searches:     {avg_s:.3f}")
print(f"\nSaved: $EVAL_OUT")
PYEOF