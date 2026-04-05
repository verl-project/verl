#!/bin/bash
# ============================================================
# eval.sh — merge + evaluate + ROUGE + BERTScore + RAGChecker metrics
# Auto-detects Llama models and uses correct chat template.
#
# Usage:
#   bash eval.sh <experiment_name> <step> [test_file] [output_suffix]
#
# Examples:
#   # Medical dataset (default)
#   bash eval.sh 2.5-7b-triage-guarded-29-15-02 200
#
#   # Extra eval datasets
#   SKIP_MERGE=1 bash eval.sh 2.5-7b-triage-guarded-29-15-02 200 \
#       /ocean/projects/med230010p/yji3/BrowseCamp/verl/searchr1_data/extra_eval/medicationqa_test.parquet \
#       _medicationqa
#
#   # Llama model (auto-detected)
#   bash eval.sh llama-3.1-8b-checker-guarded-30-22-44 200
#
#   # Skip merge
#   SKIP_MERGE=1 bash eval.sh 2.5-7b-triage-guarded-29-15-02 200
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
CKPT_DIR="$VERL_ROOT/checkpoints/search_r1_like_async_rl/${EXPERIMENT}/global_step_${STEP}/actor"
MERGED_DIR="$VERL_ROOT/merged_models/merged_${EXPERIMENT}-step${STEP}"
EVAL_OUT="eval_${EXPERIMENT}-step${STEP}${OUTPUT_SUFFIX}.json"
TOOL_CONFIG="$VERL_ROOT/examples/sglang_multiturn/config/tool_config/medical_search_checker_tool_config.yaml"

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

# ── Auto-detect Llama model ──────────────────────────────────
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
    echo "Skipping merge (SKIP_MERGE=1), using: $MERGED_DIR"
else
    echo "============================================================"
    echo "STEP 1/4  MERGE"
    echo "  from: $CKPT_DIR"
    echo "  to:   $MERGED_DIR"
    echo "============================================================"

    if [[ ! -d "$CKPT_DIR" ]]; then
        echo "ERROR: checkpoint not found: $CKPT_DIR"
        echo "Available steps:"
        ls "$VERL_ROOT/checkpoints/search_r1_like_async_rl/${EXPERIMENT}/" 2>/dev/null || true
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
echo "  model:      $MERGED_DIR"
echo "  data:       $TEST_FILE"
echo "  output:     $EVAL_OUT"
echo "  template:   ${TEMPLATE_ARG:-default (Qwen)}"
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
echo "STEP 3/4  ROUGE-L + BERTScore"
echo "============================================================"

export CUDA_VISIBLE_DEVICES=""

python3 << PYEOF
import json, os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HOME"] = "/ocean/projects/med230010p/yji3/.cache/huggingface"
os.environ["HF_DATASETS_CACHE"] = "/ocean/projects/med230010p/yji3/.cache/huggingface/datasets"
os.environ["TRANSFORMERS_CACHE"] = "/ocean/projects/med230010p/yji3/.cache/huggingface/transformers"
os.environ["HF_HUB_CACHE"] = "/ocean/projects/med230010p/yji3/.cache/huggingface/hub"

with open("$EVAL_OUT") as f:
    d = json.load(f)
m       = d['metrics']
results = d['results']
preds   = [r.get('model_answer') or '' for r in results]
refs    = [r['golden_answers'][0] if r.get('golden_answers') else '' for r in results]

f1        = m.get('f1_mean', 0)
tag_rate  = m.get('has_answer_tag_rate', 0)
avg_s     = m.get('avg_searches', 0)
avg_c     = m.get('avg_checks', 0)
sup_rate  = m.get('avg_checker_support_rate', 0)
avg_turns = m.get('avg_turns', 0)

# ROUGE
r1 = r2 = rl = 0.0
try:
    from rouge_score import rouge_scorer
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = [rouge.score(ref, pred) for ref, pred in zip(refs, preds) if ref and pred]
    if scores:
        r1 = sum(s['rouge1'].fmeasure for s in scores) / len(scores)
        r2 = sum(s['rouge2'].fmeasure for s in scores) / len(scores)
        rl = sum(s['rougeL'].fmeasure for s in scores) / len(scores)
    print(f"ROUGE-1: {r1:.4f}  ROUGE-2: {r2:.4f}  ROUGE-L: {rl:.4f}")
except Exception as e:
    print(f"ROUGE failed: {e}")

# BERTScore (CPU)
bs = 0.0
print("Computing BERTScore on CPU (~3 min)...")
try:
    from bert_score import score as bscore
    valid = [(p, r) for p, r in zip(preds, refs) if p and r]
    if valid:
        vp, vr = zip(*valid)
        _, _, F = bscore(list(vp), list(vr), lang='en',
                         model_type='microsoft/deberta-xlarge-mnli',
                         device='cpu', batch_size=8, verbose=False)
        bs = F.mean().item()
    print(f"BERTScore (F1): {bs:.4f}")
except Exception as e:
    print(f"BERTScore failed: {e}")

# Save standard metrics
m['rouge1']     = round(r1, 4)
m['rouge2']     = round(r2, 4)
m['rouge_l']    = round(rl, 4)
m['bert_score'] = round(bs, 4)
with open("$EVAL_OUT", 'w') as f:
    json.dump(d, f, indent=2)
print(f"Standard metrics saved to: $EVAL_OUT")
PYEOF

# ── Step 4: RAGChecker-style metrics ─────────────────────────
echo ""
echo "============================================================"
echo "STEP 4/4  RAGChecker-style Faithfulness Metrics"
echo "============================================================"

export CUDA_VISIBLE_DEVICES=""

python3 << PYEOF
import json, os, re

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HOME"] = "/ocean/projects/med230010p/yji3/.cache/huggingface"
os.environ["HF_HUB_CACHE"] = "/ocean/projects/med230010p/yji3/.cache/huggingface/hub"

with open("$EVAL_OUT") as f:
    d = json.load(f)

results = d['results']
m = d['metrics']

# ── RAGChecker metric 1: Faithfulness ────────────────────────
# For each sample, check if model answer is supported by its tool responses
# We use the checker logs already embedded in the rollout

faithful_count = 0
total_with_checker = 0
claim_supports = []
claim_totals   = []

for r in results:
    cs = r.get('checker_supports',       0) or 0
    cc = r.get('checker_contradictions', 0) or 0
    cn = r.get('checker_neutrals',       0) or 0
    total_claims = cs + cc + cn
    if total_claims > 0:
        total_with_checker += 1
        claim_supports.append(cs)
        claim_totals.append(total_claims)
        # Faithful = no contradictions
        if cc == 0:
            faithful_count += 1

faithfulness = faithful_count / total_with_checker if total_with_checker > 0 else 0.0
claim_precision = sum(claim_supports) / sum(claim_totals) if sum(claim_totals) > 0 else 0.0

# ── RAGChecker metric 2: Grounded Answer Rate ─────────────────
# Fraction of samples where model both searched AND got checker support
grounded = sum(1 for r in results
               if r.get('num_searches', 0) > 0
               and r.get('checker_supports', 0) > 0)
grounded_rate = grounded / len(results) if results else 0.0

# ── RAGChecker metric 3: Retrieval Utilization ────────────────
# Did the model actually use retrieved evidence in its answer?
# Proxy: samples with searches > 0 that also have checker calls
search_and_check = sum(1 for r in results
                       if r.get('num_searches', 0) > 0
                       and r.get('num_checks',   0) > 0)
retrieval_util = search_and_check / len(results) if results else 0.0

# ── RAGChecker metric 4: Answer Completeness (length ratio) ───
# Ratio of answer length to reference length
length_ratios = []
for r in results:
    pred = r.get('model_answer') or ''
    refs = r.get('golden_answers', [])
    ref  = refs[0] if refs else ''
    if ref and pred:
        ratio = len(pred.split()) / max(len(ref.split()), 1)
        length_ratios.append(min(ratio, 2.0))   # cap at 2x
completeness = sum(length_ratios) / len(length_ratios) if length_ratios else 0.0

# ── Print ─────────────────────────────────────────────────────
print(f"\nRAGChecker-style Faithfulness Metrics")
print(f"{'─'*45}")
print(f"Faithfulness:        {faithfulness:.3f}  (no contradictions / samples with checker)")
print(f"Claim precision:     {claim_precision:.3f}  (supported claims / total claims)")
print(f"Grounded rate:       {grounded_rate:.3f}  (searched + checker support)")
print(f"Retrieval util:      {retrieval_util:.3f}  (searched + checked / all samples)")
print(f"Answer completeness: {completeness:.3f}  (answer len / reference len, capped 2x)")
print(f"Samples w/ checker:  {total_with_checker} / {len(results)}")

# ── Save ──────────────────────────────────────────────────────
m['rag_faithfulness']    = round(faithfulness,    4)
m['rag_claim_precision'] = round(claim_precision, 4)
m['rag_grounded_rate']   = round(grounded_rate,   4)
m['rag_retrieval_util']  = round(retrieval_util,  4)
m['rag_completeness']    = round(completeness,    4)

with open("$EVAL_OUT", 'w') as f:
    json.dump(d, f, indent=2)

# ── Final summary ─────────────────────────────────────────────
f1       = m.get('f1_mean', 0)
rl       = m.get('rouge_l', 0)
bs       = m.get('bert_score', 0)
tag_rate = m.get('has_answer_tag_rate', 0)
sup_rate = m.get('avg_checker_support_rate', 0)
avg_s    = m.get('avg_searches', 0)

print("\n============================================================")
print("FINAL SUMMARY")
print("============================================================")
print(f"Experiment:           $EXPERIMENT (step $STEP)")
print(f"Dataset:              $(basename $TEST_FILE)")
print(f"N samples:            {len(results)}")
print(f"---  Answer Quality ---")
print(f"Token F1:             {f1:.4f}   {'✓' if f1 > 0.19 else '✗'}")
print(f"ROUGE-L:              {rl:.4f}")
print(f"BERTScore:            {bs:.4f}   {'✓' if bs > 0.58 else '✗'}")
print(f"--- Faithfulness (RAGChecker) ---")
print(f"Faithfulness:         {faithfulness:.3f}   {'✓' if faithfulness > 0.7 else '✗'}")
print(f"Claim precision:      {claim_precision:.3f}   {'✓' if claim_precision > 0.5 else '✗'}")
print(f"Grounded rate:        {grounded_rate:.3f}")
print(f"--- Behavior ---")
print(f"Answer tag rate:      {tag_rate:.1%}   {'✓' if tag_rate > 0.90 else '✗'}")
print(f"Support rate:         {sup_rate:.1%}   {'✓ signal' if sup_rate > 0.4 else '✗ collapse'}")
print(f"Avg searches:         {avg_s:.3f}")
print(f"\nAll metrics saved to: $EVAL_OUT")
PYEOF
