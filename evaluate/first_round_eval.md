export HF_HOME=/ocean/projects/med230010p/yji3/.cache/huggingface
export TRANSFORMERS_CACHE=/ocean/projects/med230010p/yji3/.cache/huggingface/transformers
export HF_HUB_CACHE=/ocean/projects/med230010p/yji3/.cache/huggingface/hub

python3 << 'EOF'
import json, glob, os
from rouge_score import rouge_scorer
from bert_score import score as bscore

rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

eval_files = {
    'Baseline':                    'eval_clean.json',
    'Search only (no triage)':     'eval_search_only_no_triage.json',
    'Search only (triage)':        'eval_search_only_triage.json',
    'Checker no triage (s93)':     'eval_checker_no_triage.json',
    'Checker triage fix (s93)':    'eval_checker_triage_after_fix.json',
    'No triage explicit v2':       'eval_no_triage_explicitcheck_after_fix_v2.json',
    'Triage explicit v2':          'eval_triage_explicitcheck_after_fix_v2.json',
    'Evidence fix':                'eval_no_triage_explicitcheck_after_evidence_fix.json',
    'Hybrid no triage (s188)':     'eval_hybrid_no_triage_explicit_step188.json',
    'Hybrid triage (s188)':        'eval_hybrid_triage_explicit_step188.json',
    'OpenAI checker (old s377)':   'eval_openai_checker_guarded_step377.json',
    'OpenAI checker (new s377)':   'eval_qwen2.5-7b-checker-guarded-ablation-26-19-48-step377.json',
}

print(f"{'Config':<35} {'Token F1':>8} {'ROUGE-L':>8} {'BERTScore':>10}")
print("="*65)

all_names, all_preds, all_refs = [], [], []
results_cache = {}

for label, fname in eval_files.items():
    if not os.path.exists(fname):
        print(f"{label:<35} FILE NOT FOUND")
        continue
    with open(fname) as f:
        d = json.load(f)
    results = d.get('results', [])
    if not results:
        continue

    token_f1 = d['metrics'].get('f1_mean', 0)
    preds = [r['model_answer'] or '' for r in results]
    refs  = [r['golden_answers'][0] if r['golden_answers'] else '' for r in results]

    # ROUGE-L
    rl = sum(rouge.score(ref, pred)['rougeL'].fmeasure
             for pred, ref in zip(preds, refs)) / len(preds)

    results_cache[label] = (token_f1, rl, preds, refs)
    print(f"{label:<35} {token_f1:>8.4f} {rl:>8.4f} {'computing...':>10}")
    all_names.append(label)
    all_preds.append(preds)
    all_refs.append(refs)

# BERTScore — compute all at once to save time
print("\nComputing BERTScore for all configs (this takes ~2 min)...")
for label in all_names:
    token_f1, rl, preds, refs = results_cache[label]
    _, _, F = bscore(preds, refs, lang='en',
                     model_type='microsoft/deberta-xlarge-mnli',
                     verbose=False)
    bs = F.mean().item()
    print(f"{label:<35} {token_f1:>8.4f} {rl:>8.4f} {bs:>10.4f}")

EOF
结果 Baseline                              0.1909   0.1205 computing...
Search only (no triage)               0.1903   0.1437 computing...
Search only (triage)                  0.1157   0.1235 computing...
Checker no triage (s93)               0.2100   0.1606 computing...
Checker triage fix (s93)              0.2193   0.1698 computing...
No triage explicit v2                 0.1980   0.1588 computing...
Triage explicit v2                    0.2052   0.1637 computing...
Evidence fix                          0.1897   0.1530 computing...
Hybrid no triage (s188)               0.1938   0.1560 computing...
Hybrid triage (s188)                  0.2042   0.1608 computing...
OpenAI checker (old s377)             0.1676   0.1387 computing...
OpenAI checker (new s377)             0.1596   0.1348 computing...
Computing BERTScore for all configs (this takes ~2 min)...
tokenizer_config.json: 100%|██████████████████████████████████████████████████████████████████| 52.0/52.0 [00:00<00:00, 677kB/s]
config.json: 100%|█████████████████████████████████████████████████████████████████████████████| 792/792 [00:00<00:00, 12.8MB/s]
vocab.json: 899kB [00:00, 13.7MB/s]
merges.txt: 456kB [00:00, 14.4MB/s]
pytorch_model.bin: 100%|████████████████████████████████████████████████████████████████████| 3.04G/3.04G [00:16<00:00, 188MB/s]
model.safetensors:  67%|█████████████████████████████████████████████▍                      | 2.03G/3.04G [00:04<00:01, 715MB/s]Baseline                              0.1909   0.1205     0.5375
model.safetensors: 100%|████████████████████████████████████████████████████████████████████| 3.04G/3.04G [00:05<00:00, 523MB/s]
Search only (no triage)               0.1903   0.1437     0.5366
Search only (triage)                  0.1157   0.1235     0.5073
Checker no triage (s93)               0.2100   0.1606     0.5946
Checker triage fix (s93)              0.2193   0.1698     0.5930
No triage explicit v2                 0.1980   0.1588     0.5886
Triage explicit v2                    0.2052   0.1637     0.6020
Evidence fix                          0.1897   0.1530     0.5891
Hybrid no triage (s188)               0.1938   0.1560     0.5913
Hybrid triage (s188)                  0.2042   0.1608     0.5999
OpenAI checker (old s377)             0.1676   0.1387     0.5314
OpenAI checker (new s377)             0.1596   0.1348     0.5534