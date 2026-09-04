export HF_HOME=/ocean/projects/med230010p/yji3/.cache/huggingface
export TRANSFORMERS_CACHE=/ocean/projects/med230010p/yji3/.cache/huggingface/transformers
export HF_HUB_CACHE=/ocean/projects/med230010p/yji3/.cache/huggingface/hub
export CUDA_VISIBLE_DEVICES=""   # ← 强制用 CPU，避免 OOM

python3 << 'EOF'
import json, os
os.environ["CUDA_VISIBLE_DEVICES"] = ""   # 双重保险

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
    'Format penalty only':         'eval_qwen2.5-7b-no_checker-guarded-ablation-27-14-24-step200.json',
    'Format+GPT checker (NEW)':    'eval_qwen2.5-7b-checker-guarded-ablation-27-16-40-step200.json',
}

print(f"{'Config':<35} {'Token F1':>8} {'ROUGE-L':>8} {'BERTScore':>10}")
print("="*65)

results_cache = {}
all_names = []

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
    preds = [r.get('model_answer') or '' for r in results]
    refs  = [r['golden_answers'][0] if r.get('golden_answers') else '' for r in results]
    rl = sum(rouge.score(ref, pred)['rougeL'].fmeasure
             for pred, ref in zip(preds, refs)) / len(preds)
    results_cache[label] = (token_f1, rl, preds, refs)
    print(f"{label:<35} {token_f1:>8.4f} {rl:>8.4f} {'pending':>10}")
    all_names.append(label)

print("\nComputing BERTScore on CPU (takes ~5 min)...")
for label in all_names:
    token_f1, rl, preds, refs = results_cache[label]
    _, _, F = bscore(
        preds, refs,
        lang='en',
        model_type='microsoft/deberta-xlarge-mnli',
        device='cpu',      # ← 强制 CPU
        batch_size=8,      # ← 小 batch 避免内存爆
        verbose=False
    )
    bs = F.mean().item()
    marker = " ←" if 'NEW' in label else ""
    print(f"{label:<35} {token_f1:>8.4f} {rl:>8.4f} {bs:>10.4f}{marker}")
EOF

结果 
Computing BERTScore on CPU (takes ~5 min)...
Baseline                              0.1909   0.1205     0.5375
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
Format penalty only                   0.1741   0.1309     0.5041
Format+GPT checker (NEW)              0.1936   0.1544     0.5888 ←
(sglang_srv) [yji3@w001 verl]$ 

