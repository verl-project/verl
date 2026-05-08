# What Makes a Medical Verifier Trainable?
### Diagnosing Signal Collapse and Reward Hacking in Checker-Guided RAG for Biomedical QA

> **EMNLP 2025 (under review)** | [Anonymous Code](https://anonymous.4open.science/r/medchecker-verl-7FC1/)

---

## TL;DR

We study what happens when you plug an NLI verifier into the RL reward path of a medical RAG agent. Three things can go wrong — and we show how to avoid them:

| Failure | Cause | Fix |
|---|---|---|
| **Signal collapse** | LLM log-prob scoring labels >97% of claims Neutral → zero RL gradient | Use a calibrated classifier (MedNLI-Cls) |
| **Reward hacking cascade** | Strong verifier (86% support) triggers ultra-short answers → search avoidance → language collapse | Prefer moderate-signal verifier (54% support) |
| **Policy-dependent signal** | Same checker = 54% support on Qwen2.5-7B but 85% on Qwen3-4B | Measure support rate on your policy, not in isolation |

**Bottom line**: a calibrated local MedNLI classifier (no GPT dependency) achieves the highest answer quality across four held-out benchmarks (BERTScore **0.600**, +12% over zero-shot).

---

## Overview

We train a multi-turn medical RAG agent on Qwen2.5-7B-Instruct with GRPO. The agent interleaves `<search>` (dense retrieval over MedRAG) and `<check>` (NLI claim verification) calls before emitting a final `<answer>`. We compare **four NLI verification back-ends** as process rewards spanning a 2×2 ablation over (extractor × scorer):

```
                    Scorer
                ┌──────────┬──────────┐
                │ log-prob │ classify │
Extractor ──────┼──────────┼──────────┤
sent-split      │ Likeli-  │ MedNLI-  │
                │ hood-NLI │ Cls      │
GPT-atomic      │ Hybrid   │ GPT-NLI  │
                └──────────┴──────────┘
```

**Key finding**: signal collapse tracks the *scorer* column (log-prob), not the extractor column.

---

## Results

### Main results (Qwen2.5-7B, 4 held-out benchmarks, n=1,479)

| Checker | BERTScore | F1 | Support% | Faithfulness | GPT needed? |
|---|---|---|---|---|---|
| Zero-shot baseline | 0.538 | — | — | — | No |
| Likelihood-NLI | 0.599 | 0.197 | 0%* | 0.964 | No |
| **MedNLI-Cls** | **0.600** | **0.215** | 54.0% | 0.972 | **No** |
| Hybrid | 0.565 | 0.186 | 75.8% | **0.987** | Partial |
| GPT-NLI | 0.591 | 0.184 | **86.1%** | 0.972 | Full |

\* Self-evaluation gives 0% due to collapse; GPT external evaluation of the same outputs gives 80.1%.

### Reward hacking cascade (GPT-NLI, unguarded → guarded)

| Stage | Shortcut | Countermeasure |
|---|---|---|
| GPT-NLI checker | Ultra-short answers (130 chars) | Format penalty |
| + Format penalty | Search avoidance (98% zero-search) | Search bonus |
| + Search bonus | Short answers again (141 chars) | Triage budget |
| + Triage | Language collapse (64.4% Chinese) | English-only prompt |

---

## Installation

```bash
git clone https://github.com/JoyDajunSpaceCraft/medchecker-verl.git
cd medchecker-verl
pip install -r requirements.txt
```

**Requirements**: Python 3.9+, PyTorch 2.0+, `transformers`, `peft`, `vllm` (for fast inference), `faiss-gpu`, `openai` (for GPT-NLI / Hybrid only)

**Hardware**: 2× H100 or A100 80GB for GRPO training (~2 h per run).

---

## Repository Structure

```
medchecker-verl/
├── agents/
│   ├── rollout.py          # Multi-turn search-check-answer loop (Alg. 1)
│   └── triage.py           # Easy/medium/hard budget controller
├── verifiers/
│   ├── likelihood_nli.py   # Meditron-3-8B log-prob scoring
│   ├── mednli_cls.py       # PubMedBERT MedNLI classifier (recommended)
│   ├── hybrid.py           # GPT extraction + local log-prob scoring
│   └── gpt_nli.py          # GPT-4o-mini extraction + verification
├── retriever/
│   └── medcpt_server.py    # MedCPT bi-encoder over MedRAG corpus
├── rewards/
│   └── reward_fn.py        # r_base × (1 + α·φ_check) + P_fmt
├── training/
│   └── grpo_train.py       # GRPO training loop (Qwen2.5-7B primary)
├── eval/
│   └── evaluate.py         # BERTScore, F1, ROUGE-L, Support%, Faith
├── data/                   # Dataset loaders
└── scripts/
    ├── train_mednli_cls.sh # Fine-tune PubMedBERT on MedNLI
    └── run_eval.sh         # Held-out evaluation
```

---

## Quick Start

### Step 1 — Train the MedNLI-Cls verifier (no GPT needed)

```bash
bash scripts/train_mednli_cls.sh
# Fine-tunes PubMedBERT-base on MedNLI (~11K pairs)
# Reaches 84.1% test accuracy (Entail/Neutral/Contradict)
# Saves to checkpoints/mednli_cls/
```

### Step 2 — Start the retrieval server

```bash
python retriever/medcpt_server.py \
    --corpus_dir /path/to/medrag_corpus \
    --port 8080
# MedCPT bi-encoder over PubMed + StatPearls + textbooks
# Serves top-5 passages per query, <80ms latency
```

### Step 3 — Run GRPO training

```bash
python training/grpo_train.py \
    --base_model Qwen/Qwen2.5-7B-Instruct \
    --verifier mednli_cls \
    --verifier_ckpt checkpoints/mednli_cls/ \
    --retriever_url http://localhost:8080 \
    --train_data data/train_combined.jsonl \
    --output_dir checkpoints/agent_mednli/ \
    --alpha 1.0 \
    --n_rollouts 4 \
    --max_steps 200
```

To use GPT-NLI instead (requires API key):

```bash
export OPENAI_API_KEY=sk-...
python training/grpo_train.py \
    --verifier gpt_nli \
    --format_penalty -0.3 \   # Required to prevent cascade Stage 1
    --search_bonus 0.1 \      # Required to prevent cascade Stage 2
    --english_only \          # Required to prevent cascade Stage 3
    ...
```

### Step 4 — Evaluate

```bash
python eval/evaluate.py \
    --model_ckpt checkpoints/agent_mednli/ \
    --datasets medicationqa biasq medquad \
    --retriever_url http://localhost:8080 \
    --verifier mednli_cls \
    --verifier_ckpt checkpoints/mednli_cls/ \
    --output results/mednli_eval.json
```

---

## Verification Back-Ends

### MedNLI-Cls (recommended — no GPT)

Fine-tunes `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` on MedNLI with a 3-way classification head. Inputs are formatted as `[CLS] evidence [SEP] claim [SEP]`.

- **Support rate**: 54.0% (moderate — avoids cascade)
- **Does not collapse**: non-degenerate on the same pairs that collapse Likelihood-NLI
- **Latency**: <50ms per rollout on A100

### Likelihood-NLI (collapse baseline)

Scores entailment via `Meditron-3-8B` log-probability comparison. Collapses to >97% Neutral on sentence-split claims against truncated evidence. Causes three root failures:

1. Evidence truncation (256-token limit → 1–2 sentences)
2. Non-atomic claim extraction (sentence splitting → compound claims)
3. Residual log-prob bias

Diagnostic chain recovery:

```
Baseline: 0% support
+ Evidence fix (256→768 tok): 35.7%
+ GPT claim extraction:       37.5%
+ GPT verification:           65.6–77.5%
MedNLI-Cls (no fixes):        54.0%  ← same corpus, no cascade
```

### Hybrid

GPT-4o-mini atomic-claim extraction + Likelihood-NLI log-prob scoring. Support rate 75.8% — does **not** collapse because cleaner single-proposition inputs mitigate log-prob bias.

### GPT-NLI

GPT-4o-mini for both extraction and verification. Highest support rate (86.1%) but triggers the reward hacking cascade without additional guardrails.

---

## Reward Function

```
R(τ) = r_base(τ) · (1 + α · φ_check(â, D)) + P_fmt(â)

r_base = 0.58·EM + 0.25·F1 + 0.17·FmtScore   (normalised weights)
φ_check = mean_k[ s_k · p_k ]                  ∈ [-1, +1]
  s_k = +1.0 if Entail, 0 if Neutral, -1.5 if Contradict
P_fmt = -0.5 (missing tag), -0.3 (<50 chars), 0 (otherwise)
```

Multiplicative coupling prevents high faithfulness from substituting for correctness. Neutral verdicts contribute zero — a collapsed verifier produces no checker gradient.

---

## Triage Controller

Assigns each question to easy / medium / hard before rollout:

| Tier | Search budget | Check budget | Turn budget |
|---|---|---|---|
| Easy | 1 | 1 | 3 |
| Medium | 2 | 2 | 5 |
| Hard | 4 | 3 | 7 |

Mid-rollout escalation triggers when: contradiction rate >0.30 or support rate <0.40 or search fails. Triage reduces search calls ~3× without quality loss but **cannot rescue verifier collapse**.

---

## Datasets

**Training** (combined, 1,513 samples):

| Dataset | Domain |
|---|---|
| [CSIRO MedRedQA](https://github.com/CSIRO-NLP/MedRedQA) | Consumer health (Reddit r/AskDocs) |
| [LiveQA-Med](https://github.com/abachaa/LiveQA_MedicalTask_TREC2017) | Consumer health (NLM) |
| [PubMedQA](https://pubmedqa.github.io/) | Biomedical research |

**Held-out evaluation** (1,479 samples total):

| Dataset | n | Domain |
|---|---|---|
| [MedicationQA](https://github.com/abachaa/MedicationQA) | 674 | Drug safety |
| [BioASQ-Y/N](http://bioasq.org/) | 618 | Biomedical literature (2019–2023) |
| [MedQuAD](https://github.com/abachaa/MedQuAD) | 100 | NIH consumer health |
| [MEDIQA](https://github.com/abachaa/MEDIQA-2021) | 85 | Clinical summarisation |

---

## Cross-Model Validation

| Base model | Checker | BERTScore | F1 | Support% |
|---|---|---|---|---|
| Qwen2.5-7B | MedNLI-Cls | **0.600** | **0.215** | 54.0% |
| Qwen2.5-7B | GPT-NLI | 0.591 | 0.184 | 86.1% |
| Qwen3-4B | MedNLI-Cls | 0.591 | 0.217 | 85.3%† |
| Qwen3-4B | Likelihood-NLI | 0.594 | 0.209 | 51.0% |
| Llama-3.1-8B | MedNLI-Cls | 0.521 | 0.163 | 49.4% |
| Llama-3.1-8B | GPT-NLI | 0.437 | 0.173 | 0%‡ |

† MedNLI-Cls registers as *strong* signal on Qwen3-4B (85.3% vs 54.0% on Qwen2.5-7B) — same checkpoint, same data — demonstrating that signal strength is a property of the **policy–verifier pair**, not the verifier alone.  
‡ Format degradation: only 50% valid answer tags on Llama under GPT-NLI.

---

## Key Takeaways for Practitioners

1. **Do not use LLM log-prob scoring as an NLI reward.** It collapses to Neutral regardless of evidence quality. Use a fine-tuned classification head instead.

2. **Measure the verifier's support rate on your policy's rollouts, not on a static NLI benchmark.** Accuracy on MedNLI test set does not predict reward-path behaviour.

3. **Moderate signal (40–60% support) outperforms strong signal (>80%) on answer quality** because strong signal triggers the three-stage reward hacking cascade.

4. **If you must use GPT-NLI**, apply all four countermeasures simultaneously: format penalty, search bonus, triage, English-only constraint. Adding them one at a time exposes a new shortcut each time.

5. **A calibrated local classifier is Pareto-optimal**: highest BERTScore (0.600), no API cost, no cascade, 100% format stability.

---

## Citation

```bibtex
@inproceedings{anonymous2025verifier,
  title     = {What Makes a Medical Verifier Trainable?
               Diagnosing Signal Collapse and Reward Hacking in
               Checker-Guided {RAG} for Biomedical {QA}},
  author    = {Anonymous},
  booktitle = {Proceedings of the 2025 Conference on Empirical Methods
               in Natural Language Processing},
  year      = {2025}
}
```

---

## License

Code: MIT. Model weights follow base model licenses (Qwen2.5, Qwen3, Llama 3.1). Dataset usage follows original dataset licenses. MedRAG corpus usage follows MedRAG terms.
