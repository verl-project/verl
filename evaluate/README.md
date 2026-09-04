# MedRAGChecker-verl: Checker-Guided Medical RAG with RL Training

A multi-turn medical RAG agent integrating NLI-based claim verification into Search-R1-style reinforcement learning. This codebase documents two undocumented failure modes in retrieval-augmented RL: **checker signal collapse** and a **reward hacking cascade**.

---

## Project Structure

```
verl/
├── examples/sglang_multiturn/
│   ├── config/
│   │   ├── search_multiturn_grpo_explicitcheck.yaml   # Qwen training config
│   │   ├── search_multiturn_grpo_explicitcheck_llama.yaml  # Llama training config
│   │   └── tool_config/
│   │       └── medical_search_checker_tool_config.yaml
│   └── search_r1_like/
│       └── run_search_checker_ablation_2gpu.sh        # Main training script
├── evaluate/
│   ├── eval.sh                                        # Merge + eval + metrics
│   ├── eval_extra_datasets.sh                         # Batch eval on extra datasets
│   └── evaluate_search_r1.py                          # Core eval script
├── search_r1_preprocess/
│   ├── checker_medrag_gpt4omini.py                    # GPT-4o-mini checker server
│   └── prepare_extra_eval_datasets.py                 # Download + format extra datasets
├── compute_bertscore_all.py                           # Batch BERTScore for all eval JSONs
├── verl/utils/reward_score/
│   ├── med_rag_checker.py                             # Main reward function
│   └── checkers/correctness.py                        # Format penalty logic
└── verl/experimental/agent_loop/
    └── tool_parser.py                                  # Tool call parsing (Qwen + Llama)
```

---

## Setup

```bash
conda activate sglang_srv
cd /ocean/projects/med230010p/yji3/BrowseCamp/verl
export HF_HUB_OFFLINE=1
export HF_HOME=/ocean/projects/med230010p/yji3/.cache/huggingface
```

---

## Training

The training script supports Qwen2.5-7B (default) and Llama-3.1-8B. Model family is auto-detected from `MODEL_PATH`.

### Qwen2.5-7B (default)

```bash
# Student checker — demonstrates signal collapse
CUDA_VISIBLE_DEVICES=2,3 \
bash examples/sglang_multiturn/search_r1_like/run_search_checker_ablation_2gpu.sh \
    checker_guarded \
    "trainer.total_training_steps=200"

# GPT-4o-mini checker — resolves collapse
MEDRAG_SEARCH_BONUS=1 \
CUDA_VISIBLE_DEVICES=2,3 \
bash examples/sglang_multiturn/search_r1_like/run_search_checker_ablation_2gpu.sh \
    checker_guarded \
    "trainer.total_training_steps=200"

# Triage + GPT + English constraint — best configuration
CUDA_VISIBLE_DEVICES=2,3 \
bash examples/sglang_multiturn/search_r1_like/run_search_checker_ablation_2gpu.sh \
    triage_guarded \
    "trainer.total_training_steps=200"
```

### Llama-3.1-8B (cross-model validation)

```bash
LLAMA=/ocean/projects/med230010p/yji3/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659

MODEL_PATH=$LLAMA \
MEDRAG_SEARCH_BONUS=0 \
CUDA_VISIBLE_DEVICES=2,3 \
bash examples/sglang_multiturn/search_r1_like/run_search_checker_ablation_2gpu.sh \
    checker_guarded \
    "trainer.total_training_steps=200" \
    "actor_rollout_ref.actor.fsdp_config.param_offload=True" \
    "actor_rollout_ref.ref.fsdp_config.param_offload=True" \
    "actor_rollout_ref.rollout.gpu_memory_utilization=0.30"
```

### Training Modes

| Mode | Description | Key Reward Components |
|---|---|---|
| `search_only` | Search-R1 baseline | F1 only |
| `checker_guarded` | NLI verification + auto-check | F1 + checker + format penalty |
| `triage_guarded` | Difficulty-based budgets | F1 + checker + triage |

---

## Evaluation

### Single model eval

```bash
# Medical QA (default test set, 87 samples)
SKIP_MERGE=1 bash evaluate/eval.sh <experiment_name> <step>

# Extra datasets
SKIP_MERGE=1 bash evaluate/eval.sh <experiment_name> <step> \
    searchr1_data/extra_eval/medicationqa_test.parquet _medicationqa

SKIP_MERGE=1 bash evaluate/eval.sh <experiment_name> <step> \
    searchr1_data/extra_eval/bioasq_test.parquet _bioasq
```

### Batch eval on all extra datasets

```bash
bash evaluate/eval_extra_datasets.sh
```

### Compute BERTScore for all existing eval JSONs

```bash
export CUDA_VISIBLE_DEVICES=""
python compute_bertscore_all.py
```

---

## Metrics

Each eval run computes:

| Metric | Description |
|---|---|
| Token F1 | Token-level overlap with reference answer |
| ROUGE-L | Longest common subsequence overlap |
| BERTScore | Semantic similarity (deberta-xlarge-mnli) |
| `avg_searches` | Average search tool calls per sample |
| `avg_checks` | Average checker tool calls per sample |
| `has_answer_tag_rate` | Fraction of samples with `<answer>` tag |
| `avg_checker_support_rate` | Fraction of claims labelled *entail* |
| `rag_faithfulness` | Fraction of samples with zero contradictions |
| `rag_claim_precision` | Supported claims / total verified claims |
| `rag_grounded_rate` | Searched + got checker support |
| `rag_retrieval_util` | Searched + checked / all samples |

---

## Extra Evaluation Datasets

Download and format extra eval datasets (requires internet access on login node):

```bash
export HF_HUB_OFFLINE=0
python search_r1_preprocess/prepare_extra_eval_datasets.py
```

This creates:

| File | Source | Samples | Description |
|---|---|---|---|
| `medicationqa_test.parquet` | truehealth/medicationqa | 200 | Consumer medication questions |
| `bioasq_test.parquet` | rag-datasets/rag-mini-bioasq | 200 | Biomedical QA |
| `medquad_full_test.parquet` | keivalya/MedQuad | 200 | Medical Q&A pairs |

---

## Key Findings

### 1. Checker Signal Collapse

When a student NLI model (Meditron-3-8B) serves as verifier, it assigns *neutral* to 97%+ of all claims regardless of evidence quality, providing zero process reward.

**Root causes (diagnosed in order):**

1. Evidence truncation at 128 tokens → fix: extend to 768 tokens (support: 0% → 35.7%)
2. Non-atomic claim extraction via sentence splitting → fix: GPT-4o-mini extraction (35.7% → 37.5%)
3. Structural log-probability bias in student model → fix: replace with GPT-4o-mini verifier (37.5% → 77.5%)

### 2. Reward Hacking Cascade

After fixing the verifier, the model discovers three successive shortcuts:

| Stage | Shortcut | Fix |
|---|---|---|
| GPT checker active | Ultra-short answers (130 chars) | Format penalty: -0.3 if < 50 chars |
| + Format penalty | Search avoidance (98% zero-search) | Search bonus: +0.1 if ≥ 1 search |
| + Triage | Chinese outputs (64.4%) | English constraint in system prompt |

### 3. Cross-Model Validation

Signal collapse is model-agnostic. Both Qwen2.5-7B and Llama-3.1-8B exhibit support_rate=0% with student checker.

---

## Main Results (Medical QA, n=87)

| Config | F1 | ROUGE-L | BERT | Support% | Tag% |
|---|---|---|---|---|---|
| Zero-shot | 0.191 | 0.121 | 0.538 | — | 100% |
| Search-only RL | 0.190 | 0.144 | 0.537 | — | 58.6% |
| + Student checker | 0.210 | 0.161 | 0.595 | **0%** | 98.9% |
| + Evidence fix | 0.190 | 0.153 | 0.589 | 35.7% | 93.1% |
| + GPT verifier | 0.160 | 0.135 | 0.553 | 77.5% | 82.8% |
| + Format penalty | 0.194 | 0.154 | 0.589 | 75.0% | 98.9% |
| **Triage+GPT+EN** | **0.212** | 0.150 | — | **87.2%** | **100%** |

---

## Citation

```bibtex
@article{medragchecker2026,
  title={Checker-Guided Medical RAG: From Signal Collapse to Reward Hacking},
  author={Anonymous},
  journal={ARR},
  year={2026}
}
```
