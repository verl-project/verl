# TMEM pre-RL LoCoMo reproduction

This example implements the pre-RL TMEM evaluation from Table 1 of
“Scaling Self-Evolving Agents via Parametric Memory” (arXiv:2606.04536).
It uses the paper's rank-6 LoRA adapters on the final four FFN blocks,
initializes A with `Sigma_r V_r^T`, freezes A, trains B with online SGD, and
resets B between evaluation questions.

The trainer and rollout use separate GPUs. The default Transformers mode uses
a second PEFT replica and copies only LoRA A/B tensors, matching verl's native
adapter synchronization invariant. The optional `--rollout-backend sglang`
mode transfers only the 12 merged FFN matrices changed by LoRA after each
update, matching verl's required `merge=True` SGLang path. The
`--rollout-backend dflash` mode uses SGLang's batched DFlash verifier and
dynamically loads one LoRA adapter per question. The DFlash draft stays fixed;
only verifier adapters are synchronized after online SFT.

The evaluator follows the official LoCoMo conversation and answer templates,
including the temporal suffix for category 2 and randomized
`No information available` / adversarial-answer choices for category 5.
Independent question episodes use separately named adapters so rollout
generation can be batched without sharing fast-weight state between questions.

## Hyperparameter boundary

The Table 1 runner fails closed if any paper-specified TMEM setting or official
LoCoMo final-answer decoding setting differs:

| Setting | Locked value |
| --- | --- |
| LoRA | rank 6; `gate_proj`, `up_proj`, `down_proj`; final 4 layers |
| Initialization/update | `A=Sigma_r V_r^T` frozen; `B=0` then train B only |
| Online SFT | plain SGD, learning rate `5e-4`, 5 epochs, batch size 16 |
| Trigger dynamics | cumulative B within an episode; working context cleared |
| LoCoMo context budget | 4096 tokens |
| Final-answer decoding | official LoCoMo runner: 50 tokens, temperature 0.4, top-p 0.9, top-k 10 |

Question sharding, generation batch size, SGLang memory reservation, and the
SFT *episode microbatch* are execution settings, not TMEM training
hyperparameters. The episode microbatch defaults to 1. DFlash block size is
read from the draft checkpoint and validated rather than chosen by the runner.

## Data

Download the official LoCoMo-10 file:

```bash
mkdir -p data
wget https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json \
  -O data/locomo10.json
```

## Run

Activate a Conda environment containing verl, PyTorch, Transformers, and PEFT,
then run:

```bash
CUDA_VISIBLE_DEVICES=4,5 python -m examples.tmem.run_locomo \
  --model /path/to/Qwen3-4B \
  --data data/locomo10.json \
  --output-dir outputs/tmem_locomo_qwen3_4b \
  --trainer-device cuda:0 \
  --rollout-device cuda:1 \
  --generation-batch-size 20 \
  --max-extraction-tokens 4096 \
  --extraction-retries 2 \
  --seeds 1 2 3
```

The default script maps physical GPUs 4 and 5 to `cuda:0` and `cuda:1`.
Per-seed predictions, generated supervision, templates, configuration, and a
three-run summary are written under `outputs/tmem_locomo_qwen3_4b`.
Interrupted runs can continue with `--resume`; every completed question is
checkpointed to JSONL immediately.

### DFlash rollout

DFlash requires an SGLang build with DFlash, verifier-only multi-LoRA support,
and request-seeded verifier sampling. In particular, the target worker must
keep LoRA enabled while its cloned draft-worker arguments set
`enable_lora=False`, and speculative acceptance/final-sampling draws must be
keyed by each request's sampling seed and absolute token position. The runner
fails closed when that seeded-verifier capability is absent. Run it with a
compatible DFlash draft checkpoint:

```bash
CUDA_VISIBLE_DEVICES=4,5 python -m examples.tmem.run_locomo \
  --model /path/to/Qwen3-4B \
  --dflash-draft-model /path/to/Qwen3-4B-DFlash-b16 \
  --rollout-backend dflash \
  --data data/locomo10.json \
  --output-dir outputs/tmem_locomo_qwen3_4b_dflash \
  --trainer-device cuda:0 \
  --rollout-device cuda:1 \
  --generation-batch-size 20 \
  --max-extraction-tokens 4096 \
  --extraction-retries 2 \
  --seeds 1 2 3
```

To evaluate one seed with both training and DFlash rollout active on both
physical GPUs 4 and 5, use the resumable two-shard launcher:

```bash
bash scripts/tmem/run_locomo_dflash_2gpu.sh \
  1 \
  /tmp/locomo10.json \
  outputs/tmem_locomo_qwen3_4b_dflash_seed_1
```

It partitions the ten conversations into balanced 999- and 987-question
shards. Each GPU hosts its own trainer and rollout engine, and the launcher
merges both shards into one 1,986-question seed result. It defaults to the
sibling `Draft-OPD` checkout and its `.conda/draft-opd` environment. Paths and
resources can be overridden with
`CONDA_ENV`, `MODEL_PATH`, `DRAFT_MODEL_PATH`, `GPU_IDS`,
`SGLANG_MEM_FRACTION`, `GENERATION_BATCH_SIZE`, `MAX_EXTRACTION_TOKENS`,
`EXTRACTION_RETRIES`, `EXTRACTION_FAILURE_POLICY`, and `TMEM_CUDA_HOME`. Set
`MAX_QUESTIONS=1` for a one-question-per-GPU runtime smoke test.

The DFlash verifier uses SGLang's rejection sampler for non-greedy decoding,
including the extraction and answer top-k/top-p settings. Adapter tensors are
loaded directly from PEFT after every update; base weights and the fixed draft
are not retransferred. The DFlash block size is inferred from the draft
checkpoint (`16` for the validated Draft-OPD model); DFlash itself fixes its
non-applicable EAGLE step/top-k bookkeeping fields to `1`.

For a smoke test:

```bash
CUDA_VISIBLE_DEVICES=4,5 python -m examples.tmem.run_locomo \
  --model /path/to/Qwen3-4B \
  --data data/locomo10.json \
  --output-dir outputs/tmem_smoke \
  --max-questions 2 --seeds 1
```

## Reproduction boundary

The paper specifies LoRA rank and targets, SVD initialization, frozen A,
online-SFT optimizer settings, context budget, extraction prompt, dataset, and
metrics. It does not publish its model revision, random seeds, extraction
decoding limit or sampling parameters, Qwen3-specific final-answer decoding
parameters, LoRA alpha, or gradient clipping. This example uses the paper's
Appendix system template and Figure 3 memory-writing prompt as separate
system/user messages, `lora_alpha=rank` so PEFT implements the paper's unscaled
`BA`, and plain SGD without gradient clipping. Final answers follow the
[official LoCoMo Hugging Face benchmark runner][locomo-hf-runner]
(`max_new_tokens=50` for one question, `temperature=0.4`, `top_p=0.9`,
`top_k=10`); these values come from that benchmark protocol, not from the TMEM
paper. Extraction remains a separate reproduction choice: Qwen3's standard
non-thinking sampling (`temperature=0.7`, `top_p=0.8`, `top_k=20`), a
prefilled `[` on SGLang backends, a 4096-token operational safety cap, and at
most two retries. SGLang and DFlash requests additionally require a
`</qa_pairs>` sentinel after the outer JSON array and stop on that unambiguous
string; this avoids treating `]` inside a JSON string as an endpoint. Exact QA
duplicates, ignoring case and repeated whitespace, are removed before SFT and
before extending episode history. Entries without an actual question mark, or
whose normalized instruction and output are identical, are rejected as
non-QA supervision. The runner records every extraction attempt,
sentinel-independent parse result, schema drop, and duplicate count instead of
silently losing malformed or truncated JSON. The long-run default records
`extraction_failed=true` and applies an explicit empty update after retries;
`--extraction-failure-policy error` is available for debugging. These are
operational safeguards rather than paper-reported hyperparameters. Every
operational choice and generated record is stored so differences from the
reported 25.72 F1 / 15.40 EM can be audited instead of hidden.

[locomo-hf-runner]: https://github.com/snap-research/locomo/blob/3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376/task_eval/hf_llm_utils.py#L87-L166
