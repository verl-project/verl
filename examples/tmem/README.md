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
  --max-extraction-tokens 1024 \
  --seeds 1 2 3
```

The default script maps physical GPUs 4 and 5 to `cuda:0` and `cuda:1`.
Per-seed predictions, generated supervision, templates, configuration, and a
three-run summary are written under `outputs/tmem_locomo_qwen3_4b`.
Interrupted runs can continue with `--resume`; every completed question is
checkpointed to JSONL immediately.

### DFlash rollout

DFlash requires an SGLang build with DFlash and verifier-only multi-LoRA
support. In particular, the target worker must keep LoRA enabled while its
cloned draft-worker arguments set `enable_lora=False`. Run it with a compatible
DFlash draft checkpoint:

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
  --max-extraction-tokens 1024 \
  --seeds 1 2 3
```

To evaluate one seed with training and DFlash rollout isolated on physical
GPUs 4 and 5, respectively, use the resumable two-GPU launcher:

```bash
bash scripts/tmem/run_locomo_dflash_2gpu.sh \
  1 \
  /tmp/locomo10.json \
  outputs/tmem_locomo_qwen3_4b_dflash_seed_1
```

The launcher defaults to the sibling `Draft-OPD` checkout and its
`.conda/draft-opd` environment. Paths and resources can be overridden with
`CONDA_ENV`, `MODEL_PATH`, `DRAFT_MODEL_PATH`, `GPU_IDS`,
`SGLANG_MEM_FRACTION`, and `GENERATION_BATCH_SIZE`.

The DFlash verifier uses SGLang's rejection sampler for non-greedy decoding,
including the extraction and answer top-k/top-p settings. Adapter tensors are
loaded directly from PEFT after every update; base weights and the fixed draft
are not retransferred.

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
decoding limit or sampling parameters, final-answer decoding parameters, LoRA
alpha, or gradient clipping. This example uses the paper's Appendix system
template and Figure 3 memory-writing prompt as separate system/user messages,
the official LoCoMo answer protocol, `lora_alpha=rank` so PEFT implements the
paper's unscaled `BA`, Qwen3's standard non-thinking extraction sampling with
a 1024-token cap (the only response budget published, in the RL setup), and
plain SGD without gradient clipping. Every operational
choice and generated record is stored so differences from the reported 25.72
F1 / 15.40 EM can be audited instead of hidden.
