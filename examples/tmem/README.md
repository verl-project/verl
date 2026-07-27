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
update, matching verl's required `merge=True` SGLang path.

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
  --seeds 1 2 3
```

The default script maps physical GPUs 4 and 5 to `cuda:0` and `cuda:1`.
Per-seed predictions, generated supervision, templates, configuration, and a
three-run summary are written under `outputs/tmem_locomo_qwen3_4b`.
Interrupted runs can continue with `--resume`; every completed question is
checkpointed to JSONL immediately.

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
metrics. It does not publish its final-answer template or generation sampling
parameters, and it does not state exactly how the Appendix system template and
the memory-writing requirements are combined. This example preserves both in
one extraction prompt and stores all operational choices with results so
differences from the reported 25.72 F1 / 15.40 EM can be audited. The runner
also exposes and records a 1.0 gradient-norm clip: the paper does not state
this value, but it is verl's actor default and prevents the stated SVD scale
and SGD learning rate from destabilizing online updates.
