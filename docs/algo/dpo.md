# Direct Preference Optimization (DPO)

Last updated: 11/24/2026.

## Background

Direct Preference Optimization (DPO) [1] fine-tunes a model on preference pairs: for each prompt $x$ there is a
chosen (preferred) response $y_w$ and a rejected response $y_l$. It needs no reward model and no rollouts. The
policy $\pi_\theta$ is trained against a frozen reference model $\pi_\text{ref}$, usually the starting checkpoint:

$$
\mathcal{L}_\text{DPO}(\theta) = -\mathbb{E}_{(x, y_w, y_l)} \log \sigma\Big(\beta \Big[
\log \frac{\pi_\theta(y_w \mid x)}{\pi_\text{ref}(y_w \mid x)} -
\log \frac{\pi_\theta(y_l \mid x)}{\pi_\text{ref}(y_l \mid x)} \Big]\Big)
$$

$\log \pi(y \mid x)$ is the sum of the response tokens' log-probabilities. $\beta$ controls how far the policy
may move from the reference. At the first step the policy equals the reference, so the loss is $\ln 2 \approx 0.693$.

`verl/trainer/dpo_trainer.py` runs offline DPO with one process per GPU (SPMD), like `verl/trainer/sft_trainer.py`.

## Data

Each row of the parquet or JSONL file has a `prompt`, a `chosen` and a `rejected` string (column names are set by
`data.prompt_key`, `data.chosen_key` and `data.rejected_key`). Set `data.chat_template_key` to apply the
tokenizer's chat template. `data.max_length` applies to the chosen and the rejected sequence separately, and
`data.train_batch_size` counts pairs.

`examples/data_preprocess/gsm8k_dpo.py` builds a small dataset for smoke tests: the chosen response is the GSM8K
reference solution and the rejected response has a wrong final answer.

## Loss variants

Set `trainer.loss_type` to one of:

| `loss_type` | Loss |
|---|---|
| `sigmoid` | Standard DPO [1], with optional `label_smoothing` |
| `robust` | Robust DPO for noisy labels [2] |
| `ipo` | IPO [3], a squared loss on the log-ratio margin |
| `hinge` | Hinge loss on the margin (SLiC [4]) |
| `exo_pair` | EXO [5] |
| `aot_pair`, `aot` | Aligned optimal transport [6] |

New variants can be added with `verl.trainer.dpo.dpo_losses.register_dpo_loss`.

## Reference model

- **LoRA** (`model.lora_rank > 0`): the reference is the policy with its adapter disabled. No extra memory.
- **Full fine-tuning**: a second, forward-only copy of the model.
- **`trainer.reference_free=True`**: no reference model; reference log-probabilities are taken as 0.

## Run

```bash
python3 examples/data_preprocess/gsm8k_dpo.py --local_save_dir ~/data/gsm8k_dpo
torchrun --standalone --nproc-per-node=8 -m verl.trainer.dpo_trainer \
    data.train_files=$HOME/data/gsm8k_dpo/train.parquet \
    data.val_files=$HOME/data/gsm8k_dpo/test.parquet \
    data.chat_template_key=default \
    model.path=$HOME/models/Qwen/Qwen3-0.6B \
    trainer.loss_type=sigmoid trainer.beta=0.1 \
    trainer.n_gpus_per_node=8
```

## Logged metrics

`train/dpo/loss`, `train/dpo/accuracy` (share of pairs whose chosen reward is higher), `train/dpo/chosen_reward`,
`train/dpo/rejected_reward`, `train/dpo/reward_margin`, `train/dpo/chosen_logps` and `train/dpo/rejected_logps`,
with the same names under `val/` for validation. The reward of a response is
$\beta(\log\pi_\theta(y \mid x) - \log\pi_\text{ref}(y \mid x))$.

## References

1. Rafailov et al. Direct Preference Optimization: Your Language Model is Secretly a Reward Model. https://arxiv.org/abs/2305.18290
2. Chowdhury et al. Provably Robust DPO. https://arxiv.org/abs/2403.00409
3. Azar et al. A General Theoretical Paradigm to Understand Learning from Human Preferences. https://arxiv.org/abs/2310.12036
4. Zhao et al. SLiC-HF: Sequence Likelihood Calibration with Human Feedback. https://arxiv.org/abs/2305.10425
5. Ji et al. Towards Efficient Exact Optimization of Language Model Alignment. https://arxiv.org/abs/2402.00856
6. Melnyk et al. Distributional Preference Alignment of LLMs via Optimal Transport. https://arxiv.org/abs/2406.05882
