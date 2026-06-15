# Cumulative Prefix-divergence Policy Optimization (CPPO)


<div align="center">

## Beyond Uniform Token-Level Trust Region in LLM Reinforcement Learning

[![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white )](https://arxiv.org/pdf/2606.10968)
[![Project Page](https://img.shields.io/badge/Project_Page-000000?style=for-the-badge&logo=githubpages&logoColor=white)](https://hunyuan-cppo.github.io)
[![Twitter](https://img.shields.io/badge/Twitter-%23000000.svg?style=for-the-badge&logo=twitter&logoColor=white)](https://x.com/NickZhou523786/status/2066106644736667838)

</div>


## ✨ Getting started

1. Prepare the datasets by running [prepare_dapo_data.sh](https://github.com/verl-project/verl-recipe/blob/3490a22a0a3adeb7e4787fe70b1060b642efbae4/dapo/prepare_dapo_data.sh):

```bash
bash prepare_dapo_data.sh # This downloads the datasets to ${HOME}/verl/data by default
```

2. Prepare the model:

```bash
hf download Qwen/Qwen3-30B-A3B-Base --local-dir ${HOME}/verl/models/Qwen3-30B-A3B-Base
```

3. Run the script (Qwen3-30B-A3B-Base, Binary-TV):

```bash
bash examples/cppo_trainer/run_qwen3_30b_a3b_megatron.sh
```

The script mirrors the DPPO example layout. Parallelism / batch follow the paper's
Qwen3-30B-A3B-Base run: actor TP=4 / EP=8 / PP=1 / ETP=1, rollout TP=2, 8 GPUs/node,
`train_batch_size=256`, `ppo_mini_batch_size=32` (8 gradient ministeps), `rollout.n=16`,
1k prompt / 16k response.

## 📖 Introduction

🚨 **Uniform token-level trust regions are not enough for LLM RL.** CPPO is a drop-in
mask that reallocates the divergence budget by **position** and **prefix drift**. It keeps
the DPPO ratio-advantage surrogate and the same per-token divergence — only the masking
decision changes (no new loss terms).

<div align="left">
  <img src="https://hunyuan-cppo.github.io/assets/readme/fig1.png" alt="CPPO vs PPO/DPPO masks" style="width: 96%; height: auto;">
</div>

**The surrogate and the masks.** PPO (and variants like GRPO) gate updates with a heuristic
based on the probability ratio `|ρ_t − 1| ≤ ε`. DPPO replaces this with a principled
divergence threshold `D_t ≤ δ`. CPPO goes further: it scales the threshold by a **position
weight** `w_t` and adds a **cumulative prefix-average budget** `δ_b`, so a token is kept only
when both `w_t·D_t ≤ δ` and the running prefix average of weighted divergence stays under
`δ_b` (or when the update moves `π` back toward the rollout policy `μ`).

🤔 **Why break "uniform thresholds"?** In autoregressive generation, early deviations
cascade — a small policy shift at token 10 alters the conditioning for the next 10k tokens.
Uniform constraints underestimate early-stage risk and over-restrict late-stage exploration.

<div align="left">
  <img src="https://hunyuan-cppo.github.io/assets/readme/fig2.png" alt="Position-weighted threshold" style="width: 96%; height: auto;">
</div>

CPPO ties the budget to position: early tokens (which condition the whole suffix) are
constrained more tightly, and the threshold relaxes toward the end of the response via the
decreasing weight `w_t ∈ [w_min, 1]`.

⏳ **Pointwise evaluation ignores cumulative prefix drift.** If the history has already
drifted far from the rollout policy, any further deviation at the current token carries high
compounding risk. CPPO dynamically tightens the limit as off-policy drift accumulates.

<div align="left">
  <img src="https://hunyuan-cppo.github.io/assets/readme/fig3.png" alt="Cumulative prefix budget" style="width: 96%; height: auto;">
</div>

The effective threshold `c_t = min(δ, δ + δ_b·W_{t-1} − S_{t-1})` shrinks once the weighted
prefix sum `S_{t-1}` over-spends the budget allotment `δ_b·W_{t-1}` — so a token can satisfy
the token-level threshold yet still be masked by the prefix constraint.

📉 **Theory.** Starting from a finite-horizon performance-difference identity, the paper
proves that early-token shifts carry a remaining-horizon penalty. By tracking prefix
deviations, CPPO provides a tighter, provably robust policy-improvement bound than uniform,
pointwise methods.

<div align="left">
  <img src="https://hunyuan-cppo.github.io/assets/readme/theorem1.png" alt="Theorem 1" style="width: 96%; height: auto;">
</div>

🚀 **Results.** On Qwen3-30B-A3B-Base (16k rollout), CPPO reaches **54.79% Avg@16 on
AIME 24/25/26**, delivering stronger performance while maintaining greater training stability.

<div align="left">
  <img src="https://hunyuan-cppo.github.io/assets/readme/table1.png" alt="AIME results" style="width: 96%; height: auto;">
</div>

💡 **Takeaway.** DPPO improved *how* to measure token-level shift; CPPO addresses *where* and
*how much* shift can accumulate. For long-horizon RL, the trust-region geometry must align
with the autoregressive prefix-to-suffix structure.

> **Divergence variant.** This example implements the **Binary-TV** divergence
> `D_t = |pi(y_t|s_t) - mu(y_t|s_t)|`, which only needs the sampled token's log-probs and is
> therefore self-contained in `core_algos`. The paper's main experiments use a Top-K (K=20)
> reduced-TV estimate of `D_t` (the two are comparable). The Top-K variant needs vLLM to emit
> per-token top-K log-probs and a matching gather on the training side, which is extra plumbing
> outside the loss; if you want it, you can wire that up in your own verl fork and feed the
> Top-K `D_t` into the same mask.

### Anchoring the divergence to the rollout policy

CPPO's per-token divergence is `D_t = |pi(y_t|s_t) - mu(y_t|s_t)|`, where `mu` is the
**rollout** policy. So the `old_log_prob` fed to the loss must be the rollout log-probs,
not a recomputed `pi_old`. The script runs the synchronous trainer
(`verl.trainer.main_ppo_sync`) with `algorithm.rollout_correction.bypass_mode=True`:
in this trainer, bypass mode simply sets `old_log_probs = rollout_log_probs` and leaves
`loss_mode` untouched, so CPPO reads `mu` directly and runs its own token mask. This
requires `rollout.calculate_log_probs=True` (the default) and the TransferQueue backend
(`pip install TransferQueue`).

## ⚙️ Key hyperparameters

CPPO adds two hyperparameters on top of the shared DPPO token-level threshold scale.

| Config | Meaning | Default |
| --- | --- | --- |
| `actor.clip_ratio` | token-level threshold scale `delta` (same field DPPO uses) | `0.20` (MoE) / `0.15` (dense) |
| `actor.policy_loss.cppo.cppo_w_min` | weight floor of the linear position schedule `w_t in [w_min, 1]` | `0.8` |
| `actor.policy_loss.cppo.cppo_delta_b` | floor `delta_b_min` of the prefix-average budget | `0.02` |
| `actor.policy_loss.cppo.cppo_delta_b_q` | quantile of `D_t` used to calibrate the budget | `0.9` |
| `actor.policy_loss.cppo.cppo_delta_b_k` | scale applied to that quantile | `1.0` |

The prefix-average budget is calibrated per sequence from its own divergence statistics
(Base-model warm-up):
`delta_b^seq = clamp(cppo_delta_b_k * quantile(D_t, cppo_delta_b_q), delta_b_min, 2 * delta_b_min)`.
The defaults `(q, k) = (0.9, 1.0)` reproduce the paper's P90 calibration; for example
`cppo.cppo_delta_b_q=0.95 cppo.cppo_delta_b_k=0.5` calibrates from half the 95th percentile.

`clip_ratio` is the **divergence** threshold here (not a ratio clip); `clip_ratio_c` is the
truncated-importance-sampling cap, kept consistent with DPPO.

## 📐 Mask

For a response of length `T`, with `D_t = |pi(y_t|s_t) - mu(y_t|s_t)|` (Binary-TV):

```
w_t = w_min + (1 - w_min) * (T - t) / (T - 1)       # decreasing position weight, w_t in [w_min, 1]
Z_t = w_t * D_t
S_t = sum_{j<=t} Z_j,   W_t = sum_{j<=t} w_j        # S_0 = W_0 = 0
c_t = min(delta, delta + delta_b * W_{t-1} - S_{t-1})
keep token t  iff  A_t * (rho_t - 1) <= 0   OR   Z_t <= c_t
```

The first clause always keeps update terms that move `pi` back toward `mu`; the budget
only restricts terms that move `pi` farther from `mu`.

## Citation

If you find our work useful for your research, please consider citing:

```bibtex
@article{mao2026beyond,
  title={Beyond Uniform Token-Level Trust Region in LLM Reinforcement Learning},
  author={Mao, Renjie and Zhou, Xiangxin and Tao, Lvfang and Ding, Yixin and Shi, Yu and Lin, Yongguang and Wu, Yuheng and Zhu, Honglin and Qiu, Qian and Zhu, Wenxi},
  journal={arXiv preprint arXiv:2606.10968},
  year={2026}
}
```

## 🌻 Acknowledgement

We implement our reinforcement learning algorithm extending from
[verl](https://github.com/verl-project/verl). We utilize [vLLM](https://github.com/vllm-project/vllm)
and [sglang](https://github.com/sgl-project/sglang) for inference. Our models are trained
primarily on the [Qwen3 family](https://huggingface.co/collections/Qwen/qwen3). Our training
data is built from [DAPO-MATH](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k).
Thanks for their great contributions!
