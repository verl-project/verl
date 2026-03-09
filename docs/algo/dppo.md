# Divergence Proximal Policy Optimization (DPPO)

Last updated: 02/25/2026.


<div align="center">

## Rethinking the Trust Region in LLM Reinforcement Learning

[![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white )](https://arxiv.org/pdf/2602.04879)
[![Github](https://img.shields.io/badge/Stable_RL-000000?style=for-the-badge&logo=github&logoColor=000&logoColor=white)](https://github.com/sail-sg/Stable-RL)
[![Twitter](https://img.shields.io/badge/Twitter-%23000000.svg?style=for-the-badge&logo=twitter&logoColor=white)](https://x.com/QPHutu/status/2019435642539897303)

</div>


## ✨Getting started

1. Prepare the datasets by running [prepare_dapo_data.sh](https://github.com/verl-project/verl-recipe/blob/3490a22a0a3adeb7e4787fe70b1060b642efbae4/dapo/prepare_dapo_data.sh):

```bash
bash prepare_dapo_data.sh # This downloads the datasets to ${HOME}/verl/data by default
```

2. Prepare the model:

```bash
hf download Qwen/Qwen3-30B-A3B-Base --local-dir ${HOME}/verl/models/Qwen3-30B-A3B-Base
```

3. Run the script:

```bash
# run DPPO-Binary-KL
LOSS_MODE=dppo_kl bash examples/dppo_trainer/run_qwen30b_dppo.sh

# run DPPO-Binary-TV
LOSS_MODE=dppo_tv bash examples/dppo_trainer/run_qwen30b_dppo.sh

# run GRPO baseline
LOSS_MODE=vanilla CLIP_LOW=0.2 CLIP_HIGH=0.2 bash examples/dppo_trainer/run_qwen30b_dppo.sh
# or GRPO with clip higher
LOSS_MODE=vanilla CLIP_LOW=0.2 CLIP_HIGH=0.28 bash examples/dppo_trainer/run_qwen30b_dppo.sh
```

## 📖Introduction

<div align="left">
  <img src="https://github.com/sail-sg/Stable-RL/blob/main/figures/ppo_vs_dppo.jpg?raw=true" alt="issue" style="width: 96%; height: auto;">
</div>

Comparison of **PPO** and the proposed **DPPO** (the Binary-TV variant). **(Left)** The surrogate objective and corresponding masks for PPO and DPPO. PPO (and variants like GRPO) employs a heuristic mask based on the probability ratio. In contrast, DPPO utilizes a more principled mask based on a direct approximation of policy divergence (e.g., Total Variation), ensuring updates stay within a theoretically grounded trust region. **(Right)** Experimental results on the AIME24 using Qwen3-30B-A3B-Base. DPPO significantly outperforms GRPO baselines, achieving superior training stability and final performance even without rollout routing replay (R3).

<div align="left">
  <img src="https://github.com/sail-sg/Stable-RL/blob/main/figures/sanity_test.png?raw=true" alt="issue" style="width: 96%; height: auto;">
</div>

DPPO variants achieve stable training while controlling the training-inference mismatch at a low level. In contrast, methods without a trust region (PG-IS, CISPO) or with a misspecified one (MiniRL) suffer from growing mismatch and eventual collapse.

<div align="left">
  <img src="https://github.com/sail-sg/Stable-RL/blob/main/figures/moe_prob_ratio_tv.png?raw=true" alt="issue" style="width: 96%; height: auto;">
</div>

The plots show numerical differences between a training and an inference engine for Qwen3-30B-A3B-Base with identical parameters. **(Left)** The probability ratio (used in PPO) is highly volatile for low-probability tokens. **(Right)** In contrast, the TV divergence is more stable. This highlights a key flaw of PPO's clipping mechanism: it **over-penalizes low-probability tokens**, which can slow down learning; and **under-penalizes high-probability tokens**, which can permit large, destabilizing updates.


<div align="left">
  <img src="https://github.com/sail-sg/Stable-RL/blob/main/figures/clipped_tokens.png?raw=true" alt="issue" style="width: 96%; height: auto;">
</div>

The most frequently clipped tokens (by GRPO) are important to the reasoning task! 
They are dominated by:
- numbers, like 1, 4
- mathematical symbols, like +, -, =
- reasoning and structural Words: Wait, Thus, Next

## Top-K divergence approximation

We only implement the DPPO-Binary-TV/DPPO-Binary-KL here due to their simplicity.

For the TopK divergence approximation, please refer to the [the original repo](https://github.com/sail-sg/Stable-RL) for a complete implementation.

## Citation
If you find our works useful for your research, please consider citing:

```bibtex
@article{qi2026dppo,
  title={Rethinking the Trust Region in LLM Reinforcement Learning},
  author={Qi, Penghui and Zhou, Xiangxin and Liu, Zichen and Pang, Tianyu and Du, Chao and Lin, Min and Lee, Wee Sun},
  journal={arXiv preprint arXiv:2602.04879},
  year={2026}
}
```

## 🌻Acknowledgement
We implement our reinforcement learning algorithm extending from [verl](https://github.com/volcengine/verl). We utilize [vLLM](https://github.com/vllm-project/vllm) and [sglang](https://github.com/sgl-project/sglang) for inference. Our models are trained primarily on [Qwen3 family](https://huggingface.co/collections/Qwen/qwen3). Our training data is built from [DAPO-MATH](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k). Thanks for their great contributions!
