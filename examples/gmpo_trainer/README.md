<div align=center>
  
# Geometric-Mean Policy Optimization
</div>

This is the official implementaion of paper [***Geometric-Mean Policy Optimization***](https://arxiv.org/abs/2507.20673).

<div align=center>
<img width="3092" height="864" alt="image" src="https://github.com/user-attachments/assets/20b04c4e-7ee8-4775-9af8-33c0158336e2" />
</div>

## 1. Contents
- Geometric-Mean Policy Optimization
  - [1. Contents](#1-contents)
  - [2. Introduction](#2-introduction)
  - [3. Code Usage](#3-code-usage)
  - [4. Contacts](#4-contacts)
  - [5. Citation](#5-citation)

## 2. Introduction

Group Relative Policy Optimization (GRPO) has significantly enhanced the reasoning capability of large language models by optimizing the arithmetic mean of token-level rewards. Unfortunately, GRPO is observed to suffer from unstable policy updates when facing tokens with outlier importance-weighted rewards, which manifest as extreme importance sampling ratios during training. In this study, we propose Geometric-Mean Policy Optimization (GMPO), with the aim to improve the stability of GRPO through suppressing token reward outliers. Instead of optimizing the arithmetic mean, GMPO maximizes the geometric mean of token-level rewards, which is inherently less sensitive to outliers and maintains a more stable range of importance sampling ratio. GMPO is plug-and-play—simply replacing GRPO's arithmetic mean with the geometric mean of token-level rewards, as the latter is inherently less sensitive to outliers. GMPO is theoretically plausible—analysis reveals that both GMPO and GRPO are weighted forms of the policy gradient while the former enjoys more stable weights, which consequently benefits policy optimization and performance. Experiments on multiple mathematical reasoning benchmarks show that GMPO-7B improves the average Pass@1 of GRPO by up to 4.1%, outperforming many state-of-the-art approaches.

## 3. Code Usage

The key configurations are:
```
clip_ratio_low=0.4
clip_ratio_high=0.4
loss_mode=geo_mean
```
We observed that using a large clip ratio during Mixture-of-Experts (MoE) model training often leads to optimization instability. When training MoE models, consider lowering the clip ratio to achieve more stable convergence.

To get started quickly:

```bash
bash examples/gmpo_trainer/run_qwen3_8b_fsdp.sh
# override any of MODEL_PATH, CLIP_RATIO, ROLLOUT_N, etc. via env vars
```

## 4. Contacts
If you have any question about our work or this repository, please don't hesitate to contact us by emails or open an issue under this project.
- [zhaoyuzhong20@mails.ucas.ac.cn](zhaoyuzhong20@mails.ucas.ac.cn)
- [liuyue171@mails.ucas.ac.cn](liuyue171@mails.ucas.ac.cn)
- [lecu@microsoft.com](lecu@microsoft.com)
- [wanfang@ucas.ac.cn](wanfang@ucas.ac.cn)

## 5. Citation
```
@article{zhao2025geometric,
  title={Geometric-mean policy optimization},
  author={Zhao, Yuzhong and Liu, Yue and Liu, Junpeng and Chen, Jingye and Wu, Xun and Hao, Yaru and Lv, Tengchao and Huang, Shaohan and Cui, Lei and Ye, Qixiang and others},
  journal={arXiv preprint arXiv:2507.20673},
  year={2025}
}
```
