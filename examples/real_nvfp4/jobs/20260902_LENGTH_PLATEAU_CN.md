# verl real-W4A4 长跑：response length 不涨（2026-09-02）

W4A4 长跑链 `verl_30b_realw4a4_r3_nativeonline_8n_long_20260901_v13`
（jobs 2695466/2695467/2695468）跑到 step 210，机制全对但**长度不涨**。本文记录证据、
已排除项和当前唯一在跑的判别实验。

## 1. 现象

210 步，`response_length/mean` 从 1015 → 1124（**+0.31 tok/step**）。同期：

| 窗口 | verl W4A4 | verl BF16 0603 | NeMo W4A4 R3-on | NeMo W4A4 R3-off |
| --- | --- | --- | --- | --- |
| 1–40 | 1004 (−0.2) | 870 (+4.9) | 841 (−4.7) | 852 (−1.6) |
| 41–80 | 959 (−1.9) | 1288 (+20.1) | 1051 (+9.3) | 1091 (+9.9) |
| 81–120 | 951 (+0.4) | 4149 (+55.9) | 1995 (+52.0) | 1787 (+30.1) |
| 121–160 | 997 (+2.5) | 5256 (+19.6) | 4134 (+18.6) | 3768 (+31.0) |
| 161–210 | 1037 (+2.0) | 6158 (+27.5) | 4432 (+13.8) | 4027 (+9.3) |

**前 40–80 步四条曲线几乎重合，随后除 verl W4A4 外全部在 step 80 附近起飞。**

reward 侧：verl W4A4 −0.637 → −0.031（确实在学），但 161–210 窗口均值 −0.106，
NeMo 同期 +0.09、BF16 +0.18。

参照 run：

- NeMo R3-on：`nv-welcome/qwen3-30b-nvfp4/bed568c189da5cc77c4a18b96560bd91`
- NeMo R3-off：同项目 `ecbf3eb94bd1d0e574797c867bbe33e7`
  （`...-20260806-0of3-tokenmean-allmlp-nooverlong-dequant-te3049-v1`）
- NeMo 指标名与 verl 不同：长度是 `train/mean_gen_tokens_per_sample`，
  reward 是 `train/reward`，entropy 是 `train/approx_entropy`。

### 1b. 全链 260 步（chunk 4 实际已完成）

chunk 4（2695469）实际跑完了，`chunk_4.pass` 存在，W&B 有 260 步。补上末段：

| 窗口 | verl W4A4 长度 | 斜率 | verl W4A4 score | BF16 长度 | BF16 score | NeMo R3-off 长度 | NeMo reward |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 121–160 | 997 | +2.53 | −0.174 | 5256 | +0.114 | 3768 | +0.049 |
| 161–210 | 1037 | +1.95 | −0.106 | 6158 | +0.177 | 4027 | +0.093 |
| 211–260 | 1188 | +1.74 | −0.089 | 7126 | +0.197 | 4665 | +0.167 |

`critic/score/mean` 在 step 260 已经**转正到 +0.055**（step 210 时是 −0.031），说明仍在学习。
长度全程斜率 +0.88 tok/step、末段 +1.74 tok/step——**不是冻住，方向也是正的（旧实现是 −0.67），
但比三条参照慢一个数量级**。定性结论不变。

## 2. 已排除

**不是实现坏了。** KL 0.0044、ESS 0.9930、每步 refit + R3 48 层 + attestation 全过、
399 GiB full-Adam checkpoint、resume 连过两次（chunk 1→2→3）。

**不是生成能力受限或截断。** 每个窗口 `response_length/max` 都打到 20480，
`response_length/clip_ratio` ≈ 0。模型写得出长的，只是分布不往长走。

**不是 entropy / 探索不足。** BF16 在 step 80 的 entropy 是 0.086，比 verl W4A4 的
0.122 还低，照样起飞。NeMo W4A4 在 81–210 是 0.085–0.101，与 verl 的 0.104–0.112 基本一致。

**不是 FP4 固有限制。** NeMo 在同一精度合同（all-MLP routed expert、0/3 loss、
token-mean、nooverlong、dequantized backward）下长到 4000+。

**dynamic sampling 基本排除（曾被误判为主因）。** verl 这条 run
`algorithm.filter_groups.enable=False`、`gen_batch_size=32`，而 BF16 0603 是
`enable=True, metric=acc, max_num_gen_batches=10, gen_batch_size=64`，NeMo 是
`use_dynamic_sampling=True, batch_multiplier=3, max_gen_batches=20`——变量确实对齐结果。
但 NeMo 记录的退化 group 比例是 `pct_0` 2.6–5.3%、`pct_1` 0.0%、**`pct_mixed` 94.7–97.4%**，
即只有约 4% 的 group 是全对/全错。**回收 4% 解释不了"平 vs 涨 8 倍"**，因此它顶多是次要因素。

## 3. 为什么 BF16 0603 不是干净对照

`gb200_30B_bf16_megatron_0603` 与 W4A4 arm 同时差三样：精度、recipe
（dynamic sampling 开/关、`gen_batch_size` 64/32）、以及整个栈
（6 月的 `verl_megatron_qat_v020` vs 现在的 v12）。所以它证明不了"差异来自精度"。

已核对**相同**的部分：`max_response_length=20480`、`train_batch_size=32`、
`rollout.n=16`、`temperature=1`、`max_prompt_length=1024`、`lr=1e-6`、`grpo`、
`token-mean`、`clip 0.2/0.28`、`ppo_mini_batch_size=32`、8×4 节点、
`norm_adv_by_std_in_grpo=True`、`rollout_is=token` + `threshold=2`。
prompt 管线也一致（`prompt_length/mean` 150.1 vs 152.2，max 909 vs 901）。

## 4. 判别实验（进行中）

`jobs/bf16_control_20260902_v14/`，job **2702860**，8 节点 100 步（首次提交 2702808 被坑 15 毙掉），
exp `verl_30b_bf16_control_8n_20260902_v14_j2702860`。**相对 W4A4 arm 只改一个变量：
`PRECISION_MODE=bf16`。** 同一个 v12 镜像、同一份数据 adapter、同样
`filter_groups.enable=False`、`gen_batch_size=32`、同样 clip/lr/TIS/R3。

判读：

- **对照涨** → 问题特定于 verl 的 W4A4 路径；下一步把 verl W4A4 与 NeMo W4A4 的数值逐项对比。
- **对照也平** → 精度被洗清，原因在 verl 共享的 recipe/plumbing；再与 6 月 BF16 对比找出是哪一项。

控制组自身的污染防护：`PRECISION_MODE=bf16` 会置
`actor_rollout_ref.actor.megatron.real_nvfp4.enable=False`，rollout 通过
`rollout.real_nvfp4=${oc.select:actor_rollout_ref.actor.megatron.real_nvfp4,null}` 镜像该值，
因此 vLLM 也走 BF16。`train.job` 额外断言两个 attestation **都不出现**、且没有选中
`nvfp4_per_token`——slime 那次"bf16 baseline 偷偷跑 W4A4 rollout"的坑不能再踩。

## 5. 下一个候选（对照结果出来后再查）

若对照涨，优先查 verl 特有、且与精度耦合的项：

- ~~**TIS**~~ **已量化，基本排除。** verl 用 `rollout_is=token` + `threshold=2.0`，
  即单边上截断 `w = min(exp(old_logp - rollout_logp), 2.0)`；NeMo 的
  `grpo.seq_logprob_error_threshold=None` 没有等价物。W&B 里的绑定比例：

  | | `rollout_is_ratio_fraction_high` | `..._fraction_low` | `rollout_is_std` | seq 级 |
  | --- | --- | --- | --- | --- |
  | W4A4 | 0.0011 | 0.0028 | 0.089 | 0.0 |
  | BF16 | 0.00007 | 0.00017 | 0.037 | 0.0 |

  W4A4 触界频率是 BF16 的约 15 倍、权重离散度约 2.4 倍——**确实是随精度变化的真实效应**，
  但绝对量只有约 0.4% 的 token，且 sequence 级触界恒为 0。和 dynamic sampling 同样的教训：
  先量化再定性。若对照实验指向 W4A4 路径，可以用一个 flag
  （`algorithm.rollout_correction.rollout_is=null`）做 TIS-off 臂来彻底证伪，成本很低。
- NeMo 开了 `reward_shaping`（overlong buffer 512 / penalty 1），verl 这条关着。
  当前长度离 20480 很远，该项应当是惰性的，属于低优先级。
