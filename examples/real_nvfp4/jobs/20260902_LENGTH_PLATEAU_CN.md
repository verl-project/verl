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

## 4b. 跑通 BF16 对照踩到的三个 bug（2026-09-02）

`PRECISION_MODE=bf16` 在此之前从未被执行过，一跑就连挂。按发现顺序：

### bug 1：receiver 异常会把 sender 永久挂死（已修）

`receive_weights` 在 consumer 抛异常时既不 ACK 也不 drain，直接 `_cleanup()` 关 socket。
sender 正阻塞在 `socket.recv()` 等这个 bucket 的 ACK，而 verl 是先跑完
`sender.async_send_weights()` 再 `await` receiver 的 future——所以 **receiver 的异常永远没机会抛出来**，
表现为静默挂死。8 节点那次因此烧满 5 小时、0 步、无任何报错。

`_iter_weights`（real NVFP4 用的迭代器路径）本来就有 `_drain_remaining_buckets_after_failure()`，
legacy 路径没有。**就是这个不对称让 W4A4 一路正常、BF16 一碰即死。**
修法：ACK 放进 `finally`、drain 剩余 bucket、再 re-raise。修完故障从「5 小时静默」变成「5 分 23 秒带 traceback」。

py-spy 证据（从镜像抽出 aarch64 二进制、宿主侧 attach）：4 个 sender 全部冻在
`bucketed_weight_transfer.py:138`、同一个 tensor `model.layers.0.mlp.experts.77.gate_proj.weight`，
4 次采样 offset 一字不变；4 个 receiver 全部已返回 `worker_busy_loop`。

### bug 2：老 MoE loader 补丁在现代 vLLM 上有害（已修，但不是本次病根）

`patch_vllm_moe_model_weight_loader` 是给 vLLM **0.8.2** 打的（当时 w13/w2 param 没有 weight_loader），
它会把 param 上的 loader 覆盖成模块级的 `experts.weight_loader`。vLLM ≥ 0.11 在 `create_weights`
里已经装了正确的 loader，覆盖它有害无益。补丁自己的注释就写着「not need anymore for newer vllm version」。
已改为：只要 vLLM 暴露 `RoutedExperts` 就跳过。

**但这不是本次的病根**——完整 traceback 显示实际被调用的是 vLLM **自己的**
`RoutedExperts.weight_loader`（routed_experts.py:858）。我一开始判错了方向。

### bug 3（真病根）：verl legacy MoE 权重同步不兼容 vLLM 0.26

```
routed_experts.py:914  param.weight_loader(...)
routed_experts.py:858  weight_loader -> _load_model_weight_or_group_weight_scale
routed_experts.py:356  -> _load_w13
routed_experts.py:490  hidden_dim = self._get_hidden_dim(shard_dim, expert_data.ndim)
routed_experts.py:409  ValueError: shard_dim=0 is not a valid data dimension for a 3D tensor
```

vLLM 0.26 把 MoE 加载重构成 `RoutedExperts`，其 `weight_loader` 要的是逐 expert 的 **2D** 视图，
而 verl 的 legacy `model.load_weights` 递过去的是堆叠的 **3D** expert param。
**verl 主线还 pin 在 vLLM 0.24，所以这条路径在 0.26 上根本没人跑过**——是上游空档，不是本分支弄坏的。
real NVFP4 完全看不到它，因为那条路径压根不调 `model.load_weights`。

处理：`VERL_VLLM_NATIVE_RELOAD=1` 让非量化路径改走
`model_runner.reload_weights(..., is_checkpoint_format=True)`，即 real NVFP4 在同一个 vLLM 上
已经跑通的那个 API。默认行为不变，只有对照 opt-in；`train.job` 断言 marker，防止悄悄回落到坏路径
还冒充对照。把 legacy loader 移植到 RoutedExperts API 是更完整的修法，但那是相对当前问题的绕路。

**注意**：worker 进程只认 Ray `runtime_env` 注入的环境变量，driver shell 里 `export` 传不到——
第一次就是这么白跑一轮的（`VERL_VLLM_NATIVE_RELOAD PASS` 计数为 0 即是证据）。
控制组专用的 `runtime_env_native_reload.yaml` 放在 bundle 目录内（provenance 排除范围），不必重建镜像。

### 结果

1 节点 smoke：step 1/2/3 全部完成、final validation 通过、`shard_dim` 错误归零、
`VERL_VLLM_NATIVE_RELOAD PASS` × 9。唯一失败是收尾 checkpoint 被 SIGKILL（宿主 OOM）——
BF16 rollout 每卡常驻约 61GB 权重（NVFP4 约 15GB），1 节点存 30B full-Adam 时宿主内存不够。
8 节点上 checkpoint 分片到 32 rank，W4A4 已证明可存 399GB，因此不阻塞对照。

## 5. 下一个候选（对照结果出来后再查）

若对照涨，优先查 verl 特有、且与精度耦合的项：

- **TIS —— 我用错了统计量，这条结论已推翻，现为首要假设（见第 6 节）。** verl 用 `rollout_is=token` + `threshold=2.0`，
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


## 6. 配方逐项比对（零机时）与 TIS 假设（2026-09-02）

BF16 对照三次失败后停下来重估，把 verl 与 NeMo 的 RL 配方全部比完：

| 项 | verl | NeMo | |
| --- | --- | --- | --- |
| adv estimator | grpo | grpo | 同 |
| ratio clip | 0.2 / 0.28 / c=10 | `ratio_clip_min/max/c` 0.2 / 0.28 / 10 | **同** |
| loss 归一化 | `token-mean` = `sum(loss*mask)/global_tokens * dp_size` | token-level `masked_mean(..., global_normalization_factor=global_valid_toks)` = `sum(loss*mask)/global_valid_toks` | **同**（verl 的 `*dp_size` 只是抵消 DDP 梯度平均） |
| lr / batch / temp / 数据 / 0-3 loss | 1e-6 / 512 / 1.0 / DAPO-Math-17k / 0of3 | 同 | 同 |
| **TIS** | `rollout_is=token`, `threshold=2.0` | `loss_fn` **无任何 IS 修正** | **不同** |

其余差异（dynamic sampling、reward shaping overlong buffer、lr warmup 10、
`max_input_seq_length` 2048 vs 1024）单项都太小，且 prompt 长度实测一致（150.1 vs 152.2）。

### 为什么之前把 TIS 排掉是错的

我用的判据是「触界比例只有 0.4%」。**这个统计量不对**：TIS 是给**每个** token 的 loss 乘
`w = clamp(exp(old_logp - rollout_logp), max=2.0)`，触界与否不重要，重要的是 `w` 的分布。

| arm | `rollout_is_std` | `rollout_corr/kl` | 长度 |
| --- | --- | --- | --- |
| verl BF16 0603 | 0.037 | 0.0012 | 涨（≤265 步 +34.6/step） |
| **verl W4A4 v13** | **0.089** | **0.0044** | **平（260 步 +0.88）** |
| NeMo W4A4（无 IS 修正） | — | — | 涨（+16.9/step） |

W4A4 的 mismatch 是尾部集中的（本工作线既有结论：按 `rollout_logp` 排序的 bottom 20%
贡献约 74% 的 |delta|）。若 mismatch 随 token 位置累积，TIS 就会系统性压低靠后的 token
——正好压住「长度」这个量。

**一个机制同时解释四条曲线**：NeMo 无 TIS → 涨；verl BF16 有 TIS 但 mismatch 小 4 倍、
`w≈1` 失效 → 涨；verl W4A4 TIS 生效 → 平。

### 验证臂

`jobs/w4a4_tis_off_20260902_v18/`，job **2707772**，8 节点 120 步，
exp `verl_30b_w4a4_tisoff_8n_20260902_v18_j2707772`。相对 v13 只改一个变量：
`ROLLOUT_IS=null`。判读：**起飞 → TIS 是 real W4A4 的长度抑制源，修法是精度感知的 IS 策略，
不是长度/entropy 旋钮；仍平 → TIS 洗清，锁定 verl 的 W4A4 数值路径**，届时再把 BF16 sync
移植到 RoutedExperts API 拿干净对照。

另：拉了 v13 step 259 的实际生成，模型输出是结构完整、评分 1.0 的数学推理——策略健康，
只是学不会写更长，不是退化。
