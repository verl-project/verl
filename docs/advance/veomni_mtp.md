# VeOmni 后端的 MTP 强化学习训练

本接入支持 Qwen3.5 dense / MoE 的 `ForConditionalGeneration`，要求同时使用包含本次修改的 verl 和 VeOmni。VeOmni 模型生成基于 `transformers==5.16.1`。本次仅进行了代码生成和静态审查，未在 GPU/NPU 上验证训练、数值一致性或吞吐。

## 配置

在现有可运行的 VeOmni PPO/GRPO 配置上增加：

```yaml
model_engine: veomni
actor_rollout_ref:
  model:
    use_remove_padding: true
    use_fused_kernels: true
    mtp:
      enable: true
      enable_train: true
      enable_rollout: false
      detach_encoder: true
      mtp_loss_scaling_factor: 0.1
  actor:
    veomni:
      init_device: meta
      ulysses_parallel_size: 1
      cross_entropy_loss_implementation: chunk_loss
      router_replay:
        mode: disabled
```

配置键是 **`use_fused_kernels`**，不是 `use_fuse_kernel`。checkpoint 的 `text_config.mtp_num_hidden_layers` 必须大于 0，并应包含匹配的 `mtp.*` 权重；不要通过盲目覆盖层数改变 checkpoint 结构。

示例脚本复用现有 Qwen3.5 GRPO 启动参数：

```bash
model_path=/path/to/Qwen3.5-35B-A3B \
data_path=/path/to/geo3k \
USE_FUSED_KERNELS=True \
bash examples/mtp_trainer/run_qwen3_5_mtp_veomni.sh
```

设置 `USE_FUSED_KERNELS=False` 切换另一条 policy 路径。训练规模、rollout TP、数据长度等仍需按实际环境修改或通过脚本末尾参数覆盖。

## 两条 kernel 路径

| 配置 | 主 policy | MTP |
| --- | --- | --- |
| `use_fused_kernels=True` | 传入 `labels`、已移位的 `shift_labels`、`return_log_probs=True`，VeOmni 返回可反传的 log-prob/entropy，避免主分支完整词表 logits | 独立 teacher-forced CE，返回标量 `output.mtp_loss` |
| `use_fused_kernels=False` | 不传主分支 labels，返回 logits，由 verl 计算 PPO log-prob/entropy | 显式 `compute_mtp=True`，允许主分支 `labels=None` 时训练 MTP |

`cross_entropy_loss_implementation` 独立控制 MTP CE 的实现。即使 policy 使用 non-fused 路径，也可以让 MTP 使用 `chunk_loss`，减少额外词表投影的显存开销。选择 `eager` 会物化 MTP logits。

MTP CE 使用温度 1，不继承 policy 的 sampling temperature、`return_log_probs`、teacher top-K 张量或主分支 loss 分母。MTP 与 PPO 的 loss scale 各自只应用一次。

两条 policy 路径均要求 `use_remove_padding=True`，因为当前 Qwen3.5 decoder 依赖 packed varlen metadata。packed 后的 bucket padding 用 `-100` 补齐 MTP labels。fused top-K distillation 仍遵循既有的 `pad_to_length` 限制。

## 目标、梯度与归一化

先在每个样本内构造完整序列目标：prompt 和 response mask 为 0 的 token 均设为 `-100`。对于多轮 agent 数据，保留 tool/observation mask 中的空洞；实际 response 长度来自 attention mask 或 jagged 长度，不用 response mask 的有效 token 数代替。

随后，depth `d` 在位置 `i` 预测同一样本的 token `i+d+2`，尾部越界目标为 `-100`。因此 packed 样本之间不会互相借用目标。

设 `C_m` 为一个 micro-batch 所有 MTP depth 的有效目标数，`C` 为当前 DP batch 的总有效目标数，`D` 为 DP size，`α` 为 `mtp_loss_scaling_factor`：

```text
L_micro = L_policy_micro + α × MTP_CE_mean_micro × C_m / max(C, 1) × D
```

累积所有 micro-batch，并经 FSDP/EP 梯度平均后，MTP 项就是全 DP batch 的有效目标平均 CE。不会额外按 micro-batch 数或 depth 数再除一次。`mtp_loss` 指标记录未乘 α 的归一化贡献，`mtp_loss_scaled` 记录乘 α 后的贡献；两者在 micro-batch 间求和、DP 间平均。

某个 rank/micro-batch 没有有效目标时，仍执行 MTP forward，并返回连接计算图的零 loss，避免不同 rank 走不同的 collective/backward 路径。

`detach_encoder=True` 只切断 **MTP loss** 到主干、共享 embedding 和共享 lm_head 的梯度；PPO 仍正常更新这些参数，MTP 各深度之间仍保留递归梯度。

## 开关、保存与 rollout

- `enable=False`：不构建 MTP，保留 HFModelConfig 已处理的层数禁用和 config overrides。
- `enable=True, enable_train=False`：构建、加载和保存 head，冻结其独有参数，forward 跳过 MTP。
- `enable_train=True, mtp_loss_scaling_factor=0`：等价于不计算 MTP loss，head 保留但冻结。
- old-policy/ref log-prob 和 validation 的 `forward_only=True` 路径显式 `compute_mtp=False`，不依赖 `module.training`。
- `enable_rollout` 独立控制推理侧 speculative decoding；仅训练 MTP 无需启用它。

`mtp.*` 继续沿用原有 checkpoint 名称；MoE 的 MTP expert 使用已有 VeOmni converter 及 parallel plan，verl 导出保留 MTP 前缀。vLLM worker 已有同步主模型和 drafter 的分支，本次没有改写 rollout 算法。需要投机 rollout 时，另行打开 `enable_rollout` 并使用支持对应 Qwen3.5 MTP 的推理版本。

从 checkpoint 恢复时应保持 MTP 构建状态和训练/冻结配置一致，否则参数或 optimizer group 可能不匹配。

## 当前边界

- 支持 Qwen3.5 dense / MoE；当前 VeOmni 的 DeepSeek-v4 没有可训练 MTP head，不包含在本接入内。
- 不支持 MTP + Ulysses/SP/CP；初始化会明确拒绝，不能用于需要 CP 的超长序列方案。
- 不支持 MTP training + router replay 或 LoRA，配置会明确报错。
- RL 接入合并 MTP CE，尚未合并模型的 router load-balancing aux loss；启用 MTP 训练时，`output_router_logits=True` 且 `router_aux_loss_coef!=0` 会报错，避免丢弃配置的辅助目标。
- VeOmni GPU/NPU 共用的 forward 已同步生成；是否具备对应设备的算子依赖仍由实际环境决定。
- 未验证分布式训练、重启恢复、权重同步和 speculative acceptance；静态检查不能替代这些数值与集成验证。
