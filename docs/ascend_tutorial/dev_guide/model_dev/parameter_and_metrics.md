# 训练配置参数与指标说明

Last updated: 05/12/2026.

verl 通过层级化的 YAML 配置文件管理所有参数，涉及到的所有配置文件均在 `verl\trainer\config` 目录下。

---

## 1. 配置参数说明

### 1.1 公共配置参数

以下参数在 FSDP 方案和 Megatron 方案中均存在且含义一致。

#### 1.1.1 Actor 优化器配置

| 参数名 | 默认值 | 说明 |
|--------|--------|------|
| `actor_rollout_ref.actor.optim.lr` | `1.0e-06` | Actor 学习率 |
| `actor_rollout_ref.actor.optim.lr_warmup_steps_ratio` | `0.0` | 学习率预热步数占总训练步数的比例 |
| `actor_rollout_ref.actor.optim.total_training_steps` | `-1` | 总训练步数，-1 表示自动计算 |
| `actor_rollout_ref.actor.optim.weight_decay` | `0.01` | 权重衰减，用于防止模型过拟合 |
| `actor_rollout_ref.actor.optim.lr_warmup_steps` | `-1` | 学习率预热步数，-1 表示由 ratio 自动计算 |
| `actor_rollout_ref.actor.optim.betas` | `[0.9, 0.999]` | Adam 优化器的一阶和二阶动量系数 |
| `actor_rollout_ref.actor.optim.clip_grad` | `1.0` | 梯度裁剪阈值 |
| `actor_rollout_ref.actor.optim.override_optimizer_config` | `null` / `{}` | 覆盖优化器配置（FSDP 为 null，Megatron 为 {}） |

#### 1.1.2 Actor 策略配置

| 参数名 | 默认值 | 说明 |
|--------|--------|------|
| `actor_rollout_ref.actor.strategy` | `fsdp` / `megatron` | 训练策略，FSDP 方案为 fsdp，Megatron 方案为 megatron |
| `actor_rollout_ref.actor.ppo_mini_batch_size` | `256` | PPO 训练的 mini batch 大小 |
| `actor_rollout_ref.actor.ppo_micro_batch_size` | `null` | PPO 训练的 micro batch 大小 |
| `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu` | `null` | 每 GPU 的 PPO micro batch 大小 |
| `actor_rollout_ref.actor.use_dynamic_bsz` | `false` | 是否使用动态 batch size |
| `actor_rollout_ref.actor.ppo_max_token_len_per_gpu` | `16384` | 每 GPU 的 PPO 最大 token 长度 |
| `actor_rollout_ref.actor.clip_ratio` | `0.2` | PPO 裁剪比例，控制策略更新幅度，一般取值范围 [0.1, 0.3] |
| `actor_rollout_ref.actor.clip_ratio_low` | `0.2` | PPO 下界裁剪比例 |
| `actor_rollout_ref.actor.clip_ratio_high` | `0.2` | PPO 上界裁剪比例 |
| `actor_rollout_ref.actor.tau_pos` | `1.0` | 正优势裁剪的 tau 参数 |
| `actor_rollout_ref.actor.tau_neg` | `1.05` | 负优势裁剪的 tau 参数 |
| `actor_rollout_ref.actor.freeze_vision_tower` | `false` | 是否冻结视觉塔（多模态模型） |
| `actor_rollout_ref.actor.clip_ratio_c` | `3.0` | 裁剪比例的上限常数 |
| `actor_rollout_ref.actor.loss_agg_mode` | `token-mean` | 损失聚合模式，可选 token-mean 等 |
| `actor_rollout_ref.actor.loss_scale_factor` | `null` | 损失缩放因子 |
| `actor_rollout_ref.actor.entropy_coeff` | `0` | 熵正则化系数，控制策略探索程度 |
| `actor_rollout_ref.actor.calculate_entropy` | `false` | 是否计算策略熵 |
| `actor_rollout_ref.actor.use_kl_loss` | `false` | 是否使用 KL 散度损失 |
| `actor_rollout_ref.actor.use_prefix_grouper` | `false` | 是否使用前缀分组器 |
| `actor_rollout_ref.actor.use_torch_compile` | `true` | 是否使用 torch.compile 加速 |
| `actor_rollout_ref.actor.kl_loss_coef` | `0.001` | KL 损失系数 |
| `actor_rollout_ref.actor.kl_loss_type` | `low_var_kl` | KL 损失类型，可选 low_var_kl 等 |
| `actor_rollout_ref.actor.ppo_epochs` | `1` | PPO 更新轮数 |
| `actor_rollout_ref.actor.shuffle` | `false` | 训练时是否对 mini batch 进行 shuffle |
| `actor_rollout_ref.actor.data_loader_seed` | `42` | 数据加载器随机种子 |
| `actor_rollout_ref.actor.grad_clip` | `1.0` | 梯度裁剪值 |
| `actor_rollout_ref.actor.ulysses_sequence_parallel_size` | `1` | Ulysses 序列并行大小 |
| `actor_rollout_ref.actor.entropy_from_logits_with_chunking` | `false` | 是否使用分块方式从 logits 计算熵 |
| `actor_rollout_ref.actor.entropy_checkpointing` | `false` | 是否对熵计算使用梯度检查点 |
| `actor_rollout_ref.actor.use_remove_padding` | 引用自 `model.use_remove_padding` | 是否移除 padding |
| `actor_rollout_ref.actor.calculate_sum_pi_squared` | `false` | 是否计算策略概率平方和 |
| `actor_rollout_ref.actor.sum_pi_squared_checkpointing` | `false` | 是否对策略概率平方和计算使用梯度检查点 |
| `actor_rollout_ref.actor.use_fused_kernels` | 引用自 `model.use_fused_kernels` | 是否使用融合内核 |

#### 1.1.3 Policy Loss 配置

| 参数名 | 默认值 | 说明 |
|--------|--------|------|
| `actor_rollout_ref.actor.policy_loss.loss_mode` | `vanilla` | 策略损失模式，可选 vanilla、clip_cov、kl_cov、dppo_tv、dppo_kl、gspo、sapo、geo_mean、cispo、gpg、bypass_mode、reinforce_is 等 |
| `actor_rollout_ref.actor.policy_loss.clip_cov_ratio` | `0.0002` | clip_cov 模式的协方差比率 |
| `actor_rollout_ref.actor.policy_loss.clip_cov_lb` | `1.0` | clip_cov 模式的协方差下界 |
| `actor_rollout_ref.actor.policy_loss.clip_cov_ub` | `5.0` | clip_cov 模式的协方差上界 |
| `actor_rollout_ref.actor.policy_loss.kl_cov_ratio` | `0.0002` | kl_cov 模式的协方差比率 |
| `actor_rollout_ref.actor.policy_loss.ppo_kl_coef` | `0.1` | PPO KL 散度系数 |

#### 1.1.4 Rollout 配置

| 参数名 | 默认值 | 说明 |
|--------|--------|------|
| `actor_rollout_ref.rollout.name` | `???` | Rollout 引擎名称，需用户指定 |
| `actor_rollout_ref.rollout.mode` | `async` | Rollout 模式，可选 async、sync 等 |
| `actor_rollout_ref.rollout.nnodes` | `0` | Rollout 使用的节点数 |
| `actor_rollout_ref.rollout.n_gpus_per_node` | 引用自 `trainer.n_gpus_per_node` | 每节点 GPU 数 |
| `actor_rollout_ref.rollout.temperature` | `1.0` | 采样温度，控制生成随机性 |
| `actor_rollout_ref.rollout.top_k` | `-1` | Top-K 采样参数，-1 表示不启用 |
| `actor_rollout_ref.rollout.top_p` | `1` | Top-P（nucleus）采样参数 |
| `actor_rollout_ref.rollout.prompt_length` | 引用自 `data.max_prompt_length` | Prompt 最大长度 |
| `actor_rollout_ref.rollout.response_length` | 引用自 `data.max_response_length` | Response 最大长度 |
| `actor_rollout_ref.rollout.dtype` | `bfloat16` | Rollout 推理数据类型 |
| `actor_rollout_ref.rollout.gpu_memory_utilization` | `0.5` | GPU 内存利用率，推理时使用 GPU 内存的比例 |
| `actor_rollout_ref.rollout.ignore_eos` | `false` | 是否忽略 EOS token |
| `actor_rollout_ref.rollout.enforce_eager` | `false` | 是否强制使用 PyTorch eager 模式 |
| `actor_rollout_ref.rollout.cudagraph_capture_sizes` | `null` | CUDA Graph 捕获大小列表 |
| `actor_rollout_ref.rollout.free_cache_engine` | `true` | 是否在每次推理后释放缓存引擎 |
| `actor_rollout_ref.rollout.tensor_model_parallel_size` | `2` | 推理时 TP 并行大小 |
| `actor_rollout_ref.rollout.data_parallel_size` | `1` | 推理时数据并行大小 |
| `actor_rollout_ref.rollout.expert_parallel_size` | `1` | 推理时专家并行大小 |
| `actor_rollout_ref.rollout.pipeline_model_parallel_size` | `1` | 推理时 PP 并行大小 |
| `actor_rollout_ref.rollout.max_num_batched_tokens` | `8192` | 单步最大批处理 token 数 |
| `actor_rollout_ref.rollout.max_model_len` | `null` | 模型最大序列长度，null 表示自动推断 |
| `actor_rollout_ref.rollout.max_num_seqs` | `1024` | 推理并发最大样本数 |
| `actor_rollout_ref.rollout.enable_chunked_prefill` | `true` | 是否启用分块预填充 |
| `actor_rollout_ref.rollout.enable_prefix_caching` | `true` | 是否启用前缀缓存（KV Cache 复用） |
| `actor_rollout_ref.rollout.logprobs_mode` | `processed_logprobs` | logprobs 计算模式 |
| `actor_rollout_ref.rollout.scheduling_policy` | `fcfs` | 调度策略，可选 fcfs 等 |
| `actor_rollout_ref.rollout.load_format` | `dummy` | 模型加载格式 |
| `actor_rollout_ref.rollout.log_prob_micro_batch_size` | `null` | log prob 计算的 micro batch 大小 |
| `actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu` | `null` | 每 GPU 的 log prob micro batch 大小 |
| `actor_rollout_ref.rollout.log_prob_use_dynamic_bsz` | 引用自 `actor.use_dynamic_bsz` | log prob 是否使用动态 batch size |
| `actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu` | 引用自 `actor.ppo_max_token_len_per_gpu` | 每 GPU 的 log prob 最大 token 长度 |
| `actor_rollout_ref.rollout.disable_log_stats` | `true` | 是否禁用推理日志统计 |
| `actor_rollout_ref.rollout.do_sample` | `true` | 是否进行采样（false 则为贪心解码） |
| `actor_rollout_ref.rollout.n` | `1` | 每个 prompt 生成的 response 数量 |
| `actor_rollout_ref.rollout.over_sample_rate` | `0` | 过采样率 |
| `actor_rollout_ref.rollout.multi_stage_wake_up` | `false` | 是否启用多阶段唤醒 |
| `actor_rollout_ref.rollout.calculate_log_probs` | `false` | 是否在 rollout 阶段计算 log probs |
| `actor_rollout_ref.rollout.skip_rollout` | `false` | 是否跳过 rollout |
| `actor_rollout_ref.rollout.skip_dump_dir` | `/tmp/rollout_dump` | 跳过 rollout 时的 dump 目录 |
| `actor_rollout_ref.rollout.skip_tokenizer_init` | `true` | 是否跳过分词器初始化 |
| `actor_rollout_ref.rollout.enable_rollout_routing_replay` | `false` | 是否启用 rollout 路由重放 |
| `actor_rollout_ref.rollout.quantization` | `null` | 量化方式 |
| `actor_rollout_ref.rollout.quantization_config_file` | `null` | 量化配置文件路径 |
| `actor_rollout_ref.rollout.layered_summon` | `false` | 是否启用分层召唤（仅 FSDP 方案） |

#### 1.1.5 Rollout 验证采样配置

| 参数名 | 默认值 | 说明 |
|--------|--------|------|
| `actor_rollout_ref.rollout.val_kwargs.top_k` | `-1` | 验证时 Top-K 采样参数 |
| `actor_rollout_ref.rollout.val_kwargs.top_p` | `1.0` | 验证时 Top-P 采样参数 |
| `actor_rollout_ref.rollout.val_kwargs.temperature` | `0` | 验证时采样温度，0 表示贪心解码 |
| `actor_rollout_ref.rollout.val_kwargs.n` | `1` | 验证时每个 prompt 生成的 response 数 |
| `actor_rollout_ref.rollout.val_kwargs.do_sample` | `false` | 验证时是否采样 |

#### 1.1.6 Multi-Turn 配置

| 参数名 | 默认值 | 说明 |
|--------|--------|------|
| `actor_rollout_ref.rollout.multi_turn.enable` | `false` | 是否启用多轮对话 |
| `actor_rollout_ref.rollout.multi_turn.max_assistant_turns` | `null` | 最大助手轮数 |
| `actor_rollout_ref.rollout.multi_turn.tool_config_path` | `null` | 工具配置文件路径 |
| `actor_rollout_ref.rollout.multi_turn.max_user_turns` | `null` | 最大用户轮数 |
| `actor_rollout_ref.rollout.multi_turn.max_parallel_calls` | `1` | 最大并行工具调用数 |
| `actor_rollout_ref.rollout.multi_turn.max_tool_response_length` | `256` | 工具响应最大长度 |
| `actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side` | `middle` | 工具响应截断方向 |
| `actor_rollout_ref.rollout.multi_turn.interaction_config_path` | `null` | 交互配置文件路径 |
| `actor_rollout_ref.rollout.multi_turn.use_inference_chat_template` | `false` | 是否使用推理聊天模板 |
| `actor_rollout_ref.rollout.multi_turn.tokenization_sanity_check_mode` | `strict` | 分词完整性检查模式 |
| `actor_rollout_ref.rollout.multi_turn.format` | `hermes` | 多轮对话格式 |
| `actor_rollout_ref.rollout.multi_turn.num_repeat_rollouts` | `null` | 重复 rollout 次数 |

#### 1.1.7 Agent 配置

| 参数名 | 默认值 | 说明 |
|--------|--------|------|
| `actor_rollout_ref.rollout.agent.num_workers` | `8` | Agent 工作进程数 |
| `actor_rollout_ref.rollout.agent.default_agent_loop` | `single_turn_agent` | 默认 Agent 循环类型 |
| `actor_rollout_ref.rollout.agent.agent_loop_config_path` | `null` | Agent 循环配置文件路径 |
| `actor_rollout_ref.rollout.agent.custom_async_server.path` | `null` | 自定义异步服务路径 |
| `actor_rollout_ref.rollout.agent.custom_async_server.name` | `null` | 自定义异步服务名称 |

#### 1.1.8 Checkpoint Engine 配置

| 参数名 | 默认值 | 说明 |
|--------|--------|------|
| `actor_rollout_ref.rollout.checkpoint_engine.backend` | `naive` | Checkpoint 引擎后端 |
| `actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes` | `2048` | 权重更新桶大小（MB） |

#### 1.1.9 Trace 配置

| 参数名 | 默认值 | 说明 |
|--------|--------|------|
| `actor_rollout_ref.rollout.trace.project_name` | 引用自 `trainer.project_name` | 追踪项目名称 |
| `actor_rollout_ref.rollout.trace.experiment_name` | 引用自 `trainer.experiment_name` | 追踪实验名称 |
| `actor_rollout_ref.rollout.trace.backend` | `null` | 追踪后端 |
| `actor_rollout_ref.rollout.trace.token2text` | `false` | 是否将 token 转为文本 |
| `actor_rollout_ref.rollout.trace.max_samples_per_step_per_worker` | `null` | 每步每 worker 最大样本数 |

#### 1.1.10 Prometheus 配置

| 参数名 | 默认值 | 说明 |
|--------|--------|------|
| `actor_rollout_ref.rollout.prometheus.enable` | `false` | 是否启用 Prometheus 监控 |
| `actor_rollout_ref.rollout.prometheus.port` | `9090` | Prometheus 端口 |
| `actor_rollout_ref.rollout.prometheus.file` | `/tmp/ray/session_latest/metrics/prometheus/prometheus.yml` | Prometheus 配置文件路径 |
| `actor_rollout_ref.rollout.prometheus.served_model_name` | 引用自 `model.path` | 服务模型名称 |

#### 1.1.11 Reference 模型配置

| 参数名 | 默认值 | 说明 |
|--------|--------|------|
| `actor_rollout_ref.ref.rollout_n` | 引用自 `rollout.n` | Rollout 次数 |
| `actor_rollout_ref.ref.strategy` | 引用自 `actor.strategy` | 训练策略 |
| `actor_rollout_ref.ref.use_torch_compile` | 引用自 `actor.use_torch_compile` | 是否使用 torch.compile |
| `actor_rollout_ref.ref.log_prob_micro_batch_size` | `null` | log prob 计算的 micro batch 大小 |
| `actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu` | `null` | 每 GPU 的 log prob micro batch 大小 |
| `actor_rollout_ref.ref.log_prob_use_dynamic_bsz` | 引用自 `actor.use_dynamic_bsz` | log prob 是否使用动态 batch size |
| `actor_rollout_ref.ref.log_prob_max_token_len_per_gpu` | 引用自 `actor.ppo_max_token_len_per_gpu` | 每 GPU 的 log prob 最大 token 长度 |
| `actor_rollout_ref.ref.ulysses_sequence_parallel_size` | 引用自 `actor.ulysses_sequence_parallel_size` | Ulysses 序列并行大小 |
| `actor_rollout_ref.ref.entropy_from_logits_with_chunking` | `false` | 是否使用分块方式从 logits 计算熵 |
| `actor_rollout_ref.ref.entropy_checkpointing` | `false` | 是否对熵计算使用梯度检查点 |

#### 1.1.12 Critic 优化器配置

| 参数名 | 默认值 | 说明 |
|--------|--------|------|
| `critic.optim.lr` | `1.0e-05` | Critic 学习率 |
| `critic.optim.lr_warmup_steps_ratio` | `0.0` | 学习率预热步数比例 |
| `critic.optim.total_training_steps` | `-1` | 总训练步数 |
| `critic.optim.weight_decay` | `0.01` | 权重衰减 |
| `critic.optim.lr_warmup_steps` | `-1` | 学习率预热步数 |
| `critic.optim.betas` | `[0.9, 0.999]` | Adam 优化器动量系数 |
| `critic.optim.clip_grad` | `1.0` | 梯度裁剪阈值 |
| `critic.optim.override_optimizer_config` | `null` / `{}` | 覆盖优化器配置 |

#### 1.1.13 Critic 策略配置

| 参数名 | 默认值 | 说明 |
|--------|--------|------|
| `critic.strategy` | `fsdp` / `megatron` | 训练策略 |
| `critic.enable` | `null` | 是否启用 Critic，null 表示自动决定 |
| `critic.ppo_mini_batch_size` | 引用自 `actor.ppo_mini_batch_size` | PPO mini batch 大小 |
| `critic.ppo_micro_batch_size` | `null` | PPO micro batch 大小 |
| `critic.ppo_micro_batch_size_per_gpu` | `null` | 每 GPU 的 PPO micro batch 大小 |
| `critic.use_dynamic_bsz` | 引用自 `actor.use_dynamic_bsz` | 是否使用动态 batch size |
| `critic.ppo_max_token_len_per_gpu` | `32768` | 每 GPU 的 PPO 最大 token 长度 |
| `critic.forward_max_token_len_per_gpu` | 引用自 `.ppo_max_token_len_per_gpu` | 前向计算每 GPU 最大 token 长度 |
| `critic.ppo_epochs` | 引用自 `actor.ppo_epochs` | PPO 更新轮数 |
| `critic.shuffle` | 引用自 `actor.shuffle` | 是否 shuffle |
| `critic.data_loader_seed` | `42` / 引用自 `actor.data_loader_seed` | 数据加载器随机种子 |
| `critic.cliprange_value` | `0.5` | Critic 值函数裁剪范围，一般取值范围 [0.1, 0.3] |
| `critic.loss_agg_mode` | 引用自 `actor.loss_agg_mode` | 损失聚合模式 |
| `critic.grad_clip` | `1.0` | 梯度裁剪值 |
| `critic.ulysses_sequence_parallel_size` | `1` | Ulysses 序列并行大小 |
| `critic.forward_micro_batch_size` | 引用自 `.ppo_micro_batch_size` | 前向计算 micro batch 大小 |
| `critic.forward_micro_batch_size_per_gpu` | 引用自 `.ppo_micro_batch_size_per_gpu` | 前向计算每 GPU micro batch 大小 |

#### 1.1.14 Critic 模型配置

| 参数名 | 默认值 | 说明 |
|--------|--------|------|
| `critic.model.path` | `~/models/deepseek-llm-7b-chat` | Critic 模型路径 |
| `critic.model.tokenizer_path` | 引用自 `model.path` | 分词器路径 |
| `critic.model.override_config` | `{}` | 覆盖模型配置 |
| `critic.model.external_lib` | 引用自 `model.external_lib` | 外部库路径 |
| `critic.model.trust_remote_code` | 引用自 `model.trust_remote_code` | 是否信任远程代码 |
| `critic.model.use_shm` | `false` | 是否使用共享内存 |
| `critic.model.enable_gradient_checkpointing` | `true` | 是否启用梯度检查点 |
| `critic.model.enable_activation_offload` | `false` | 是否启用激活卸载 |
| `critic.model.use_remove_padding` | `false` / `true` | 是否移除 padding |
| `critic.model.lora_rank` | `0` | LoRA 秩 |
| `critic.model.lora_alpha` | `16` | LoRA alpha |
| `critic.model.target_modules` | `all-linear` | LoRA 目标模块 |
| `critic.model.tiled_mlp.enabled` | `false` | 是否启用分片 MLP |
| `critic.model.tiled_mlp.num_shards` | `4` | MLP 分片数 |

#### 1.1.15 数据配置

| 参数名 | 默认值 | 说明 |
|--------|--------|------|
| `data.tokenizer` | `null` | 分词器路径 |
| `data.use_shm` | `false` | 是否使用共享内存 |
| `data.train_files` | `~/data/rlhf/gsm8k/train.parquet` | 训练数据文件路径 |
| `data.val_files` | `~/data/rlhf/gsm8k/test.parquet` | 验证数据文件路径 |
| `data.train_max_samples` | `-1` | 训练最大样本数，-1 表示不限制 |
| `data.val_max_samples` | `-1` | 验证最大样本数 |
| `data.prompt_key` | `prompt` | 数据中 prompt 的键名 |
| `data.reward_fn_key` | `data_source` | 奖励函数的键名 |
| `data.max_prompt_length` | `512` | 最大 prompt 长度 |
| `data.max_response_length` | `512` | 最大 response 长度 |
| `data.train_batch_size` | `1024` | 训练 batch 大小 |
| `data.val_batch_size` | `null` | 验证 batch 大小 |
| `data.tool_config_path` | 引用自 `rollout.multi_turn.tool_config_path` | 工具配置文件路径 |
| `data.return_raw_input_ids` | `false` | 是否返回原始 input ids |
| `data.return_raw_chat` | `true` | 是否返回原始聊天内容 |
| `data.return_full_prompt` | `false` | 是否返回完整 prompt |
| `data.shuffle` | `true` | 是否 shuffle 训练数据 |
| `data.seed` | `null` | 数据 shuffle 随机种子 |
| `data.dataloader_num_workers` | `8` | 数据加载器工作进程数 |
| `data.image_patch_size` | `14` | 图像 patch 大小 |
| `data.validation_shuffle` | `false` | 验证时是否 shuffle |
| `data.filter_overlong_prompts` | `false` | 是否过滤超长 prompt |
| `data.filter_overlong_prompts_workers` | `1` | 过滤超长 prompt 的工作进程数 |
| `data.truncation` | `error` | 截断策略 |
| `data.image_key` | `images` | 图像数据的键名 |
| `data.video_key` | `videos` | 视频数据的键名 |
| `data.trust_remote_code` | `false` | 是否信任远程代码 |
| `data.return_multi_modal_inputs` | `true` | 是否返回多模态输入 |

#### 1.1.16 奖励配置

| 参数名 | 默认值 | 说明 |
|--------|--------|------|
| `reward.num_workers` | `8` | 奖励计算工作进程数 |
| `reward.custom_reward_function.path` | `null` | 自定义奖励函数路径 |
| `reward.custom_reward_function.name` | `compute_score` | 自定义奖励函数名称 |
| `reward.reward_manager.source` | `register` | 奖励管理器来源 |
| `reward.reward_manager.name` | `naive` | 奖励管理器名称 |
| `reward.reward_model.enable` | `false` | 是否启用奖励模型 |
| `reward.reward_model.enable_resource_pool` | `false` | 是否启用奖励模型资源池 |
| `reward.reward_model.n_gpus_per_node` | `8` | 奖励模型每节点 GPU 数 |
| `reward.reward_model.nnodes` | `0` | 奖励模型节点数 |
| `reward.reward_model.model_path` | `null` | 奖励模型路径 |
| `reward.sandbox_fusion.url` | `null` | Sandbox Fusion URL |
| `reward.sandbox_fusion.max_concurrent` | `64` | Sandbox Fusion 最大并发数 |
| `reward.sandbox_fusion.memory_limit_mb` | `1024` | Sandbox Fusion 内存限制（MB） |

#### 1.1.17 算法配置

| 参数名 | 默认值 | 说明 |
|--------|--------|------|
| `algorithm.gamma` | `1.0` | 折扣因子 |
| `algorithm.lam` | `1.0` | GAE lambda 参数 |
| `algorithm.adv_estimator` | `gae` | 优势估计方法，可选 gae 等 |
| `algorithm.norm_adv_by_std_in_grpo` | `true` | GRPO 中是否按标准差归一化优势 |
| `algorithm.use_kl_in_reward` | `false` | 是否在奖励中使用 KL 惩罚 |
| `algorithm.kl_penalty` | `kl` | KL 惩罚类型 |
| `algorithm.kl_ctrl.type` | `fixed` | KL 控制器类型，可选 fixed、kl_adapter 等 |
| `algorithm.kl_ctrl.kl_coef` | `0.001` | KL 惩罚系数 |
| `algorithm.kl_ctrl.horizon` | `10000` | KL 适配器的 horizon |
| `algorithm.kl_ctrl.target_kl` | `0.1` | 目标 KL 散度 |
| `algorithm.use_pf_ppo` | `false` | 是否使用 PF-PPO |
| `algorithm.pf_ppo.reweight_method` | `pow` | PF-PPO 重加权方法 |
| `algorithm.pf_ppo.weight_pow` | `2.0` | PF-PPO 加权幂次 |

#### 1.1.18 Rollout Correction 配置

| 参数名 | 默认值 | 说明 |
|--------|--------|------|
| `algorithm.rollout_correction.rollout_is` | `null` | 是否启用 IS 重要性采样校正 |
| `algorithm.rollout_correction.rollout_is_threshold` | `2.0` | IS 权重阈值 |
| `algorithm.rollout_correction.rollout_rs` | `null` | 是否启用拒绝采样校正 |
| `algorithm.rollout_correction.rollout_rs_threshold` | `null` | RS 阈值 |
| `algorithm.rollout_correction.bypass_mode` | `false` | 是否启用旁路模式 |
| `algorithm.rollout_correction.loss_type` | `ppo_clip` | 校正损失类型 |
| `algorithm.rollout_correction.rollout_is_batch_normalize` | `false` | IS 权重是否批量归一化 |

#### 1.1.19 训练器配置

| 参数名 | 默认值 | 说明 |
|--------|--------|------|
| `trainer.balance_batch` | `true` | 是否平衡 batch |
| `trainer.total_epochs` | `30` | 总训练 epoch 数 |
| `trainer.total_training_steps` | `null` | 总训练步数，null 表示由 epoch 自动计算 |
| `trainer.project_name` | `verl_examples` | 项目名称 |
| `trainer.experiment_name` | `gsm8k` | 实验名称 |
| `trainer.logger` | `[console, wandb]` | 日志后端列表 |
| `trainer.log_val_generations` | `0` | 验证生成日志数量 |
| `trainer.nnodes` | `1` | 训练节点数 |
| `trainer.n_gpus_per_node` | `8` | 每节点 GPU 数 |
| `trainer.save_freq` | `-1` | 保存频率，-1 表示不保存 |
| `trainer.esi_redundant_time` | `0` | ESI 冗余时间 |
| `trainer.resume_mode` | `auto` | 恢复模式，可选 auto 等 |
| `trainer.resume_from_path` | `null` | 恢复路径 |
| `trainer.val_before_train` | `true` | 训练前是否先验证 |
| `trainer.val_only` | `false` | 是否仅验证 |
| `trainer.test_freq` | `-1` | 测试频率 |
| `trainer.critic_warmup` | `0` | Critic 预热步数 |
| `trainer.default_hdfs_dir` | `null` | 默认 HDFS 目录 |
| `trainer.del_local_ckpt_after_load` | `false` | 加载后是否删除本地 checkpoint |
| `trainer.default_local_dir` | `checkpoints/${trainer.project_name}/${trainer.experiment_name}` | 默认本地 checkpoint 目录 |
| `trainer.max_actor_ckpt_to_keep` | `null` | 最多保留的 Actor checkpoint 数 |
| `trainer.max_critic_ckpt_to_keep` | `null` | 最多保留的 Critic checkpoint 数 |
| `trainer.ray_wait_register_center_timeout` | `300` | Ray 注册中心等待超时（秒） |
| `trainer.device` | `cuda` | 训练设备 |
| `trainer.use_legacy_worker_impl` | `auto` | 是否使用旧版 worker 实现 |
| `trainer.rollout_data_dir` | `null` | 保存每轮 rollout 结果的地址配置 |

#### 1.1.20 模型配置

| 参数名 | 默认值 | 说明 |
|--------|--------|------|
| `actor_rollout_ref.model.path` | `~/models/deepseek-llm-7b-chat` | 模型路径 |
| `actor_rollout_ref.model.hf_config_path` | `null` | HuggingFace 配置路径 |
| `actor_rollout_ref.model.tokenizer_path` | `null` | 分词器路径 |
| `actor_rollout_ref.model.use_shm` | `false` | 是否使用共享内存 |
| `actor_rollout_ref.model.trust_remote_code` | `false` | 是否信任远程代码 |
| `actor_rollout_ref.model.custom_chat_template` | `null` | 自定义聊天模板 |
| `actor_rollout_ref.model.external_lib` | `null` | 外部库路径 |
| `actor_rollout_ref.model.override_config` | `{}` | 覆盖模型配置 |
| `actor_rollout_ref.model.enable_gradient_checkpointing` | `true` | 是否启用梯度检查点 |
| `actor_rollout_ref.model.enable_activation_offload` | `false` | 是否启用激活卸载 |
| `actor_rollout_ref.model.use_remove_padding` | `true` / `false` | 是否移除 padding |
| `actor_rollout_ref.model.lora_rank` | `0` | LoRA 秩，0 表示不使用 LoRA |
| `actor_rollout_ref.model.lora_alpha` | `16` | LoRA alpha |
| `actor_rollout_ref.model.target_modules` | `all-linear` | LoRA 目标模块 |
| `actor_rollout_ref.model.exclude_modules` | `null` | LoRA 排除模块 |
| `actor_rollout_ref.model.lora_adapter_path` | `null` | LoRA 适配器路径 |
| `actor_rollout_ref.model.use_liger` | `false` | 是否使用 Liger 内核 |
| `actor_rollout_ref.model.use_fused_kernels` | `false` | 是否使用融合内核 |
| `actor_rollout_ref.model.fused_kernel_options.impl_backend` | `torch` | 融合内核实现后端 |
| `actor_rollout_ref.model.tiled_mlp.enabled` | `false` | 是否启用分片 MLP |
| `actor_rollout_ref.model.tiled_mlp.num_shards` | `4` | MLP 分片数 |

#### 1.1.21 公共引擎配置

| 参数名 | 默认值 | 说明 |
|--------|--------|------|
| `actor_rollout_ref.hybrid_engine` | `true` | 是否使用混合引擎（训练推理共享权重） |
| `actor_rollout_ref.nccl_timeout` | `600` | NCCL 通信超时（秒） |
| `transfer_queue.enable` | `false` | 是否启用传输队列 |

---

### 1.2 FSDP 专属配置参数

以下参数仅在 FSDP 方案（`_generated_ppo_trainer.yaml`）中存在。

#### 1.2.1 FSDP 优化器配置

| 参数名 | 默认值 | 说明 |
|--------|--------|------|
| `actor_rollout_ref.actor.optim.optimizer` | `AdamW` | 优化器类型 |
| `actor_rollout_ref.actor.optim.optimizer_impl` | `torch.optim` | 优化器实现 |
| `actor_rollout_ref.actor.optim.min_lr_ratio` | `0.0` | 最小学习率比例 |
| `actor_rollout_ref.actor.optim.num_cycles` | `0.5` | 余弦调度周期数 |
| `actor_rollout_ref.actor.optim.lr_scheduler_type` | `constant` | 学习率调度器类型 |
| `actor_rollout_ref.actor.optim.zero_indexed_step` | `true` | 步数是否从 0 开始计数 |
| `actor_rollout_ref.actor.optim.warmup_style` | `null` | 预热风格 |

#### 1.2.2 Actor FSDP 引擎配置

| 参数名 | 默认值 | 说明 |
|--------|--------|------|
| `actor_rollout_ref.actor.fsdp_config.wrap_policy.min_num_params` | `0` | FSDP 包装的最小参数数 |
| `actor_rollout_ref.actor.fsdp_config.param_offload` | `false` | 是否将参数卸载到 CPU |
| `actor_rollout_ref.actor.fsdp_config.optimizer_offload` | `false` | 是否将优化器状态卸载到 CPU |
| `actor_rollout_ref.actor.fsdp_config.offload_policy` | `false` | 卸载策略 |
| `actor_rollout_ref.actor.fsdp_config.reshard_after_forward` | `true` | 前向计算后是否重新分片 |
| `actor_rollout_ref.actor.fsdp_config.fsdp_size` | `-1` | FSDP 组大小，-1 表示全局 |
| `actor_rollout_ref.actor.fsdp_config.forward_prefetch` | `false` | 是否预取前向参数 |
| `actor_rollout_ref.actor.fsdp_config.model_dtype` | `fp32` | 模型计算数据类型 |
| `actor_rollout_ref.actor.fsdp_config.use_orig_params` | `false` | 是否使用原始参数 |
| `actor_rollout_ref.actor.fsdp_config.seed` | `42` | 随机种子 |
| `actor_rollout_ref.actor.fsdp_config.full_determinism` | `false` | 是否启用完全确定性 |
| `actor_rollout_ref.actor.fsdp_config.forward_only` | `false` | 是否仅前向计算（Actor 为 false） |
| `actor_rollout_ref.actor.fsdp_config.strategy` | `fsdp` | 策略类型 |
| `actor_rollout_ref.actor.fsdp_config.dtype` | `bfloat16` | 模型存储数据类型 |

#### 1.2.3 Reference FSDP 引擎配置

与 Actor FSDP 引擎配置结构相同，主要区别：

| 参数名 | 默认值 | 说明 |
|--------|--------|------|
| `actor_rollout_ref.ref.fsdp_config.forward_only` | `true` | Reference 模型仅前向计算 |

其余参数（`wrap_policy`、`param_offload`、`optimizer_offload`、`reshard_after_forward`、`fsdp_size`、`dtype` 等）默认值与 Actor FSDP 引擎配置一致。

#### 1.2.4 Critic FSDP 引擎配置

与 Actor FSDP 引擎配置结构相同，主要区别：

| 参数名 | 默认值 | 说明 |
|--------|--------|------|
| `critic.model.fsdp_config.forward_only` | `false` | Critic 模型需要训练 |
| `critic.model.fsdp_config.use_remove_padding` | `false` | Critic 不移除 padding |

其余参数默认值与 Actor FSDP 引擎配置一致。

---

### 1.3 Megatron 专属配置参数

以下参数仅在 Megatron 方案（`_generated_ppo_megatron_trainer.yaml`）中存在。

#### 1.3.1 Megatron 优化器配置

| 参数名 | 默认值 | 说明 |
|--------|--------|------|
| `actor_rollout_ref.actor.optim.optimizer` | `adam` | 优化器类型 |
| `actor_rollout_ref.actor.optim.lr_warmup_init` | `0.0` | 学习率预热初始值 |
| `actor_rollout_ref.actor.optim.lr_decay_steps` | `null` | 学习率衰减步数 |
| `actor_rollout_ref.actor.optim.lr_decay_style` | `constant` | 学习率衰减风格，可选 constant、cosine、exponential 等 |
| `actor_rollout_ref.actor.optim.min_lr` | `0.0` | 最小学习率 |
| `actor_rollout_ref.actor.optim.weight_decay_incr_style` | `constant` | 权重衰减增长风格 |
| `actor_rollout_ref.actor.optim.lr_wsd_decay_style` | `exponential` | WSD 学习率衰减风格 |
| `actor_rollout_ref.actor.optim.lr_wsd_decay_steps` | `null` | WSD 学习率衰减步数 |
| `actor_rollout_ref.actor.optim.use_checkpoint_opt_param_scheduler` | `false` | 是否使用 checkpoint 优化器参数调度器 |

#### 1.3.2 Actor Megatron 引擎配置

| 参数名 | 默认值 | 说明 |
|--------|--------|------|
| `actor_rollout_ref.actor.megatron.param_offload` | `false` | 是否将参数卸载到 CPU |
| `actor_rollout_ref.actor.megatron.grad_offload` | `false` | 是否将梯度卸载到 CPU |
| `actor_rollout_ref.actor.megatron.optimizer_offload` | `false` | 是否将优化器状态卸载到 CPU |
| `actor_rollout_ref.actor.megatron.tensor_model_parallel_size` | `1` | TP 并行大小 |
| `actor_rollout_ref.actor.megatron.expert_model_parallel_size` | `1` | 专家并行大小 |
| `actor_rollout_ref.actor.megatron.expert_tensor_parallel_size` | `null` | 专家 TP 并行大小 |
| `actor_rollout_ref.actor.megatron.pipeline_model_parallel_size` | `1` | PP 并行大小 |
| `actor_rollout_ref.actor.megatron.virtual_pipeline_model_parallel_size` | `null` | 虚拟 PP 并行大小 |
| `actor_rollout_ref.actor.megatron.context_parallel_size` | `1` | 上下文并行大小 |
| `actor_rollout_ref.actor.megatron.sequence_parallel` | `true` | 是否启用序列并行 |
| `actor_rollout_ref.actor.megatron.use_distributed_optimizer` | `true` | 是否使用分布式优化器 |
| `actor_rollout_ref.actor.megatron.use_dist_checkpointing` | `false` | 是否使用分布式 checkpoint |
| `actor_rollout_ref.actor.megatron.dist_checkpointing_path` | `null` | 分布式 checkpoint 路径 |
| `actor_rollout_ref.actor.megatron.dist_checkpointing_prefix` | `''` | 分布式 checkpoint 前缀 |
| `actor_rollout_ref.actor.megatron.dist_ckpt_optim_fully_reshardable` | `false` | 分布式 checkpoint 优化器是否完全可重分片 |
| `actor_rollout_ref.actor.megatron.distrib_optim_fully_reshardable_mem_efficient` | `false` | 分布式优化器重分片是否内存高效 |
| `actor_rollout_ref.actor.megatron.seed` | `42` | 随机种子 |
| `actor_rollout_ref.actor.megatron.use_mbridge` | `true` | 是否使用 mBridge |
| `actor_rollout_ref.actor.megatron.vanilla_mbridge` | `true` | 是否使用原始 mBridge |
| `actor_rollout_ref.actor.megatron.use_remove_padding` | `true` | 是否移除 padding |
| `actor_rollout_ref.actor.megatron.forward_only` | `false` | 是否仅前向计算 |
| `actor_rollout_ref.actor.megatron.dtype` | `bfloat16` | 模型数据类型 |
| `actor_rollout_ref.actor.megatron.load_weight` | `true` | 是否加载权重 |

#### 1.3.3 Megatron Transformer 覆盖配置

| 参数名 | 默认值 | 说明 |
|--------|--------|------|
| `override_transformer_config.recompute_granularity` | `null` | 重计算粒度 |
| `override_transformer_config.recompute_modules` | `[core_attn]` | 重计算模块列表 |
| `override_transformer_config.recompute_method` | `null` | 重计算方法 |
| `override_transformer_config.recompute_num_layers` | `null` | 重计算层数 |
| `override_transformer_config.attention_backend` | `flash` | 注意力后端 |

#### 1.3.4 Reference Megatron 引擎配置

与 Actor Megatron 引擎配置结构相同，主要区别：

| 参数名 | 默认值 | 说明 |
|--------|--------|------|
| `actor_rollout_ref.ref.megatron.forward_only` | `true` | Reference 模型仅前向计算 |

其余参数默认值引用自 Actor Megatron 引擎配置（如 `param_offload`、`tensor_model_parallel_size` 等）。

#### 1.3.5 Critic Megatron 引擎配置

与 Actor Megatron 引擎配置结构相同，主要区别：

| 参数名 | 默认值 | 说明 |
|--------|--------|------|
| `critic.megatron.forward_only` | `false` | Critic 模型需要训练 |

#### 1.3.6 Megatron LoRA 配置

| 参数名 | 默认值 | 说明 |
|--------|--------|------|
| `model.lora.type` | `lora` | LoRA 类型 |
| `model.lora.merge` | `false` | 是否合并 LoRA 权重 |
| `model.lora.rank` | `0` | LoRA 秩，0 表示不使用 |
| `model.lora.alpha` | `32` | LoRA alpha |
| `model.lora.dropout` | `0.0` | LoRA dropout |
| `model.lora.target_modules` | `[linear_qkv, linear_proj, linear_fc1, linear_fc2]` | LoRA 目标模块 |
| `model.lora.exclude_modules` | `[]` | LoRA 排除模块 |
| `model.lora.dropout_position` | `pre` | LoRA dropout 位置 |
| `model.lora.lora_A_init_method` | `xavier` | LoRA A 矩阵初始化方法 |
| `model.lora.lora_B_init_method` | `zero` | LoRA B 矩阵初始化方法 |
| `model.lora.a2a_experimental` | `false` | 是否启用 a2a 实验性功能 |
| `model.lora.dtype` | `null` | LoRA 数据类型 |
| `model.lora.adapter_path` | `null` | LoRA 适配器路径 |
| `model.lora.freeze_vision_model` | `true` | 是否冻结视觉模型 |
| `model.lora.freeze_vision_projection` | `true` | 是否冻结视觉投影 |
| `model.lora.freeze_language_model` | `true` | 是否冻结语言模型 |

#### 1.3.7 模型 override_config（Megatron 方案）

| 参数名 | 默认值 | 说明 |
|--------|--------|------|
| `model.override_config.model_config` | `{}` | 模型配置覆盖 |
| `model.override_config.moe_config.freeze_moe_router` | `false` | 是否冻结 MoE 路由 |

#### 1.3.8 Rollout layer_name_map（Megatron 方案）

| 参数名 | 默认值 | 说明 |
|--------|--------|------|
| `rollout.layer_name_map.qkv_layer_name` | `qkv` | QKV 层名称映射 |
| `rollout.layer_name_map.gate_proj_layer_name` | `gate_up` | Gate 投影层名称映射 |

---

### 1.4 高级配置参数

#### 1.4.1 Profiler 配置

| 参数名 | 默认值 | 说明 |
|--------|--------|------|
| `profiler.enable` | `false` | 是否启用 Profiler |
| `profiler.tool` | 引用自 `global_profiler.tool` | Profiler 工具，可选 nsys、npu、torch、torch_memory |
| `profiler.all_ranks` | `false` | 是否在所有 rank 上启用 |
| `profiler.ranks` | `[]` | 指定启用的 rank 列表 |
| `profiler.save_path` | 引用自 `global_profiler.save_path` | Profiler 结果保存路径 |

#### 1.4.2 Global Profiler 配置

| 参数名 | 默认值 | 说明 |
|--------|--------|------|
| `global_profiler.tool` | `null` | 全局 Profiler 工具 |
| `global_profiler.steps` | `null` | Profiler 采集步数 |
| `global_profiler.profile_continuous_steps` | `false` | 是否连续步采集 |
| `global_profiler.save_path` | `outputs/profile` | 全局保存路径 |

#### 1.4.3 Router Replay 配置

| 参数名 | 默认值 | 说明 |
|--------|--------|------|
| `router_replay.mode` | `disabled` | 路由重放模式，可选 disabled、record、replay |
| `router_replay.record_file` | `null` | 路由记录文件路径 |
| `router_replay.replay_file` | `null` | 路由重放文件路径 |

#### 1.4.4 Checkpoint 配置

| 参数名 | 默认值 | 说明 |
|--------|--------|------|
| `checkpoint.save_contents` | `[model, optimizer, extra]` | Checkpoint 保存内容 |
| `checkpoint.load_contents` | 引用自 `.save_contents` | Checkpoint 加载内容 |
| `checkpoint.async_save` | `false` | 是否异步保存 Checkpoint |
| `checkpoint.mbridge_config` | `{}` | mBridge 配置 |

#### 1.4.5 QAT 配置

| 参数名 | 默认值 | 说明 |
|--------|--------|------|
| `qat.enable` | `false` | 是否启用量化感知训练（QAT） |
| `qat.mode` | `w4a16` | 量化模式 |
| `qat.group_size` | `16` | 量化分组大小 |
| `qat.ignore_patterns` | `[lm_head, embed_tokens, re:.*mlp.gate$]` | 量化忽略的模式列表 |
| `qat.activation_observer` | `static_minmax` | 激活值观察器类型 |
| `qat.quantization_config_path` | `null` | 量化配置文件路径 |

#### 1.4.6 MTP 配置

| 参数名 | 默认值 | 说明 |
|--------|--------|------|
| `mtp.enable` | `false` | 是否启用多 token 预测（MTP） |
| `mtp.enable_train` | `false` | 是否在训练中启用 MTP |
| `mtp.enable_rollout` | `false` | 是否在推理中启用 MTP |
| `mtp.detach_encoder` | `false` | 是否分离编码器 |
| `mtp.mtp_loss_scaling_factor` | `0.1` | MTP 损失缩放因子 |
| `mtp.speculative_algorithm` | `EAGLE` | 推测解码算法 |
| `mtp.speculative_num_steps` | `3` | 推测步数 |
| `mtp.speculative_eagle_topk` | `1` | EAGLE Top-K |
| `mtp.speculative_num_draft_tokens` | `4` | 推测 draft token 数 |
| `mtp.method` | `mtp` | MTP 方法 |
| `mtp.num_speculative_tokens` | `1` | 推测 token 数 |

---

## 2. 训练指标说明

强化学习算法每个 iteration 打印的日志指标说明如下：

### 2.1 训练基础指标

| 指标 | 说明 |
|------|------|
| `training/global_step` | 当前全局训练步数 |
| `training/epoch` | 当前训练 epoch |

### 2.2 Actor 模型指标

| 指标 | 说明 |
|------|------|
| `actor/pg_loss` | 策略梯度损失（PPO clip loss），基于优势函数的策略梯度目标函数值 |
| `actor/kl_loss` | KL 散度损失，衡量当前策略与参考策略之间的偏离程度（仅 `use_kl_loss=True` 时打印） |
| `actor/entropy` | 策略熵，表示策略的随机性或探索能力（仅 `calculate_entropy=True` 或 `entropy_coeff!=0` 时打印） |
| `actor/grad_norm` | Actor 梯度范数（裁剪后），表示反向传播中参数梯度的整体幅度 |
| `actor/lr` | Actor 当前学习率 |
| `actor/pg_clipfrac` | PPO 裁剪机制生效的比例，反映策略更新幅度的稳定性 |
| `actor/ppo_kl` | PPO 算法的实际 KL 散度（当前策略 vs 旧策略） |
| `actor/pg_clipfrac_lower` | PPO 下界裁剪比例（部分 `loss_mode` 有此指标） |
| `actor/reward_kl_penalty` | KL 惩罚值，当前策略与参考策略的 KL 均值（仅 `use_kl_in_reward=True` 时打印） |
| `actor/reward_kl_penalty_coeff` | KL 惩罚系数 beta（仅 `use_kl_in_reward=True` 时打印） |
| `actor/kl_coef` | KL 损失系数（仅 `use_kl_loss=True` 时打印） |

### 2.3 Critic 模型指标

| 指标 | 说明 |
|------|------|
| `critic/vf_loss` | 值函数损失 |
| `critic/vf_clipfrac` | Critic 裁剪机制生效的比例，反映值函数更新幅度的稳定性 |
| `critic/vpred_mean` | 预测值的均值 |
| `critic/grad_norm` | Critic 梯度范数（裁剪后） |
| `critic/lr` | Critic 当前学习率 |
| `critic/vf_explained_var` | 值函数解释方差 1 - Var(returns-values)/Var(returns)（仅 `use_critic=True` 时打印） |

### 2.4 数据统计指标

| 指标 | 说明 |
|------|------|
| `critic/score/mean` | 非中止样本的序列分数均值 |
| `critic/score/max` | 非中止样本的序列分数最大值 |
| `critic/score/min` | 非中止样本的序列分数最小值 |
| `critic/rewards/mean` | 非中止样本的序列奖励均值 |
| `critic/rewards/max` | 非中止样本的序列奖励最大值 |
| `critic/rewards/min` | 非中止样本的序列奖励最小值 |
| `critic/advantages/mean` | 有效 token 的优势值均值 |
| `critic/advantages/max` | 有效 token 的优势值最大值 |
| `critic/advantages/min` | 有效 token 的优势值最小值 |
| `critic/returns/mean` | 有效 token 的回报均值 |
| `critic/returns/max` | 有效 token 的回报最大值 |
| `critic/returns/min` | 有效 token 的回报最小值 |
| `critic/values/mean` | 有效 token 的 Critic 值均值（仅 `use_critic=True` 时打印） |
| `critic/values/max` | 有效 token 的 Critic 值最大值（仅 `use_critic=True` 时打印） |
| `critic/values/min` | 有效 token 的 Critic 值最小值（仅 `use_critic=True` 时打印） |
| `response_length/mean` | 响应长度均值（含中止样本） |
| `response_length/max` | 响应长度最大值 |
| `response_length/min` | 响应长度最小值 |
| `response_length/clip_ratio` | 响应长度达到最大长度的比例 |
| `response_length_non_aborted/mean` | 非中止样本的响应长度均值 |
| `response_length_non_aborted/max` | 非中止样本的响应长度最大值 |
| `response_length_non_aborted/min` | 非中止样本的响应长度最小值 |
| `response_length_non_aborted/clip_ratio` | 非中止样本的响应长度达到最大长度的比例 |
| `response/aborted_ratio` | 中止样本（响应长度为 0）的比例 |
| `prompt_length/mean` | 提示长度均值 |
| `prompt_length/max` | 提示长度最大值 |
| `prompt_length/min` | 提示长度最小值 |
| `prompt_length/clip_ratio` | 提示长度达到最大长度的比例 |
| `num_turns/mean` | 多轮对话轮数均值（仅多轮对话时打印） |
| `num_turns/max` | 多轮对话轮数最大值（仅多轮对话时打印） |
| `num_turns/min` | 多轮对话轮数最小值（仅多轮对话时打印） |
| `tool_call_counts/mean` | 工具调用次数均值（仅存在 `tool_call_counts` 时打印） |
| `tool_call_counts/max` | 工具调用次数最大值 |
| `tool_call_counts/min` | 工具调用次数最小值 |

### 2.5 时间指标

| 指标 | 说明 |
|------|------|
| `timing_s/gen` | 生成（rollout）耗时（秒） |
| `timing_s/ref` | Reference 模型计算 log_p 耗时（秒） |
| `timing_s/values` | Critic 模型计算 values 耗时（秒） |
| `timing_s/adv` | 计算优势值耗时（秒） |
| `timing_s/update_critic` | Critic 模型更新耗时（秒） |
| `timing_s/update_actor` | Actor 模型更新耗时（秒） |
| `timing_s/step` | 一步总耗时（秒） |
| `timing_s/old_log_prob` | Actor 模型计算旧 log_p 耗时（秒） |
| `timing_s/reward` | 奖励计算耗时（秒） |
| `timing_s/testing` | 验证耗时（秒） |
| `timing_s/save_checkpoint` | 保存 checkpoint 耗时（秒） |
| `timing_s/update_weights` | 权重同步耗时（秒） |
| `timing_per_token_ms/gen` | 生成阶段每 token 耗时（毫秒） |
| `timing_per_token_ms/ref` | Reference 模型每 token 耗时（毫秒） |
| `timing_per_token_ms/values` | Critic 模型每 token 耗时（毫秒） |
| `timing_per_token_ms/adv` | 优势值计算每 token 耗时（毫秒） |
| `timing_per_token_ms/update_critic` | Critic 更新每 token 耗时（毫秒） |
| `timing_per_token_ms/update_actor` | Actor 更新每 token 耗时（毫秒） |

### 2.6 性能指标

| 指标 | 说明 |
|------|------|
| `perf/total_num_tokens` | 本步处理的总 token 数 |
| `perf/time_per_step` | 本步总耗时（秒） |
| `perf/throughput` | 吞吐量：tokens / (time * n_gpus) |
| `perf/max_memory_allocated_gb` | GPU 最大已分配内存（GB） |
| `perf/max_memory_reserved_gb` | GPU 最大预留内存（GB） |
| `perf/cpu_memory_used_gb` | CPU 已使用内存（GB） |
| `perf/mfu/actor` | Actor 训练的 MFU（模型浮点利用率） |
| `perf/mfu/critic` | Critic 训练的 MFU |
| `perf/mfu/actor_infer` | Actor 推理阶段的 MFU |

### 2.7 方差代理指标

| 指标 | 说明 |
|------|------|
| `variance_proxy/proxy1_signal_strength` | 信号强度：梯度均值的平方范数 \|\|g_mean\|\|^2 |
| `variance_proxy/proxy2_total_power` | 总功率：梯度平方范数的期望 E[\|\|g_tau\|\|^2] |
| `variance_proxy/proxy3_pure_noise` | 纯噪声：梯度方差代理 (1/(N-1)) * (Proxy2 - Proxy1) |
| `variance_proxy/expected_a_squared` | 优势平方的期望 E[A^2] |
| `variance_proxy/expected_w` | W-score 代理的期望 E[W] |

### 2.8 条件性指标

以下指标仅在特定条件满足时打印：

#### 2.8.1 Rollout Correction 指标

仅启用 `rollout_correction` 时打印，均带 `rollout_corr/` 前缀。

**IS 权重指标**（仅启用 IS 校正时）：

| 指标 | 说明 |
|------|------|
| `rollout_corr/rollout_is_mean` | IS 权重均值 |
| `rollout_corr/rollout_is_max` | IS 权重最大值 |
| `rollout_corr/rollout_is_min` | IS 权重最小值 |
| `rollout_corr/rollout_is_std` | IS 权重标准差 |
| `rollout_corr/rollout_is_ratio_fraction_high` | 超过上限阈值的 IS 权重比例 |
| `rollout_corr/rollout_is_ratio_fraction_low` | 低于下限阈值的 IS 权重比例 |
| `rollout_corr/rollout_is_eff_sample_size` | 有效样本大小（ESS） |
| `rollout_corr/rollout_is_seq_mean` | 序列级 IS 权重均值 |
| `rollout_corr/rollout_is_seq_std` | 序列级 IS 权重标准差 |
| `rollout_corr/rollout_is_seq_max` | 序列级 IS 权重最大值 |
| `rollout_corr/rollout_is_seq_min` | 序列级 IS 权重最小值 |
| `rollout_corr/rollout_is_seq_max_deviation` | 序列级 IS 权重与理想值 1.0 的最大偏差 |
| `rollout_corr/rollout_is_seq_fraction_high` | 序列级 IS 权重超过上限的比例 |
| `rollout_corr/rollout_is_seq_fraction_low` | 序列级 IS 权重低于下限的比例 |
| `rollout_corr/rollout_is_batch_norm_factor` | IS 权重批量归一化因子（仅 `rollout_is_batch_normalize=True` 时打印） |

**Rejection Sampling 指标**（仅启用 RS 校正时）：

| 指标 | 说明 |
|------|------|
| `rollout_corr/rollout_rs_{option}_mean` | RS 统计量均值 |
| `rollout_corr/rollout_rs_{option}_max` | RS 统计量最大值 |
| `rollout_corr/rollout_rs_{option}_min` | RS 统计量最小值 |
| `rollout_corr/rollout_rs_{option}_std` | RS 统计量标准差 |
| `rollout_corr/rollout_rs_{option}_fraction_high` | 超过上限阈值的比例 |
| `rollout_corr/rollout_rs_{option}_fraction_low` | 低于下限阈值的比例 |
| `rollout_corr/rollout_rs_{option}_seq_mean` | 序列级 RS 统计量均值 |
| `rollout_corr/rollout_rs_{option}_seq_std` | 序列级 RS 统计量标准差 |
| `rollout_corr/rollout_rs_{option}_seq_max` | 序列级 RS 统计量最大值 |
| `rollout_corr/rollout_rs_{option}_seq_min` | 序列级 RS 统计量最小值 |
| `rollout_corr/rollout_rs_{option}_seq_max_deviation` | 序列级 RS 统计量与 0 的最大偏差 |
| `rollout_corr/rollout_rs_{option}_seq_fraction_high` | 序列级超过上限的比例 |
| `rollout_corr/rollout_rs_{option}_seq_fraction_low` | 序列级低于下限的比例 |
| `rollout_corr/rollout_rs_{option}_masked_fraction` | token 级被 mask 掉的比例 |
| `rollout_corr/rollout_rs_{option}_seq_masked_fraction` | 序列级被 mask 掉的比例 |
| `rollout_corr/rollout_rs_masked_fraction` | 总体 token 级被 mask 掉的比例 |
| `rollout_corr/rollout_rs_seq_masked_fraction` | 总体序列级被 mask 掉的比例 |

**Off-policy 诊断指标**（仅启用 off-policy 诊断时）：

| 指标 | 说明 |
|------|------|
| `rollout_corr/training_ppl` | 训练策略的困惑度 |
| `rollout_corr/training_log_ppl` | 训练策略的 log 困惑度 |
| `rollout_corr/kl` | KL(π_rollout \|\| π_training) 直接估计 |
| `rollout_corr/k3_kl` | K3 KL 估计（更稳定） |
| `rollout_corr/rollout_ppl` | Rollout 策略的困惑度 |
| `rollout_corr/rollout_log_ppl` | Rollout 策略的 log 困惑度 |
| `rollout_corr/log_ppl_diff` | log PPL 差值（rollout - training） |
| `rollout_corr/log_ppl_abs_diff` | log PPL 绝对差值的均值 |
| `rollout_corr/log_ppl_diff_max` | log PPL 差值最大值 |
| `rollout_corr/log_ppl_diff_min` | log PPL 差值最小值 |
| `rollout_corr/ppl_ratio` | PPL 比率（training_ppl / rollout_ppl） |
| `rollout_corr/chi2_token` | token 级卡方散度 |
| `rollout_corr/chi2_seq` | 序列级卡方散度 |

#### 2.8.2 序列长度平衡指标

仅启用 `balance_batch` 时打印：

| 指标 | 说明 |
|------|------|
| `global_seqlen/min` | 平衡前各 DP 分区的最小序列长度和 |
| `global_seqlen/max` | 平衡前各 DP 分区的最大序列长度和 |
| `global_seqlen/minmax_diff` | 平衡前 max - min 差值 |
| `global_seqlen/balanced_min` | 平衡后各 DP 分区的最小序列长度和 |
| `global_seqlen/balanced_max` | 平衡后各 DP 分区的最大序列长度和 |
| `global_seqlen/mean` | 各分区的平均序列长度和 |

#### 2.8.3 GDPO 奖励指标

仅使用 GDPO 估计器时打印：

| 指标 | 说明 |
|------|------|
| `gdpo/{key}/mean` | GDPO 各奖励分量的均值 |
| `gdpo/{key}/std` | GDPO 各奖励分量的标准差 |
| `gdpo/{key}/max` | GDPO 各奖励分量的最大值 |
| `gdpo/{key}/min` | GDPO 各奖励分量的最小值 |

#### 2.8.4 训推一致性指标

仅存在 `actor_rollout_ref.rollout.calculate_log_probs=True` 时打印：

| 指标 | 说明 |
|------|------|
| `training/rollout_probs_diff_valid` | 标记为 1（有效） |
| `training/rollout_probs_diff_max` | rollout 与 actor 概率差异的最大值 |
| `training/rollout_probs_diff_mean` | rollout 与 actor 概率差异的均值 |
| `training/rollout_probs_diff_std` | rollout 与 actor 概率差异的标准差 |
| `training/rollout_actor_probs_pearson_corr` | rollout 与 actor 概率的 Pearson 相关系数 |

#### 2.8.5 验证指标

验证阶段打印：

| 指标 | 说明 |
|------|------|
| `val-core/{data_source}/{var_name}/{metric_name}` | 核心验证指标（mean@N, maj@N, best@N 等） |
| `val-aux/{data_source}/{var_name}/{metric_name}` | 辅助验证指标（std@N, worst@N 等） |
| `val-aux/num_turns/mean` | 验证集多轮对话轮数均值 |
| `val-aux/num_turns/max` | 验证集多轮对话轮数最大值 |
| `val-aux/num_turns/min` | 验证集多轮对话轮数最小值 |
