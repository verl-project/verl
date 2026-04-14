# SFT 和 PPO 训练脚本参数说明

## SFT (Supervised Fine-Tuning) 参数

### 数据配置 (data.*)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `data.train_files` | str | - | 训练数据文件路径（parquet格式），支持本地路径或HDFS路径 |
| `data.val_files` | str | - | 验证数据文件路径（parquet格式） |
| `data.prompt_key` | str | `question` | 数据中prompt字段的键名 |
| `data.response_key` | str | `answer` | 数据中response字段的键名 |
| `data.prompt_dict_keys` | list | null | 多层嵌套时，prompt所在的键路径（如 `['question']`） |
| `data.response_dict_keys` | list | null | 多层嵌套时，response所在的键路径（如 `['answer']`） |
| `data.micro_batch_size_per_gpu` | int | 4 | 每个GPU的微批次大小 |
| `data.max_length` | int | 1024 | 最大序列长度 |
| `data.truncation` | str | `error` | 超长处理方式：`error`（报错）、`left`（左截断）、`right`（右截断） |
| `data.multiturn.enable` | bool | false | 是否启用多轮对话模式 |
| `data.multiturn.messages_key` | str | `messages` | 多轮对话中消息列表的键名 |

### 模型配置 (model.*)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model.partial_pretrain` | str | - | 预训练模型路径（本地路径或HDFS路径） |
| `model.trust_remote_code` | bool | false | 是否信任模型中的远程代码（某些模型需要） |
| `model.lora_rank` | int | 0 | LoRA秩，设为正数启用LoRA微调（如8、16、32） |
| `model.lora_alpha` | int | 16 | LoRA缩放因子 |
| `model.enable_gradient_checkpointing` | bool | true | 是否启用梯度检查点（节省显存但速度慢） |
| `model.use_liger` | bool | false | 是否使用Liger线性层融合加速 |

### 优化器配置 (optim.*)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `optim.lr` | float | 1e-5 | 学习率 |
| `optim.betas` | list | [0.9, 0.95] | Adam优化器的beta参数 |
| `optim.weight_decay` | float | 0.01 | 权重衰减 |
| `optim.clip_grad` | float | 1.0 | 梯度裁剪阈值 |
| `optim.lr_scheduler` | str | `cosine` | 学习率调度器：`cosine`、`linear`等 |
| `optim.warmup_steps_ratio` | float | 0.1 | 预热步数占总步数的比例 |

### 训练器配置 (trainer.*)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `trainer.default_local_dir` | str | - | 本地检查点保存目录 |
| `trainer.project_name` | str | `gsm8k-sft` | 项目名称（用于日志和检查点路径） |
| `trainer.experiment_name` | str | - | 实验名称（用于日志和检查点路径） |
| `trainer.total_epochs` | int | 4 | 训练总轮数 |
| `trainer.logger` | list | `["console","wandb"]` | 日志记录器：`console`（控制台）、`wandb`（Weights & Biases） |
| `trainer.default_hdfs_dir` | str | null | HDFS检查点保存目录（可选） |

---

## PPO (Proximal Policy Optimization) 参数

### 数据配置 (data.*)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `data.train_files` | str | - | 训练数据文件路径 |
| `data.val_files` | str | - | 验证数据文件路径 |
| `data.prompt_key` | str | `prompt` | prompt字段的键名 |
| `data.reward_fn_key` | str | `data_source` | 用于选择奖励函数的字段 |
| `data.max_prompt_length` | int | 512 | 最大prompt长度（左填充） |
| `data.max_response_length` | int | 512 | 最大response长度（生成时的最大长度） |
| `data.train_batch_size` | int | 1024 | 训练批次大小 |
| `data.val_batch_size` | int | null | 验证批次大小 |
| `data.shuffle` | bool | true | 是否打乱训练数据 |
| `data.validation_shuffle` | bool | false | 是否打乱验证数据 |
| `data.filter_overlong_prompts` | bool | false | 是否过滤超长prompt |
| `data.truncation` | str | `error` | 超长处理方式 |
| `data.return_raw_input_ids` | bool | false | 是否返回原始input_ids（不应用chat template） |
| `data.return_raw_chat` | bool | false | 是否返回原始chat（不应用chat template） |
| `data.return_full_prompt` | bool | false | 是否返回完整prompt（应用chat template） |

### Actor/Rollout/Reference 模型配置 (actor_rollout_ref.model.*)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `actor_rollout_ref.model.path` | str | - | 模型路径（本地或HDFS） |
| `actor_rollout_ref.model.use_shm` | bool | false | 是否使用共享内存加速模型加载 |
| `actor_rollout_ref.model.enable_gradient_checkpointing` | bool | true | 是否启用梯度检查点 |
| `actor_rollout_ref.model.lora_rank` | int | 0 | LoRA秩 |
| `actor_rollout_ref.model.lora_alpha` | int | 16 | LoRA缩放因子 |
| `actor_rollout_ref.model.use_liger` | bool | false | 是否使用Liger融合 |
| `actor_rollout_ref.model.trust_remote_code` | bool | false | 是否信任远程代码 |

### Actor 配置 (actor_rollout_ref.actor.*)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `actor_rollout_ref.actor.strategy` | str | `fsdp` | 分布式策略：`fsdp`、`fsdp2`、`megatron` |
| `actor_rollout_ref.actor.ppo_mini_batch_size` | int | 256 | PPO小批次大小 |
| `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu` | int | null | 每GPU微批次大小 |
| `actor_rollout_ref.actor.use_dynamic_bsz` | bool | false | 是否动态调整批次大小 |
| `actor_rollout_ref.actor.ppo_max_token_seq_len` | int | - | 每GPU最大token数 |

### Critic 配置 (actor_rollout_ref.critic.*)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `actor_rollout_ref.critic.strategy` | str | `fsdp` | 分布式策略 |
| `actor_rollout_ref.critic.critic_micro_batch_size_per_gpu` | int | null | 每GPU微批次大小 |

### PPO 算法配置 (algorithm.ppo.*)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `algorithm.ppo.num_episodes` | int | - | 每个PPO更新的episode数 |
| `algorithm.ppo.num_collect_rounds` | int | 1 | 数据收集轮数 |
| `algorithm.ppo.episode_len` | int | - | 每个episode的长度 |
| `algorithm.ppo.num_mini_batches` | int | 4 | 小批次数量 |
| `algorithm.ppo.num_ppo_epochs` | int | 4 | PPO更新轮数 |
| `algorithm.ppo.gamma` | float | 1.0 | 折扣因子 |
| `algorithm.ppo.gae_lambda` | float | 0.95 | GAE lambda参数 |
| `algorithm.ppo.entropy_coeff` | float | 0.0 | 熵系数（鼓励探索） |
| `algorithm.ppo.cliprange` | float | 0.2 | PPO裁剪范围 |
| `algorithm.ppo.cliprange_value` | float | 0.2 | 价值函数裁剪范围 |
| `algorithm.ppo.vf_coeff` | float | 1.0 | 价值函数损失系数 |
| `algorithm.ppo.whiten_rewards` | bool | true | 是否对奖励进行白化 |
| `algorithm.ppo.use_adaptive_kl_ctrl` | bool | false | 是否使用自适应KL控制 |

### 训练器配置 (trainer.*)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `trainer.default_local_dir` | str | - | 本地检查点保存目录 |
| `trainer.project_name` | str | - | 项目名称 |
| `trainer.experiment_name` | str | - | 实验名称 |
| `trainer.total_epochs` | int | - | 训练总轮数 |
| `trainer.total_training_steps` | int | null | 总训练步数（可选，优先于total_epochs） |
| `trainer.logger` | list | `["console"]` | 日志记录器 |
| `trainer.save_freq` | int | -1 | 保存频率（-1表示每个epoch保存） |
| `trainer.test_freq` | int | -1 | 验证频率 |
| `trainer.resume_mode` | str | `auto` | 恢复模式：`auto`、`disable`、`resume_path` |
| `trainer.val_before_train` | bool | true | 是否在训练前进行验证 |
| `trainer.device` | str | `cuda` | 设备类型 |

---

## 使用示例

### SFT 训练示例
```bash
bash scripts/sft.sh 0,1,2,3 \
    data.micro_batch_size_per_gpu=2 \
    optim.lr=5e-5 \
    trainer.total_epochs=3 \
    model.lora_rank=16
```

### PPO 训练示例
```bash
bash scripts/ppo.sh \
    data.train_batch_size=512 \
    algorithm.ppo.num_ppo_epochs=2 \
    algorithm.ppo.cliprange=0.1 \
    trainer.total_training_steps=1000
```

---

## 常见参数组合

### 快速测试（小模型、小数据）
```bash
data.micro_batch_size_per_gpu=1
optim.lr=1e-4
trainer.total_epochs=1
model.lora_rank=8
```

### 生产训练（大模型、大数据）
```bash
data.micro_batch_size_per_gpu=4
optim.lr=5e-5
trainer.total_epochs=3
model.lora_rank=32
model.enable_gradient_checkpointing=true
```

### 内存优化
```bash
model.enable_gradient_checkpointing=true
model.use_liger=true
actor_rollout_ref.model.use_shm=true
```

### 加速训练
```bash
model.use_liger=true
use_remove_padding=true
ulysses_sequence_parallel_size=2
```
