# DAPO 介绍

Last updated: 01/27/2026.

DAPO的论文可以参考：[DAPO](https://arxiv.org/pdf/2503.14476)，其中包含以下几个关键技术。

* ​**Clip-Higher**​: 通过对重要性采样比的上限剪裁促进了系统的多样性并避免了熵坍缩（Entropy Collapse）。
* ​**Dynamic Sampling**​: 提高了训练效率和稳定性。DAPO出了一种执行动态采样的策略，并过滤掉准确率等于1和0的提示组，从而保持批次间具有有效梯度的提示数量一致。
* ​**Token-level Policy Gradient Loss**​: 在长链思维强化学习 (long-CoT RL) 场景中至关重要。
* ​**Overlong Reward Shaping**​: 减少奖励噪声并稳定了训练。

在verl中，可以进行如下设置，从而进行DAPO算法的运行。

- **奖励模型的管理策略为 DAPO**
  在dapo算法中，必须配置成dapo。

```
reward_model.reward_manager=dapo
```

- **Clip-Higher 更高裁剪 **
  `clip_ratio_low` 和 `clip_ratio_high` 用于指定 DAPO 目标函数中的 $\varepsilon_{\text {low }}$ 和 $\varepsilon_{\text {high }}$。

```
clip_ratio_low=0.2  # 裁剪比例下限，默认值为0.2
clip_ratio_high=0.28 # 裁剪比例上限，默认值为0.28
```

- **动态采样的相关配置 **
  将 `filter_groups.enable` 设置为 `True` 会过滤掉输出 `metric` 完全相同的组，例如对于 `acc` 指标，过滤掉输出准确率全部为 1 或 0 的组。
  训练器会使用 `gen_batch_size` 进行重复采样，直到生成足够数量的符合条件的组，或者达到 `max_num_gen_batches` 所指定的上限为止。

```
data.gen_batch_size=${gen_prompt_bsz}
algorithm.filter_groups.enable=${enable_filter_groups} # 动态采样开关
algorithm.filter_groups.metric=${filter_groups_metric} # 使用准确率作为过滤标准
algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} # 最大生成批次数量,最多重复生成数据的次数
```

- **Token-level Loss **
  将 `loss_agg_mode` 设置为 `token-mean` 意味着计算一个批次中所有序列内所有 token 的（策略梯度）损失的平均值。

```
actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode}
#注意：“token-mean”是默认行为。
```

- **奖励模型对超长回答的惩罚配置 **
  将 `overlong_buffer.enable` 设置为 `True` 将对输出长度过长但仍未超过硬上下文限制的输出进行惩罚。具体来说，当输出的长度超过 `max_response_length - overlong_buffer.len` 且超出 `0` 到 `overlong_buffer.len` 个 token 时，惩罚值会从 `0` 线性增加到 `overlong_buffer.penalty_factor`。

```
reward_model.overlong_buffer.enable=${enable_overlong_buffer} # 启用超长缓冲区惩罚,开启对超长输出的惩罚机制
reward_model.overlong_buffer.len=${overlong_buffer_len}  # 缓冲区长度,定义缓冲区的toke,最大惩罚强度
reward_model.overlong_buffer.penalty_factor=${overlong_penalty_factor}   #惩罚因子,最大惩罚强度
```

相关参数涉及的代码可以参考：[Recipe: Decoupled Clip and Dynamic Sampling Policy Optimization (DAPO)](https://github.com/verl-project/verl-recipe/blob/main/dapo/README.md)

# 硬件要求

当前支持Atlas 800T A3 与 Atlas 900 A3 SuperPoD。完成跑完本次最佳实践需要 2台Atlas 800T A3。关键软件版本可以参考：[Ascend Quickstart](https://github.com/volcengine/verl/blob/main/docs/ascend_tutorial/ascend_quick_start.rst)

# 模型训练

## 数据集准备

Geometry3k 数据集是由加利福尼亚大学洛杉矶分校与浙江大学联合研发的几何领域专用数据集，核心面向视觉问答（VQA）任务展开研究与模型训练。该数据集总计包含 3002 个样本，采用图像和文本两种模态数据形式构建，其中文本模态涵盖各类几何问题描述，图像则以可视化图表呈现问题中的几何图形信息，包括三角形、圆形、四边形等基础几何形状，以及不同图形间的位置、嵌套、相交等关联关系。可以从Hugging Face库下载对应的原始数据集：[Geometry3k ](https://huggingface.co/datasets/hiyouga/geometry3k)

```python
# 下载原始数据并预处理
python ./examples/data_preprocess/geo3k.py --local_dir=./data/geo3k
```

## 权重下载

从Hugging Face库下载对应的模型权重：[Qwen3-VL-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct/tree/main
)

## 全局变量导入

- 为了确保 Ray 进程能够正常回收内存，需要安装并使能 jemalloc 库进行内存管理，用于更好管理内存，避免长跑过程中内存 OOM。

```
# 根据实际安装路径设置 jemalloc 环境变量
export LD_PRELOAD=/usr/local/lib/libjemalloc.so.2
```

- 某些模型是通过 vllm ascend 进行优化的。但在某些情况下，优化后的模型可能并不适用。此时，将此值设置为 0 即可禁用优化后的模型。

```
export USE_OPTIMIZED_MODEL=0
```

- 启用vLLM V1

```
export VLLM_USE_V1=1
```

昇腾多卡通信的兜底配置，延长连接超时时间，避免集群环境下训练启动因连接慢而失败

```
export HCCL_CONNECT_TIMEOUT=5400
```

- 控制 vLLM 在昇腾芯片上是否启用NZ优化

```
export VLLM_ASCEND_ENABLE_NZ=0
```

- 根据使用机器的情况，修改相关配置， 例如双机机 A2 可设置`trainer.nnodes`为 1 、`trainer.n_gpus_per_node`为8

## 训练脚本

基于以上修改，提供了示例配置文件，创建 run_dapo_qwen3_vl_30b.sh 文件。

```bash
set -xeuo pipefail

export VLLM_USE_V1=1
export HCCL_CONNECT_TIMEOUT=5400
export VLLM_ASCEND_ENABLE_NZ=0
export LD_PRELOAD=/usr/local/lib/libjemalloc.so.2
# Some models are optimized by vllm ascend. While in some case, e.g. rlhf training, 
# the optimized model may not be suitable. In this case, set this value to 0 to disable the optimized model.
export USE_OPTIMIZED_MODEL=0

project_name='DAPO'
exp_name='DAPO-Qwen3-vl-30B'

adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

max_prompt_length=1024
max_response_length=2048
enable_overlong_buffer=False
overlong_buffer_len=$((1024 * 2))
overlong_penalty_factor=1.0

loss_agg_mode="token-mean"

enable_filter_groups=True
filter_groups_metric=acc
max_num_gen_batches=4
train_prompt_bsz=64
gen_prompt_bsz=$((train_prompt_bsz * 3))
n_resp_per_prompt=8
train_prompt_mini_bsz=16

# Ray
PWD=./
RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}
WORKING_DIR=${WORKING_DIR:-"${PWD}"}
RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/verl/trainer/runtime_env.yaml"}

# Paths
RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
MODEL_PATH=${MODEL_PATH:-"${RAY_DATA_HOME}/models/Qwen3-VL-30B-A3B-Instruct"}
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/geo3k/train.parquet"}
TEST_FILE=${TEST_FILE:-"${RAY_DATA_HOME}/data/geo3k/test.parquet"}

# Algorithm
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
val_top_p=0.7

# Performance Related Parameter
sp_size=8
use_dynamic_bsz=True
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) / sp_size))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) / sp_size))
gen_tp=8
fsdp_size=16

ray job submit --no-wait --runtime-env="${RUNTIME_ENV}" \
    --working-dir "${WORKING_DIR}" \
    --address "${RAY_ADDRESS}" \
    -- python3 -m recipe.dapo.main_dapo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.train_batch_size=${train_prompt_bsz} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    algorithm.filter_groups.metric=${filter_groups_metric} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.70 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k="${top_k}" \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.expert_parallel_size=8 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.name=vllm \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.ref.strategy=fsdp2 \
    critic.strategy=fsdp2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${fsdp_size} \
    reward_model.reward_manager=dapo \
    reward_model.overlong_buffer.enable=${enable_overlong_buffer} \
    reward_model.overlong_buffer.len=${overlong_buffer_len} \
    reward_model.overlong_buffer.penalty_factor=${overlong_penalty_factor} \
    trainer.logger=console \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=2 \
    trainer.val_before_train=True \
    trainer.test_freq=1 \
    trainer.save_freq=20 \
    trainer.resume_mode=auto \
    trainer.device=npu \
    trainer.total_epochs=30 \
    trainer.total_training_steps=100 \
    trainer.default_local_dir="${CKPTS_DIR}"
```

# 优化参考

- **启动动态批次大小**
  根据单 GPU 的最大 Token 总数（ppo_max_token_len_per_gpu）动态调整批次大小

```
actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz}
actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz}
actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz}
```

- **单个 GPU 能处理的最大 Token 总数**
  当`use_dynamic_bsz=True`时，单 GPU 在一个微批次中能处理的最大 Token 数量

```
actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len}  
actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} 
actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len}
```

- **单个 GPU 微批次大小**
  当`use_dynamic_bsz=True`时，框架会以该值为​初始批次大小​，再根据`ppo_max_token_len_per_gpu`向上 / 向下调整

```
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 
actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 
actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2
```

- **启用 FSDP2 框架**
  “将模型参数、梯度、优化器状态分片存储在不同 GPU 上”，避免单卡加载全量模型导致显存溢出。

```
# 启用 FSDP2 框架
actor_rollout_ref.actor.strategy=fsdp2 
actor_rollout_ref.ref.strategy=fsdp2 
critic.strategy=fsdp2

# 仅用于 FSDP2：前向传播后重新分片以减少内存占用。
actor_rollout_ref.actor.fsdp_config.reshard_after_forward=True
# 仅用于 FSDP2：是否在模型前向传播后重新分片以节省内存。
actor_rollout_ref.ref.fsdp_config.reshard_after_forward=True
```

- **启用专家并行配置**
  指定有多少个 GPU用于并行计算不同的专家网络

```
# MoE 架构 Actor 模型的专家并行配置
actor_rollout_ref.rollout.expert_parallel_size=8
```


