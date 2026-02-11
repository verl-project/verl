# DAPO multi model optimization practice

## DAPO 介绍

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
reward_model.reward_manager.name=dapo
```

- **Clip-Higher 更高裁剪**
  `clip_ratio_low` 和 `clip_ratio_high` 用于指定 DAPO 目标函数中的 $\varepsilon_{\text {low }}$ 和 $\varepsilon_{\text {high }}$。

```
clip_ratio_low=0.2  # 裁剪比例下限，默认值为0.2
clip_ratio_high=0.28 # 裁剪比例上限，默认值为0.28
```

- **动态采样的相关配置**
  将 `filter_groups.enable` 设置为 `True` 会过滤掉输出 `metric` 完全相同的组，例如对于 `acc` 指标，过滤掉输出准确率全部为 1 或 0 的组。
  训练器会使用 `gen_batch_size` 进行重复采样，直到生成足够数量的符合条件的组，或者达到 `max_num_gen_batches` 所指定的上限为止。

```
data.gen_batch_size=${gen_prompt_bsz}
algorithm.filter_groups.enable=${enable_filter_groups} # 动态采样开关
algorithm.filter_groups.metric=${filter_groups_metric} # 使用准确率作为过滤标准
algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} # 最大生成批次数量,最多重复生成数据的次数
```

- **Token-level Loss**
  将 `loss_agg_mode` 设置为 `token-mean` 意味着计算一个批次中所有序列内所有 token 的（策略梯度）损失的平均值。

```
actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode}
#注意：“token-mean”是默认行为。
```

- **奖励模型对超长回答的惩罚配置**
  将 `overlong_buffer.enable` 设置为 `True` 将对输出长度过长但仍未超过硬上下文限制的输出进行惩罚。具体来说，当输出的长度超过 `max_response_length - overlong_buffer.len` 且超出 `0` 到 `overlong_buffer.len` 个 token 时，惩罚值会从 `0` 线性增加到 `overlong_buffer.penalty_factor`。

```
reward_model.overlong_buffer.enable=${enable_overlong_buffer} # 启用超长缓冲区惩罚,开启对超长输出的惩罚机制
reward_model.overlong_buffer.len=${overlong_buffer_len}  # 缓冲区长度,定义缓冲区的toke,最大惩罚强度
reward_model.overlong_buffer.penalty_factor=${overlong_penalty_factor}   #惩罚因子,最大惩罚强度
```

相关参数涉及的代码可以参考：[Recipe: Decoupled Clip and Dynamic Sampling Policy Optimization (DAPO)](https://github.com/verl-project/verl-recipe/blob/main/dapo/README.md)

## 硬件要求

当前支持Atlas 800T A3 与 Atlas 900 A3 SuperPoD。完成跑完本次最佳实践需要 2台Atlas 800T A3。关键软件版本可以参考：[Ascend Quickstart](https://github.com/volcengine/verl/blob/main/docs/ascend_tutorial/ascend_quick_start.rst)

## 安装基础环境

| software | version|
| --- | --- |
| Python| >= 3.10, <3.12 |
| CANN | == 8.3.RC1 |
| torch | == 2.7.1 |
| torch_npu | == 2.7.1 |
| verl | main分支 commitId=252d76908b903ad8fb6969eb3a5e5f873c95ea2b |
| vllm | 	v0.11.0 |
| vllm-ascend | v0.11.0-dev|
| transformers | 	4.57.3|

在本实践中, 我们通过指定 verl 的commit id 以避免引入其他问题
```
cd verl
git checkout 252d76908b903ad8fb6969eb3a5e5f873c95ea2b
# 指定相应的recipe版本
git submodule update --init --recursive recipe
```

## 模型训练

### 数据集准备

Geometry3k 数据集是由加利福尼亚大学洛杉矶分校与浙江大学联合研发的几何领域专用数据集，核心面向视觉问答（VQA）任务展开研究与模型训练。该数据集总计包含 3002 个样本，采用图像和文本两种模态数据形式构建，其中文本模态涵盖各类几何问题描述，图像则以可视化图表呈现问题中的几何图形信息，包括三角形、圆形、四边形等基础几何形状，以及不同图形间的位置、嵌套、相交等关联关系。可以从Hugging Face库下载对应的原始数据集：[Geometry3k ](https://huggingface.co/datasets/hiyouga/geometry3k)

```python
# 下载原始数据并预处理
python ./examples/data_preprocess/geo3k.py --local_dir=./data/geo3k
```

### 权重下载

从Hugging Face库下载对应的模型权重：[Qwen3-VL-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct/tree/main
)

### 全局变量导入

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

- 昇腾多卡通信的兜底配置，延长连接超时时间，避免集群环境下训练启动因连接慢而失败

```
export HCCL_CONNECT_TIMEOUT=5400
```

- 控制 vLLM 在昇腾芯片上是否启用NZ优化

```
export VLLM_ASCEND_ENABLE_NZ=0
```

### 训练
```
# Model Weights Paths
MODEL_PATH=hf_weights/Qwen3-VL-30B-A3B-Instruct
RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}

# File System Paths
TRAIN_FILE=$RAY_DATA_HOME/datasets/geo3k/train.parquet
TEST_FILE=$RAY_DATA_HOME/datasets/geo3k/test.parquet

#保存频率，-1默认不保存，如需评测请修改此参数
trainer.save_freq=-1
```

对于单机任务 Qwen3-VL-30B , 修改脚本中参数`trainer.nnodes`为 1， `trainer.n_gpus_per_node` 为16，然后直接bash执行verl仓上示例脚本

```
bash recipe/dapo/run dapo_qwen3_vl_30b_fsdp2_npu.sh
```
对于多节点任务 Qwen3-VL-30B ，我们推荐使用以下脚本进行大规模多节点训练拉起

```
pkill -9 python
ray stop --force
rm -rf /tmp/ray
export VLLM_USE_V1=1
export HCCL_CONNECT_TIMEOUT=5400
export VLLM_ASCEND_ENABLE_NZ=0
export LD_PRELOAD=/usr/local/lib/libjemalloc.so.2
# Some models are optimized by vllm ascend. While in some case, e.g. rlhf training, 
# the optimized model may not be suitable. In this case, set this value to 0 to disable the optimized model.
export USE_OPTIMIZED_MODEL=0

# 修改为当前需要跑的用例路径
DEFAULT_SH="./run_*.sh"
echo "Use $DEFAULT_SH"

ulimit -n 32768
mkdir logs

NNODES=2
NPUS_PER_NODE=8
# 修改为对应主节点IP
MASTER_ADDR="IP FOR MASTER NODE"
# 修改为当前节点的通信网卡
SOCKET_IFNAME="Your SOCKET IFNAME"
export HCCL_SOCKET_IFNAME="SOCKET IFNAME FOR CURRENT NODE"
export GLOO_SOCKET_IFNAME="SOCKET IFNAME FOR CURRENT NODE"
# 获取当前IP
CURRENT_IP=$(ifconfig $SOCKET_IFNAME | grep -Eo 'inet (addr:)?([0-9]{1,3}\.){3}[0-9]{1,3}' | awk '{print $NF}')
if [ "$MASTER_ADDR" = "$CURRENT_IP" ]; then
  # 主节点启动
  ray start --head --port 6766 --dashboard-host=$MASTER_ADDR --node-ip-address=$CURRENT_IP --dashboard-port=8260 --resources='{"NPU": '$NPUS_PER_NODE'}'

  while true; do
      ray_status_output=$(ray status)
      npu_count=$(echo "$ray_status_output" | grep -oP '(?<=/)\d+\.\d+(?=\s*NPU)' | head -n 1)
      npu_count_int=$(echo "$npu_count" | awk '{print int($1)}')
      device_count=$((npu_count_int / $NPUS_PER_NODE))

      # 判断device_count 是否与 NNODES 相等
      if [ "$device_count" -eq "$NNODES" ]; then
          echo "Ray cluster is ready with $device_count devices (from $npu_count NPU resources), starting Python script."
          ray status
          bash $DEFAULT_SH
          break
      else
          echo "Waiting for Ray to allocate $NNODES devices. Current device count: $device_count"
          sleep 5
      fi
  done
else
  # 子节点尝试往主节点注册 ray 直到成功
  while true; do
      # 尝试连接 ray 集群
      ray start --address="$MASTER_ADDR:6766" --resources='{"NPU": '$NPUS_PER_NODE'}' --node-ip-address=$CURRENT_IP

      # 检查连接是否成功
      ray status
      if [ $? -eq 0 ]; then
          echo "Successfully connected to the Ray cluster!"
          break
      else
          echo "Failed to connect to the Ray cluster. Retrying in 5 seconds..."
          sleep 5
      fi
  done
fi

sleep 600
```
DEFAULT_SH:修改为训练所用配置 sh 文件路径。在此案例中修改为 [Qwen3_VL_30B](https://github.com/verl-project/verl-recipe/blob/main/dapo/run%20dapo_qwen3_vl_30b_fsdp2_npu.sh) 路径。

NNODES 和 NPUS_PER_NODE:修改为使用节点数量和每个节点 NPU 数量。在此案例中分别为2和8。

MASTER_ADDR:修改为对应主节点 IP。即所有节点的 MASTER_ADDR 应该相同。

SOCKET_IFNAME, HCCL_SOCKET_IFNAME, GLOO_SOCKET_IFNAME: 修改为对应通信网卡，通信网卡可以通过以下命令获取：

```
ifconfig |grep "$(hostname -I |awk '{print $1}'|awk -F '.' '{print $0}')" -B 1|awk -F ':' '{print$1}' | head -1 | tail -1
```

## 优化参考

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


