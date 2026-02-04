## NPU Qwen3-32B GSPO Optimization Practice

Last updated: 01/27/2026.

本文章对应脚本地址：[qwen3_32b_gspo_npu](https://github.com/volcengine/verl/blob/main/examples/gspo_trainer/run_qwen3_32b_gspo_npu.sh)

### 算法适配

GSPO通过将优化颗粒度从**token级**提升到**sequence级**，规避了GRPO会遇到的**方差急剧增大**导致训练不稳定的情况，增加了训练的稳定性，同时该算法也在一定程度上提升了算法的收敛速度。

想要成功在verl仓库中成功调用到GSPO算法，需要进行如下的必要配置

~~~python
# 核心算法配置  
algorithm.adv_estimator=grpo \                    # 使用GRPO优势估计器    
algorithm.use_kl_in_reward=False \                # 不在奖励中添加KL惩罚    
# GSPO策略损失模式  
actor_rollout_ref.actor.policy_loss.loss_mode=gspo \ # 启用GSPO策略损失
# 极小裁剪范围（GSPO特色）  
actor_rollout_ref.actor.clip_ratio_low=0.0003 \   # 裁剪下界，论文推荐值    
actor_rollout_ref.actor.clip_ratio_high=0.0004 \  # 裁剪上界，论文推荐值    
# KL配置（GSPO不使用KL loss）  
actor_rollout_ref.actor.use_kl_loss=False \       # 禁用KL损失    
actor_rollout_ref.actor.kl_loss_coef=0.0 \        # KL损失系数设为0    
# 序列级损失聚合模式（GSPO核心）  
actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \ # 序列级平均，GSPO论文推荐    
# 批次配置  
actor_rollout_ref.rollout.n=16 \                  # 每个prompt生成16个响应（组采样）
~~~

一般选择入口函数为`verl.trainer.main_ppo`

### 性能调优

优化从训练、推理、调度和其他四个方面入手。

#### 训练

##### 动态bsz

~~~bash
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) / sp_size))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) / sp_size))
~~~

**这个优化点主要调整上面这两个参数，不过需要注意这两个参数调整的太大会导致OOM**

**主要调整**`actor_ppo_max_token_len`,调大了会降低训练的耗时，调整`infer_ppo_max_token_len`没有明显的收益，可以不动

**这两个参数的作用介绍如下：**

**这两个参数用于控制动态批处理(dynamic batch size)模式下每个GPU处理的最大token数量**

- **`actor_ppo_max_token_len`**: Actor模型在PPO更新(前向+反向传播)时每个GPU能处理的最大token数
- **`infer_ppo_max_token_len`**: 推理阶段(Reference policy和Rollout)计算log概率时每个GPU能处理的最大token数

#### 推理

##### ACLgraph+FULL_DECODE_ONLY

推理算子下发方面的优化，平均能有`15%~20%`左右的性能收益。

先看单开**ACLgraph**，如下：

~~~bash
# 开启ACLgraph+FULL_DECODE_ONLY（注意：当设置此参数为False时，TASK_QUEUE_ENABLE必须设置为1，不然会报错）
actor_rollout_ref.rollout.enforce_eager=False
actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_capture_sizes='[8,16,32,64,128]' \ 
actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode='FULL_DECODE_ONLY' \
~~~

`FULL_DECODE_ONLY`开启成功后有如下输出：

![FULL_DECODE_ONLY result](https://github.com/wucong25/verl-data/blob/main/ascend_acl_graph.png)

**`cudagraph_capture_sizes`参数设置指南**

cudagraph_capture_sizes设置的值对应的是批大小，这里的批大小不是配置里的DP域对应的那个批次大小，这里是相较于vllm来说的批大小，单位为**token**

默认生成的算法如下，可做参考

![cudagraph_capture_sizes](https://github.com/wucong25/verl-data/blob/main/ascend_set_cudagraph_sizes.png)

##### 推理后端切换

使用方式：`export VLLM_ATTENTION_BACKEND=XFORMERS`

![VLLM_ATTENTION_BACKEND](https://github.com/wucong25/verl-data/blob/main/ascend_vllm_attn_backend.png)

注：需要注意某些后端在一些比较老的vllm-ascend版本内并不支持

##### 使能vllm v1版本

使用方式：`export VLLM_USE_V1=1`

可以常开，一般都是正收益。

#### 调度

##### AIV

打开方式：设置`export HCCL_OP_EXPANSION_MODE="AIV"`

HCCL_OP_EXPANSION_MODE环境变量用于配置通信算法的编排展开位置，支持如下取值：

- AI_CPU：代表通信算法的编排展开位置在Device侧的AI CPU计算单元。
- AIV：代表通信算法的编排展开位置在Device侧的Vector Core计算单元。
- HOST：代表通信算法的编排展开位置为Host侧CPU，Device侧根据硬件型号自动选择相应的调度器。
- HOST_TS：代表通信算法的编排展开位置为Host侧CPU，Host向Device的Task Scheduler下发任务，Device的Task Scheduler进行任务调度执行。

下面介绍两种展开机制

###### HOST展开

<img src="https://github.com/wucong25/verl-data/blob/main/ascend_task_queue1.png" alt="image-20260113194257095" style="zoom:50%;" />

- 软件栈工作在hostcpu，通信算法展开一个个task
- 每个task调用runtime接口，下发到device的rtsqueue
- STARS从rstqueue上顺序拿取task
- 根据task类型分别调用掉SDMA和RDMA引擎。
    **单算子瓶颈**：hostbound 每个task提交是2~5us，一个通信算子有几百个task，单算子场景不会在device上缓存，下发一个执行一个

###### AICpu机制展开

<img src="https://github.com/wucong25/verl-data/blob/main/ascend_task_queue3.png" alt="image-20260113194333218" style="zoom:50%;" />

- host侧不下发一个个task，把通信算子作为一个个kernel，放在通信算子kernel的队列上去。
- STARS调度kernel队列流上的kernel，把kernel放到AiCPU上去执行。
- AICPU调用函数（kernel），用一个线程执行kernel 函数，在函数内把通信task展开，把task放到rstqueue上，STARS调用。
- 降低host和aicpu交互，由几百次降低为一次。
- task的提交在AICPU上提交，做了提交的部分合并。

##### TASK_QUEUE_ENABLE

**使用方式：**`export TASK_QUEUE_ENABLE=2`

TASK_QUEUE_ENABLE，下发优化，图模式设置为1（即开启图模式的时候这个要设置为1），非图模式设置为2

示意图：

![ascend task queue](https://github.com/wucong25/verl-data/blob/main/ascend_task_queue2.png)

##### 绑核优化

**使用方式：**`export CPU_AFFINITY_CONF=1`

详细设置原理可看：https://www.hiascend.com/document/detail/zh/Pytorch/600/ptmoddevg/trainingmigrguide/performance_tuning_0059.html

#### 其他

以下内容汇总了若干全局环境变量的调优配置。由于这些参数在训练阶段与推理阶段往往都能带来正向收益，且目前尚缺乏足够精细的消融实验来严格区分它们各自对训练或推理的贡献占比，故统一归拢在此，供后续持续监控与进一步拆解分析。

##### 使能jemalloc

使用方式（注意需要先安装jemalloc库）：`export LD_PRELOAD=/usr/local/lib/libjemalloc.so.2`

**安装使用教程：**[MindSpeed-RL/docs/install_guide.md · Ascend/MindSpeed-RL - AtomGit | GitCode](https://gitcode.com/Ascend/MindSpeed-RL/blob/master/docs/install_guide.md#高性能内存库-jemalloc-安装)

##### 多流复用

内存方面有优化

使能方式：`export MULTI_STREAM_MEMORY_REUSE=1`

原理介绍：https://www.hiascend.com/document/detail/zh/Pytorch/600/ptmoddevg/trainingmigrguide/performance_tuning_0040.html

##### VLLM_ASCEND_ENABLE_FLASHCOMM

使用方式：`export VLLM_ASCEND_ENABLE_FLASHCOMM=1`

启用昇腾 NPU 特有的FLASHCOMM高速通信优化技术

地址：https://vllm-ascend.readthedocs.io/zh-cn/latest/user_guide/release_notes.html

##### VLLM_ASCEND_ENABLE_DENSE_OPTIMIZE

使用方式：`export VLLM_ASCEND_ENABLE_DENSE_OPTIMIZE=1`

启用昇腾 NPU针对大模型推理的稠密计算优化

地址：https://vllm-ascend.readthedocs.io/zh-cn/latest/user_guide/release_notes.html

##### VLLM_ASCEND_ENABLE_PREFETCH_MLP

使用方式：`export VLLM_ASCEND_ENABLE_PREFETCH_MLP=1`

启用 MLP 层的权重预取机制

<img src="https://github.com/wucong25/verl-data/blob/main/ascend_prefetch.png" alt="image-20251124173132677" style="zoom:50%;" />

##### verl框架参数设置

主要是内存方面的一些设置开关（注意，这个里面的优化都或多或少会导致吞吐量有一定程度的劣化）

~~~bash
# 梯度检查点 (Gradient Checkpointing)
# 作用: 通过重新计算激活值来节省显存,以计算换内存。在前向传播时不保存中间激活值,反向传播时重新计算,可以显著降低显存占用,允许使用更大的batch size。
actor_rollout_ref.model.enable_gradient_checkpointing=True

# 参数卸载 (Parameter Offload)
# 作用: 将模型参数卸载到CPU内存,训练时再加载回GPU。
actor_rollout_ref.actor.fsdp_config.param_offload=${offload}  # True  
actor_rollout_ref.ref.fsdp_config.param_offload=${offload}    # True

# 优化器状态卸载 (Optimizer Offload)
# 作用: 将优化器状态(如Adam的动量)卸载到CPU。优化器状态通常占用大量显存(对于Adam,每个参数需要额外8字节),卸载可以节省显存。
actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload}  # True

# 释放推理引擎缓存 (Free Cache Engine)
# 作用: 在训练阶段释放推理引擎的KV cache和权重。这是3D-HybridEngine的核心优化,允许在同一GPU上交替进行推理和训练,显著降低显存需求。
actor_rollout_ref.rollout.free_cache_engine=True

#  熵计算优化
# entropy_checkpointing: 在训练时对熵计算启用重计算,降低显存峰值
# entropy_from_logits_with_chunking: 分块处理logits张量(如2048 tokens一组),避免一次性加载整个[bsz*seq_len, vocab]张量
actor_rollout_ref.actor.entropy_checkpointing=True  
actor_rollout_ref.ref.entropy_checkpointing=True  
actor_rollout_ref.actor.entropy_from_logits_with_chunking=True  
actor_rollout_ref.ref.entropy_from_logits_with_chunking=True

# 推理引擎显存配置
# gpu_memory_utilization: 控制vLLM使用的GPU显存比例(0.90 = 90%)
# enforce_eager=False: 启用CUDA graphs加速推理,但会占用额外显存
actor_rollout_ref.rollout.gpu_memory_utilization=0.90  
actor_rollout_ref.rollout.enforce_eager=False
~~~

### NPU调优参考文章

环境变量相关：[环境变量列表-Ascend Extension for PyTorch6.0.0-昇腾社区](https://www.hiascend.com/document/detail/zh/Pytorch/600/apiref/Envvariables/Envir_001.html)

社区性能调优教程：[性能调优流程-Ascend Extension for PyTorch6.0.0-昇腾社区](https://www.hiascend.com/document/detail/zh/Pytorch/600/ptmoddevg/trainingmigrguide/performance_tuning_0001.html)

