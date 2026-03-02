Ascend Retool Best Practice
===================================

Last updated: 02/10/2026.

引言
----------------------------------

Retool论文参考([Retool](https://arxiv.org/pdf/2504.11536))
集成代码解释器工具，通过多轮实时代码执行进行策略部署，并教会模型根据结果反馈学习何时以及如何调用工具。

1. 环境构建
2. 模型训练

用例模型脚本以及其需要的硬件条件各自如下：

===============    ============    ============    ===============
模型                NPU型号         节点数量        训推后端
===============    ============    ============    ===============
``Qwen2.5-7B``     Atlas 900 A2         1          ``vllm + FSDP``
===============    ============    ============    ===============

环境构建
-----------------------------------
1.从自定义Conda环境进行构建

============    ============================================================
software        version 
============    ============================================================
Python          ``>= 3.10, <3.12``
CANN            ``== 8.3.RC1``
torch           ``== 2.7.1``
torch_npu       ``== 2.7.1``
verl            ``v0.6.1 commitId=d62da4950573d7a4b7ef2362337952e7ab59e78d``
vllm            ``v0.11.0``
vllm-ascend     ``v0.11.0-dev``
transformers    ``4.57.6``
============    ============================================================

模型训练与评估
-----------------------------------
1.模型数据准备
^^^^^^^^^^^
`Qwen2.5-7B`
^^^^^^^^^^^
**下载模型权重**

--local-dir: 模型保存路径

.. code-block:: bash

  git clone https://huggingface.co/Qwen/Qwen2.5-7B-Instruct

**下载训练数据集**

.. code-block:: bash

  git clone https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k

**下载评估数据集**

.. code-block:: bash

  git clone https://huggingface.co/datasets/Maxwell-Jia/AIME_2024

**下载预训练数据集**

.. code-block:: bash

  python3 recipe/retool/retool_sft_preprocess.py

*注:自动下载ReTool-SFT，最后生成数据默认保存在~/ReTool-SFT/data目录下*

**执行预训练脚本**

.. code-block:: bash

  bash recipe/retool/run_qwen2_7b_sft_npu.sh # 需适配脚本中路径

**合并预训练权重生成checkpoint**

.. code-block:: bash

  python3 -m verl.model_merger merge --backend fsdp \
      --local_dir ${DATASETS}/checkpoint/multiturn-sft-qwen-2.5-7b-instruct/global_step_372 \
      --target_dir ${DATASETS}/checkpoint/multiturn-sft-qwen-2.5-7b-instruct/global_step_372/huggingface

2.代码沙箱准备

开源沙箱代码及部署参考
https://github.com/bytedance/SandboxFusion

**沙箱代码下载**

.. code-block:: bash

  git clone -b main https://github.com/bytedance/SandboxFusion.git

**沙箱安装**

.. code-block:: bash

  conda create -n sandbox -y python=3.11
  conda activate sandbox
  pip install poetry
  poetry lock
  poetry install
  mkdir -p docs/build
  cd runtime/python
  bash install-python-runtime.sh
  make run-online

3.训练

示例配置文件如下，在recipe/retool目录下创建一个run_qwen2.5_7b_dapo_npu.sh
根据开发者实际路径配置情况修改模型训练脚本中的以下参数

.. code-block:: bash 

  set -x

  export VLLM_USE_V1=1
  export TORCHDYNAMO_DISABLE=1
  export VLLM_ASCEND_ENABLE_NZ=0
  export TASK_QUEUE_ENABLE=1
  export VLLM_ENABLE_GRAPH_MODE=1
  export HCCL_OP_EXPANSION_MODE="AIV"
  export VLLM_ASCEND_ENABLE_MLP_OPTIMIZE=1
  export LD_PRELOAD=/usr/local/lib/libjemalloc.so.2
  
  # ================= data/model/tool =================
  HDFS_ROOT=${HDFS_ROOT:-"${PWD}"}
  DATA_ROOT=${DATA_ROOT:-"${PWD}"}
  
  dapo_math_17k=$DATA_ROOT/dataset/BytedTsinghua-SIA/DAPO-Math-17k
  aime_2024=$DATA_ROOT/dataset/Maxwell-Jia/AIME_2024
  #aime_2025=$DATA_ROOT/dataset/yentinglin/aime_2025
  model_path=$DATA_ROOT/dataset/checkpoint/multiturn-sft-qwen-2.5-7b-instruct/  global_step_372/huggingface
  
  train_files="['$dapo_math_17k']"
  test_files="['$aime_2024']"
  
  # tool
  tool_config_path=recipe/retool/sandbox_fusion_tool_config.yaml
  
  # wandb
  project_name=retool
  experiment_name=qwen2.5-7b_dapo
  default_local_dir=$DATA_ROOT/checkpoint/$experiment_name
  
  # 创建日志文件
  export TIMESTAMP=$(date +%Y%m%d_%H%M%S)
  LOG_DIR="$HDFS_ROOT/verl/logs/$project_name/$experiment_name"
  # 判断路径是否存在
  if [ ! -d "$LOG_DIR" ]; then
    # 路径不存在，创建路径
    mkdir -p "$LOG_DIR"
    echo "Directory $LOG_DIR created."
  else
    echo "Directory $LOG_DIR already exists."
  fi
  
  LOG_FILE="${LOG_DIR}/${TIMESTAMP}.log"
  touch "$LOG_FILE"
  echo "Log file $LOG_FILE created."

  # ================= algorithm =================
  adv_estimator=grpo
  
  use_kl_in_reward=False
  kl_coef=0.0
  use_kl_loss=False
  kl_loss_coef=0.0
  
  clip_ratio_low=0.2
  clip_ratio_high=0.28
  
  max_turns=16
  max_prompt_length=2048
  max_response_length=20480
  actor_lr=1e-6
  
  train_batch_size=32
  ppo_mini_batch_size=16
  
  n_resp_per_prompt=16
  n_resp_per_prompt_val=30
  
  # ================= performance =================
  infer_tp=2 # vllm
  train_sp=4 # train
  offload=True
  
  actor_max_token_len_per_gpu=$(( (max_prompt_length + max_response_length) * 1 ))
  log_prob_max_token_len_per_gpu=$(( actor_max_token_len_per_gpu * 4 ))

  PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=$adv_estimator \
    algorithm.use_kl_in_reward=$use_kl_in_reward \
    algorithm.kl_ctrl.kl_coef=$kl_coef \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.return_raw_chat=True \
    data.train_batch_size=$train_batch_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.custom_cls.path=recipe/retool/retool.py \
    data.custom_cls.name=CustomRLHFDataset \
    custom_reward_function.path=recipe/retool/retool.py \
    custom_reward_function.name=compute_score \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_kl_loss=$use_kl_loss \
    actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
    actor_rollout_ref.actor.clip_ratio_low=$clip_ratio_low \
    actor_rollout_ref.actor.clip_ratio_high=$clip_ratio_high \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.optim.lr=$actor_lr \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$actor_max_token_len_per_gpu \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$train_sp \
    actor_rollout_ref.actor.fsdp_config.param_offload=$offload \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$offload \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$log_prob_max_token_len_per_gpu \
    actor_rollout_ref.rollout.max_num_batched_tokens=$actor_max_token_len_per_gpu \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.max_num_seqs=1024 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$infer_tp \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=$max_turns \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=$max_turns \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=$tool_config_path \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    actor_rollout_ref.rollout.n=$n_resp_per_prompt \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.6 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n=$n_resp_per_prompt_val \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    trainer.logger=['console'] \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=8 \
    trainer.val_before_train=False \
    trainer.log_val_generations=20 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.default_local_dir=$default_local_dir \
    trainer.test_freq=20 \
    trainer.device=npu \
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=True \
    actor_rollout_ref.ref.entropy_from_logits_with_chunking=True \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.ref.use_torch_compile=False \
    actor_rollout_ref.actor.entropy_checkpointing=True \
    actor_rollout_ref.ref.entropy_checkpointing=True \
    actor_rollout_ref.ref.use_torch_compile=False \
    trainer.total_epochs=1 $@ > $LOG_FILE 2>&1 &
