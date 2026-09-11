Ascend ReTool Best Practices
===================================

Last updated: 07/03/2026.

Introduction
----------------------------------

For the ReTool paper, refer to ([ReTool](https://arxiv.org/pdf/2504.11536))
It integrates a code interpreter tool, deploys the policy through multi-turn real-time code execution, and teaches the model to learn when and how to call the tool based on result feedback.

1. Environment setup
2. Model training

The use case model scripts and their required hardware conditions are as follows:

.. list-table::
   :header-rows: 1

   * - Model
     - NPU Model
     - Number of Nodes
     - Training and Inference Backend
   * - ``Qwen2.5-7B``
     - Atlas 900 A2
     - 1
     - ``vLLM + FSDP``

Environment Setup
-----------------------------------
1. Build from a custom Conda environment

.. list-table::
   :header-rows: 1

   * - software
     - version
   * - Python
     - ``3.11``
   * - CANN
     - ``==9.0.0.B160`` (CANN900B160)
   * - torch
     - ``==2.9.0``
   * - torch_npu
     - ``==2.9.0``
   * - triton_ascend
     - ``==3.2.1``
   * - verl
     - ``main``
   * - vllm
     - ``v0.18.0``
   * - vllm-ascend
     - ``v0.18.0``
   * - transformers
     - ``5.3.0``

Model Training and Evaluation
-----------------------------------
1. Model Data Preparation
^^^^^^^^^^^
`Qwen2.5-7B`
^^^^^^^^^^^^
**Download Model Weights**

.. code-block:: bash

  git clone https://huggingface.co/Qwen/Qwen2.5-7B-Instruct

**Download the training dataset**

.. code-block:: bash

  git clone https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k

**Download the evaluation dataset**

.. code-block:: bash

  git clone https://huggingface.co/datasets/Maxwell-Jia/AIME_2024

**Pre-training data preprocessing**

.. code-block:: bash

  python3 recipe/retool/retool_sft_preprocess.py

*Note: ReTool-SFT is automatically downloaded, and the generated data is saved in the ~/ReTool-SFT/data directory by default.*

**Run the pre-training script**

.. code-block:: bash

  bash recipe/retool/run_qwen2_7b_sft_npu.sh # Adjust the path in the script

**Merge pre-trained weights to generate a checkpoint**

.. code-block:: bash

  python3 -m verl.model_merger merge --backend fsdp \
      --local_dir /PATH/TO/checkpoint/multiturn-sft-qwen-2.5-7b-instruct/global_step_372 \
      --target_dir /PATH/TO/checkpoint/multiturn-sft-qwen-2.5-7b-instruct/global_step_372/huggingface

2. Code Sandbox Preparation

Open-source sandbox code and deployment reference
https://github.com/bytedance/SandboxFusion

**Downloading Sandbox Code**

.. code-block:: bash

  git clone -b main https://github.com/bytedance/SandboxFusion.git

**Sandbox Installation**

.. code-block:: bash

  cd SandboxFusion
  conda create -n sandbox -y python=3.11
  conda activate sandbox
  pip install poetry
  poetry lock
  poetry install
  mkdir -p docs/build
  cd runtime/python
  bash install-python-runtime.sh
  cd ../../
  make run-online

3. Training

The example configuration file is as follows. Create a run_qwen2.5_7b_dapo_npu.sh file in the recipe/retool directory.
Modify the following parameters in the model training script according to your actual path configuration.

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
  model_path=$DATA_ROOT/dataset/checkpoint/multiturn-sft-qwen-2.5-7b-instruct/global_step_372/huggingface

  train_files="['$dapo_math_17k']"
  test_files="['$aime_2024']"

  # tool
  tool_config_path=recipe/retool/sandbox_fusion_tool_config.yaml

  # wandb
  project_name=retool
  experiment_name=qwen2.5-7b_dapo
  default_local_dir=$DATA_ROOT/checkpoint/$experiment_name

  # Create a log file
  export TIMESTAMP=$(date +%Y%m%d_%H%M%S)
  LOG_DIR="$HDFS_ROOT/verl/logs/$project_name/$experiment_name"
  # Check whether the path exists
  if [ ! -d "$LOG_DIR" ]; then
    # The path does not exist. Create the path.
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
