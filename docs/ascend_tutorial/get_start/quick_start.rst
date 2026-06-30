Ascend Quickstart
=================

**Last updated:** 06/30/2026.

关键更新
--------

- 2026/06/30：新增覆盖四种常用训推后端组合，便于用户在 quickstart 阶段快速选择合适的启动脚本。
- 2026/05/13：将 quick start 和 install guidance 分开。
- 2025/12/11：verl 存量场景目前支持自动识别 NPU 设备类型，GPU 脚本在昇腾上运行，原则上不再需要显式设置 ``trainer.device=npu`` 参数，新增特性通过设置 ``trainer.device`` 仍可优先使用，逐步适配自动识别能力。

硬件支持
--------

- Atlas 200T A2 Box16
- Atlas 900 A2 PODc
- Atlas 800T A3



基础验证场景
------------

本文面向 Ascend NPU 环境，提供基于 GSM8K 和 Qwen3-0.6B 的最小 GRPO 训练验证流程。
文档覆盖四种常用训推后端组合，便于用户在 quickstart 阶段快速选择合适的启动脚本：

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - 组合
     - 训练后端
     - rollout 后端
   * - vLLM + FSDP2
     - FSDP2
     - vLLM-Ascend
   * - vLLM + Megatron
     - Megatron
     - vLLM-Ascend
   * - SGLang + FSDP2
     - FSDP2
     - SGLang
   * - SGLang + Megatron
     - Megatron
     - SGLang


运行本文脚本前，请确认已完成 verl Ascend 环境安装，并已准备好模型权重和训练数据。
环境安装详见 `install_guidance <./install_guidance.rst>`_ 。
四个脚本均默认使用 ``Qwen/Qwen3-0.6B`` 和 GSM8K 数据集进行基础链路验证。

该场景用于检查：

- verl 入口是否可用；
- 数据是否可读取；
- actor、rollout、reference worker 是否能初始化；
- vLLM-Ascend/sglang rollout 是否能生成；
- 训练链路是否能完成首个 step。

准备 Qwen3-0.6B 权重
~~~~~~

权重需自行从huggingface上下载

脚本中的默认读取权重路径为 ``~${HOME}/models/Qwen/Qwen3-0.6B``

建议将权重放在该路径下，或者修改脚本中MODEL_PATH指向本地路径


准备 GSM8K 数据
~~~~~~

.. code-block:: bash

   python3 examples/data_preprocess/gsm8k.py --local_dataset_path /download/path/hf_data/gsm8k/

gsm8k原始数据集需自行从huggingface上下载

生成文件：

.. code-block:: text

   ~/data/gsm8k/train.parquet
   ~/data/gsm8k/test.parquet

运行方式
~~~~~~

可以将下方任一体验脚本保存为 ``.sh`` 文件后执行，例如：

.. code-block:: bash

   bash run_qwen3_0_6b_grpo_fsdp2_vllm_ascend.sh


Ascend Quickstart with vLLM+FSDP2 Backend
-------------------------------------------

该组合使用 FSDP2 作为训练后端，使用 vLLM-Ascend 作为 rollout 后端

体验脚本
~~~~~~~~

.. code-block:: bash

   set -xeuo pipefail
   
   MODEL_ID=${MODEL_ID:-Qwen/Qwen3-0.6B}
   MODEL_PATH=${MODEL_PATH:-${HOME}/models/${MODEL_ID}}
   NNODES=${NNODES:-1}
   NDEVICES_PER_NODE=${NDEVICES_PER_NODE:-}
   
   TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
   PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-8}
   MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
   MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
   PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-24576}
   
   ACTOR_LR=${ACTOR_LR:-1e-6}
   KL_LOSS_COEF=${KL_LOSS_COEF:-0.01}
   ENTROPY_COEFF=${ENTROPY_COEFF:-0}
   
   ROLLOUT_TP=${ROLLOUT_TP:-2}
   ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-}
   ROLLOUT_N=${ROLLOUT_N:-5}
   SP_SIZE=${SP_SIZE:-1}
   
   TOTAL_EPOCHS=${TOTAL_EPOCHS:-15}
   TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-15}
   SAVE_FREQ=${SAVE_FREQ:-20}
   TEST_FREQ=${TEST_FREQ:-20}
   
   PROJECT_NAME=${PROJECT_NAME:-verl_grpo_gsm8k}
   EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_0.6b_grpo_fsdp2_vllm_$(date +%Y%m%d_%H%M)}
   
   TRAIN_FILE=${TRAIN_FILE:-$HOME/data/gsm8k/train.parquet}
   TEST_FILE=${TEST_FILE:-$HOME/data/gsm8k/test.parquet}
   
   ########################### derived defaults ###########################
   n_devices_per_node=${NDEVICES_PER_NODE:-8}
   
   export HCCL_CONNECT_TIMEOUT=1500
   export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
   export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
   export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
   
   rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.6}
   
   ########################### parameter arrays ###########################
   
   DATA=(
       algorithm.adv_estimator=grpo
       algorithm.use_kl_in_reward=False
       data.train_files=${TRAIN_FILE}
       data.val_files=${TEST_FILE}
       data.train_batch_size=${TRAIN_BATCH_SIZE}
       data.max_prompt_length=${MAX_PROMPT_LENGTH}
       data.max_response_length=${MAX_RESPONSE_LENGTH}
       data.filter_overlong_prompts=True
       data.truncation='error'
   )
   
   MODEL=(
       actor_rollout_ref.model.path="$MODEL_PATH"
       actor_rollout_ref.model.use_remove_padding=True
       actor_rollout_ref.model.enable_gradient_checkpointing=True
   )
   
   ACTOR=(
       actor_rollout_ref.actor.strategy=fsdp2
       actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
       actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
       actor_rollout_ref.actor.use_dynamic_bsz=True
       actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
       actor_rollout_ref.actor.use_kl_loss=True
       actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF}
       actor_rollout_ref.actor.kl_loss_type=low_var_kl
       actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF}
   )
   
   ROLLOUT=(
       actor_rollout_ref.rollout.name=vllm
       actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}
       actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util}
       actor_rollout_ref.rollout.enable_chunked_prefill=False
       actor_rollout_ref.rollout.n=${ROLLOUT_N}
       actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
       actor_rollout_ref.rollout.calculate_log_probs=True
       actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
   )
   
   REF=(
       actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
       actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
       actor_rollout_ref.ref.fsdp_config.param_offload=True
   )
   
   TRAINER=(
       trainer.balance_batch=True
       trainer.logger='["console"]'
       trainer.project_name=${PROJECT_NAME}
       trainer.experiment_name=${EXPERIMENT_NAME}
       trainer.n_gpus_per_node=${n_devices_per_node}
       trainer.nnodes=${NNODES}
       trainer.save_freq=${SAVE_FREQ}
       trainer.test_freq=${TEST_FREQ}
       trainer.total_epochs=${TOTAL_EPOCHS}
       trainer.total_training_steps=${TOTAL_TRAINING_STEPS}
   )
   
   EXTRA=(
       actor_rollout_ref.actor.use_torch_compile=False
       actor_rollout_ref.actor.fsdp_config.param_offload=True
       actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
       actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=${SP_SIZE}
       actor_rollout_ref.ref.fsdp_config.ulysses_sequence_parallel_size=${SP_SIZE}
       actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=20
   )
   
   ########################### launch ###########################
   python3 -m verl.trainer.main_ppo \
       "${DATA[@]}" \
       "${MODEL[@]}" \
       "${ACTOR[@]}" \
       "${ROLLOUT[@]}" \
       "${REF[@]}" \
       "${TRAINER[@]}" \
       "${EXTRA[@]}" \
       "$@"


Ascend Quickstart with vLLM+Megatron Backend
-------------------------------------------

该组合使用 Megatron 作为训练后端，使用 vLLM-Ascend 作为 rollout 后端

体验脚本
~~~~~~~~

.. code-block:: bash

   set -xeuo pipefail
   
   MODEL_ID=${MODEL_ID:-Qwen/Qwen3-0.6B}
   MODEL_PATH=${MODEL_PATH:-${HOME}/models/${MODEL_ID}}
   NNODES=${NNODES:-1}
   NDEVICES_PER_NODE=${NDEVICES_PER_NODE:-}
   
   TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
   MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
   MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
   
   TOTAL_EPOCHS=${TOTAL_EPOCHS:-15}
   TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-15}
   SAVE_FREQ=${SAVE_FREQ:-20}
   TEST_FREQ=${TEST_FREQ:-20}
   
   PROJECT_NAME=${PROJECT_NAME:-verl_grpo_gsm8k}
   EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_0.6b_grpo_megatron_vllm_$(date +%Y%m%d_%H%M)}
   
   TRAIN_FILE=${TRAIN_FILE:-$HOME/data/gsm8k/train.parquet}
   TEST_FILE=${TEST_FILE:-$HOME/data/gsm8k/test.parquet}
   
   ########################### derived defaults ###########################
   n_devices_per_node=${NDEVICES_PER_NODE:-8}
   
   export HCCL_CONNECT_TIMEOUT=1500
   export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
   export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
   export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
   
   ########################### parameter arrays ###########################
   
   DATA=(
       algorithm.adv_estimator=grpo
       algorithm.use_kl_in_reward=False
       data.train_files=${TRAIN_FILE}
       data.val_files=${TEST_FILE}
       data.train_batch_size=${TRAIN_BATCH_SIZE}
       data.max_prompt_length=${MAX_PROMPT_LENGTH}
       data.max_response_length=${MAX_RESPONSE_LENGTH}
       data.filter_overlong_prompts=True
       data.truncation='error'
   )
   
   MODEL=(
       actor_rollout_ref.model.path="$MODEL_PATH"
   )
   
   ACTOR=(
       actor_rollout_ref.actor.optim.lr=5e-7
       actor_rollout_ref.actor.ppo_mini_batch_size=8
       actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
       actor_rollout_ref.actor.strategy=megatron
       actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=2
       actor_rollout_ref.actor.megatron.tensor_model_parallel_size=2
       actor_rollout_ref.actor.megatron.expert_model_parallel_size=1
       actor_rollout_ref.actor.use_kl_loss=True
       actor_rollout_ref.actor.kl_loss_coef=0.001
       actor_rollout_ref.actor.kl_loss_type=low_var_kl
       actor_rollout_ref.actor.use_torch_compile=False
   )
   
   ROLLOUT=(
       actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
       actor_rollout_ref.rollout.enable_chunked_prefill=False
       actor_rollout_ref.rollout.tensor_model_parallel_size=2
       actor_rollout_ref.rollout.name=vllm
       actor_rollout_ref.rollout.gpu_memory_utilization=0.6
       +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode="FULL_AND_PIECEWISE"
       actor_rollout_ref.rollout.n=2
   )
   
   REF=(
       actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
       actor_rollout_ref.ref.strategy=megatron
       actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=2
       actor_rollout_ref.ref.megatron.tensor_model_parallel_size=2
       actor_rollout_ref.ref.megatron.expert_model_parallel_size=1
       actor_rollout_ref.ref.use_torch_compile=False
   )
   
   TRAINER=(
       trainer.balance_batch=True
       trainer.logger='["console"]'
       trainer.project_name=${PROJECT_NAME}
       trainer.experiment_name=${EXPERIMENT_NAME}
       trainer.n_gpus_per_node=${n_devices_per_node}
       trainer.nnodes=${NNODES}
       trainer.save_freq=${SAVE_FREQ}
       trainer.test_freq=${TEST_FREQ}
       trainer.total_epochs=${TOTAL_EPOCHS}
       trainer.total_training_steps=${TOTAL_TRAINING_STEPS}
   )
   
   
   ########################### launch ###########################
   python3 -m verl.trainer.main_ppo --config-path=config \
       --config-name='ppo_megatron_trainer.yaml' \
       "${DATA[@]}" \
       "${MODEL[@]}" \
       "${ACTOR[@]}" \
       "${ROLLOUT[@]}" \
       "${REF[@]}" \
       "${TRAINER[@]}" \
       "$@"


SGLang 后端通用说明
-------------------------------------------

当前 NPU 上支持 SGLang 后端必须添加以下环境变量。

.. code-block:: bash

   # 支持 NPU 单卡多进程
   # https://www.hiascend.com/document/detail/zh/canncommercial/850/commlib/hcclug/hcclug_000091.html
   export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
   export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050

   # 规避 Ray 在 device 侧调用无法根据 is_npu_available 接口识别设备可用性
   export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1

   # 根据当前设备和需要卡数定义
   export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

   # 使能推理 EP 时需要
   export SGLANG_DEEPEP_BF16_DISPATCH=1

当前 verl 已解析推理常见参数，详见 `async_sglang_server.py <https://github.com/verl-project/verl/blob/main/verl/workers/rollout/sglang_rollout/async_sglang_server.py>`_ 中 ``ServerArgs`` 初始化传参。

其他 `SGLang 参数 <https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/server_arguments.md>`_ 均可通过 ``engine_kwargs`` 进行参数传递。

vLLM 后端脚本转换为 SGLang
~~~~~~~~

vLLM 后端推理脚本转换为 SGLang，需要添加或修改以下参数。

.. code-block:: bash

   # 必须
   actor_rollout_ref.rollout.name=sglang \
   +actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend="ascend" \

   # 可选
   # 使能推理 EP，详细使用方法见：
   # https://github.com/sgl-project/sgl-kernel-npu/blob/main/python/deep_ep/README_CN.md
   ++actor_rollout_ref.rollout.engine_kwargs.sglang.deepep_mode="auto" \
   ++actor_rollout_ref.rollout.engine_kwargs.sglang.moe_a2a_backend="deepep" \

   # MoE 模型多 DP 时必须设置为 True
   +actor_rollout_ref.rollout.engine_kwargs.sglang.enable_dp_attention=False \

Ascend Quickstart with SGLang+FSDP2 Backend
-------------------------------------------

该组合使用 FSDP2 作为训练后端，使用 SGLang 作为 rollout 后端

体验脚本
~~~~~~~~

.. code-block:: bash

   set -xeuo pipefail
   
   MODEL_ID=${MODEL_ID:-Qwen/Qwen3-0.6B}
   MODEL_PATH=${MODEL_PATH:-${HOME}/models/${MODEL_ID}}
   NNODES=${NNODES:-1}
   NDEVICES_PER_NODE=${NDEVICES_PER_NODE:-}
   
   TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
   PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-8}
   MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
   MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
   PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-24576}
   
   ACTOR_LR=${ACTOR_LR:-1e-6}
   KL_LOSS_COEF=${KL_LOSS_COEF:-0.01}
   ENTROPY_COEFF=${ENTROPY_COEFF:-0}
   
   SP_SIZE=${SP_SIZE:-1}
   
   TOTAL_EPOCHS=${TOTAL_EPOCHS:-15}
   TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-15}
   SAVE_FREQ=${SAVE_FREQ:-20}
   TEST_FREQ=${TEST_FREQ:-20}
   
   PROJECT_NAME=${PROJECT_NAME:-verl_grpo_gsm8k}
   EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_0.6b_grpo_fsdp2_sglang_$(date +%Y%m%d_%H%M)}
   
   TRAIN_FILE=${TRAIN_FILE:-$HOME/data/gsm8k/train.parquet}
   TEST_FILE=${TEST_FILE:-$HOME/data/gsm8k/test.parquet}
   
   ########################### derived defaults ###########################
   n_devices_per_node=${NDEVICES_PER_NODE:-8}
   
   export HCCL_CONNECT_TIMEOUT=1500
   export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
   export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
   export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
   export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
   
   
   ########################### parameter arrays ###########################
   
   DATA=(
       algorithm.adv_estimator=grpo
       algorithm.use_kl_in_reward=False
       data.train_files=${TRAIN_FILE}
       data.val_files=${TEST_FILE}
       data.train_batch_size=${TRAIN_BATCH_SIZE}
       data.max_prompt_length=${MAX_PROMPT_LENGTH}
       data.max_response_length=${MAX_RESPONSE_LENGTH}
       data.filter_overlong_prompts=True
       data.truncation='error'
   )
   
   MODEL=(
       actor_rollout_ref.model.path="$MODEL_PATH"
   )
   
   ACTOR=(
       actor_rollout_ref.actor.strategy=fsdp2
       actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
       actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
       actor_rollout_ref.actor.use_dynamic_bsz=True
       actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
       actor_rollout_ref.actor.use_kl_loss=True
       actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF}
       actor_rollout_ref.actor.kl_loss_type=low_var_kl
       actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF}
   )
   
   ROLLOUT=(
       actor_rollout_ref.rollout.name=sglang
       +actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend="ascend"
       actor_rollout_ref.rollout.tensor_model_parallel_size=2
       actor_rollout_ref.rollout.gpu_memory_utilization=0.6
       actor_rollout_ref.rollout.n=2
       actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1)
   
   REF=(
       actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
       actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
       actor_rollout_ref.ref.fsdp_config.param_offload=True
   )
   
   TRAINER=(
       trainer.balance_batch=True
       trainer.logger='["console"]'
       trainer.project_name=${PROJECT_NAME}
       trainer.experiment_name=${EXPERIMENT_NAME}
       trainer.n_gpus_per_node=${n_devices_per_node}
       trainer.nnodes=${NNODES}
       trainer.save_freq=${SAVE_FREQ}
       trainer.test_freq=${TEST_FREQ}
       trainer.total_epochs=${TOTAL_EPOCHS}
       trainer.total_training_steps=${TOTAL_TRAINING_STEPS}
   )
   
   EXTRA=(
       actor_rollout_ref.actor.use_torch_compile=False
       actor_rollout_ref.actor.fsdp_config.param_offload=True
       actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
       actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=${SP_SIZE}
       actor_rollout_ref.ref.fsdp_config.ulysses_sequence_parallel_size=${SP_SIZE}
   )
   
   ########################### launch ###########################
   python3 -m verl.trainer.main_ppo \
       "${DATA[@]}" \
       "${MODEL[@]}" \
       "${ACTOR[@]}" \
       "${ROLLOUT[@]}" \
       "${REF[@]}" \
       "${TRAINER[@]}" \
       "${EXTRA[@]}" \
       "$@"

Ascend Quickstart with SGLang+Megatron Backend
-------------------------------------------

该组合使用 Megatron 作为训练后端，使用 SGLang 作为 rollout 后端。

体验脚本
~~~~~~~~

.. code-block:: bash

   set -xeuo pipefail
   
   MODEL_ID=${MODEL_ID:-Qwen/Qwen3-0.6B}
   MODEL_PATH=${MODEL_PATH:-${HOME}/models/${MODEL_ID}}
   NNODES=${NNODES:-1}
   NDEVICES_PER_NODE=${NDEVICES_PER_NODE:-}
   
   TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
   MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
   MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
   
   TOTAL_EPOCHS=${TOTAL_EPOCHS:-15}
   TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-15}
   SAVE_FREQ=${SAVE_FREQ:-20}
   TEST_FREQ=${TEST_FREQ:-20}
   
   PROJECT_NAME=${PROJECT_NAME:-verl_grpo_gsm8k}
   EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_0.6b_grpo_megatron_sglang_$(date +%Y%m%d_%H%M)}
   
   TRAIN_FILE=${TRAIN_FILE:-$HOME/data/gsm8k/train.parquet}
   TEST_FILE=${TEST_FILE:-$HOME/data/gsm8k/test.parquet}
   
   ########################### derived defaults ###########################
   n_devices_per_node=${NDEVICES_PER_NODE:-8}
   
   export HCCL_CONNECT_TIMEOUT=1500
   export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
   export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
   export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
   export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
   
   
   ########################### parameter arrays ###########################
   
   DATA=(
       algorithm.adv_estimator=grpo
       algorithm.use_kl_in_reward=False
       data.train_files=${TRAIN_FILE}
       data.val_files=${TEST_FILE}
       data.train_batch_size=${TRAIN_BATCH_SIZE}
       data.max_prompt_length=${MAX_PROMPT_LENGTH}
       data.max_response_length=${MAX_RESPONSE_LENGTH}
       data.filter_overlong_prompts=True
       data.truncation='error'
   )
   
   MODEL=(
       actor_rollout_ref.model.path="$MODEL_PATH"
   )
   
   ACTOR=(
       actor_rollout_ref.actor.optim.lr=5e-7
       actor_rollout_ref.actor.ppo_mini_batch_size=8
       actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
       actor_rollout_ref.actor.strategy=megatron
       actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=2
       actor_rollout_ref.actor.megatron.tensor_model_parallel_size=2
       actor_rollout_ref.actor.megatron.expert_model_parallel_size=1
       actor_rollout_ref.actor.use_kl_loss=True
       actor_rollout_ref.actor.kl_loss_coef=0.001
       actor_rollout_ref.actor.kl_loss_type=low_var_kl
       actor_rollout_ref.actor.use_torch_compile=False
   )
   
   ROLLOUT=(
       actor_rollout_ref.rollout.name=sglang
       +actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend="ascend"
       actor_rollout_ref.rollout.tensor_model_parallel_size=2
       actor_rollout_ref.rollout.gpu_memory_utilization=0.6
       actor_rollout_ref.rollout.n=2
       actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
       )
   
   REF=(
       actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
       actor_rollout_ref.ref.strategy=megatron
       actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=2
       actor_rollout_ref.ref.megatron.tensor_model_parallel_size=2
       actor_rollout_ref.ref.megatron.expert_model_parallel_size=1
       actor_rollout_ref.ref.use_torch_compile=False
   )
   
   TRAINER=(
       trainer.balance_batch=True
       trainer.logger='["console"]'
       trainer.project_name=${PROJECT_NAME}
       trainer.experiment_name=${EXPERIMENT_NAME}
       trainer.n_gpus_per_node=${n_devices_per_node}
       trainer.nnodes=${NNODES}
       trainer.save_freq=${SAVE_FREQ}
       trainer.test_freq=${TEST_FREQ}
       trainer.total_epochs=${TOTAL_EPOCHS}
       trainer.total_training_steps=${TOTAL_TRAINING_STEPS}
   )
   
   
   ########################### launch ###########################
   python3 -m verl.trainer.main_ppo --config-path=config \
       --config-name='ppo_megatron_trainer.yaml' \
       "${DATA[@]}" \
       "${MODEL[@]}" \
       "${ACTOR[@]}" \
       "${ROLLOUT[@]}" \
       "${REF[@]}" \
       "${TRAINER[@]}" \
       "$@"



   # chunked_prefill 默认关闭
   +actor_rollout_ref.rollout.engine_kwargs.sglang.chunked_prefill_size=-1
