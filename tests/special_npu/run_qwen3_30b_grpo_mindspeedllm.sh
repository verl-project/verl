set -x

# Project Configuration
project_name='GRPO-Qwen3-30B-BASE-TEST'
exp_name='GRPO-Qwen3-30B-BASE-MindSpeedLLM-SGLang'

# Necessary env
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050

# Node Info
NNODES=${NNODES:-1}
NPUS_PER_NODE=${NPUS_PER_NODE:-8}

# Model Weights Paths
MODEL_ID=${MODEL_ID:-Qwen/Qwen3-30B-A3B-Instruct-2507}
MODEL_PATH=${MODEL_PATH:-${HOME}/.cache/models/${MODEL_ID}}

# use dummy model
if [[ "$USE_DUMMY_MODEL" == "True" ]]; then
    DUMMY_MODEL_PATH=${DUMMY_MODEL_PATH:-${HOME}/models_dummy/${MODEL_ID}}
    if [ -z "${DUMMY_MODEL_CONFIG_PATH}" ]; then
        echo "[ERROR] DUMMY_MODEL_CONFIG_PATH not set"
        exit 1
    fi

    # make sure the path is empty
    if [[ -d $DUMMY_MODEL_PATH && $DUMMY_MODEL_PATH != "/" ]]; then
        rm -rf $DUMMY_MODEL_PATH
    fi

    # init model
    python scripts/init_random_model.py \
        --hf_model_path "${MODEL_PATH}" \
        --new_config_path "${DUMMY_MODEL_CONFIG_PATH}" \
        --output_path "${DUMMY_MODEL_PATH}"

    # replace model path
    MODEL_PATH=$DUMMY_MODEL_PATH
fi

# File System Paths
TRAIN_FILE=$HOME/data/gsm8k/train.parquet
TEST_FILE=$HOME/data/gsm8k/test.parquet
# Data Length Configuration
max_prompt_length=$((512))
max_response_length=$((128))

# Training Batch Configuration
train_prompt_bsz=16
train_prompt_mini_bsz=16
n_resp_per_prompt=2
micro_batch_size=1

# Algorithm Configuration
adv_estimator=grpo
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=True
kl_loss_coef=0.001

# Performance and Memory Management Configuration
all_offload=True
use_dynamic_bsz=False
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length)))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length)))

# Megatron Parallelism Configuration
train_tp=4
train_ep=4
train_etp=1
train_pp=2
train_cp=1

# SGLang Generation Configuration
gen_tp=4
gen_dp=1
gen_ep=1
gpu_memory_utilization=0.5
max_model_len=$((max_prompt_length + max_response_length))
max_num_batched_tokens=$(((max_prompt_length + max_response_length) * 1))

# Data Configuration
DATA_CONFIG=(
    # File Paths
    data.train_files="${TRAIN_FILE}"
    data.val_files="${TEST_FILE}"
    # Data Structure
    data.prompt_key=prompt
    # Batch and Length Configuration
    data.train_batch_size=${train_prompt_bsz}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    # Preprocessing
    data.filter_overlong_prompts=True
    data.truncation='left'
)

# Model Configuration
MODEL_CONFIG=(
    # Model Path
    actor_rollout_ref.model.path="${MODEL_PATH}"
    # Model Processing
    actor_rollout_ref.model.use_remove_padding=True
)

# Reinforcement Learning Algorithm Configuration
ALGORITHM_CONFIG=(
    # Advantage Estimation
    algorithm.adv_estimator=${adv_estimator}
    # KL Divergence Control
    algorithm.use_kl_in_reward=${use_kl_in_reward}
    algorithm.kl_ctrl.kl_coef=${kl_coef}
)

ACTOR_CONFIG=(
    # Core Runtime Settings
    actor_rollout_ref.actor.use_torch_compile=False
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz}
    # Loss Function Configuration
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss}
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef}
    actor_rollout_ref.actor.entropy_coeff=0
    # PPO Training Parameters
    actor_rollout_ref.actor.ppo_epochs=1
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${micro_batch_size}
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len}
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz}
    # Optimizer Settings
    actor_rollout_ref.actor.optim.lr=1e-6
    # Megatron Parallelism Strategy
    actor_rollout_ref.actor.mindspeed.tensor_model_parallel_size=${train_tp}
    actor_rollout_ref.actor.mindspeed.pipeline_model_parallel_size=${train_pp}
    actor_rollout_ref.actor.mindspeed.expert_model_parallel_size=${train_ep}
    actor_rollout_ref.actor.mindspeed.expert_tensor_parallel_size=${train_etp}
    # Memory Optimization
    actor_rollout_ref.actor.mindspeed.param_offload=${all_offload}
    actor_rollout_ref.actor.mindspeed.optimizer_offload=${all_offload}
    actor_rollout_ref.actor.mindspeed.grad_offload=${all_offload}
    # Model Weights Management
    actor_rollout_ref.actor.mindspeed.use_mbridge=True
    actor_rollout_ref.actor.mindspeed.vanilla_mbridge=True
    # Transformer Architecture Optimizations
    actor_rollout_ref.actor.mindspeed.llm_kwargs.spec='[mindspeed_llm.tasks.models.spec.qwen3_spec, layer_spec]'
    actor_rollout_ref.actor.mindspeed.llm_kwargs.seq_length=${max_model_len}
    actor_rollout_ref.actor.mindspeed.llm_kwargs.micro_batch_size=${micro_batch_size}
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.num_query_groups=4
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.recompute_method=uniform
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.recompute_granularity=full
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.recompute_num_layers=1
    # MOE
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.moe_router_load_balancing_type=aux_loss
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.moe_permutation_async_comm=True
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.moe_token_dispatcher_type=alltoall
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.moe_aux_loss_coeff=0.001
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.moe_grouped_gemm=True
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.fix_router=True
)

REF_CONFIG=(
    # Core Runtime Settings
    actor_rollout_ref.ref.use_torch_compile=False
    # Log Probability Inference
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${micro_batch_size}
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz}
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len}
    # Megatron Parallelism Strategy
    actor_rollout_ref.ref.mindspeed.tensor_model_parallel_size=${train_tp}
    actor_rollout_ref.ref.mindspeed.pipeline_model_parallel_size=${train_pp}
    actor_rollout_ref.ref.mindspeed.expert_model_parallel_size=${train_ep}
    actor_rollout_ref.ref.mindspeed.expert_tensor_parallel_size=${train_etp}
    # Memory Optimization
    actor_rollout_ref.ref.mindspeed.param_offload=${all_offload}
    # Model Weights Management
    actor_rollout_ref.ref.mindspeed.use_mbridge=True
    actor_rollout_ref.ref.mindspeed.vanilla_mbridge=True
)

ROLLOUT_CONFIG=(
    # Rollout Engine
    actor_rollout_ref.rollout.name=sglang
    +actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend="ascend"
    # Generation Parameters
    actor_rollout_ref.rollout.n=${n_resp_per_prompt}
    actor_rollout_ref.rollout.top_p=1.0
    actor_rollout_ref.rollout.top_k=-1
    actor_rollout_ref.rollout.temperature=1.0
    # Log Probability Inference
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${micro_batch_size}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz}
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len}
    # Memory Management
    actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization}
    # Parallelism Strategy
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp}
    actor_rollout_ref.rollout.data_parallel_size=${gen_dp}
    actor_rollout_ref.rollout.expert_parallel_size=${gen_ep}
    +actor_rollout_ref.rollout.engine_kwargs.sglang.enable_dp_attention=False
    # Performance Optimization
    +actor_rollout_ref.rollout.engine_kwargs.sglang.chunked_prefill_size=-1
    actor_rollout_ref.rollout.enforce_eager=False
    # Validation Generation
    actor_rollout_ref.rollout.val_kwargs.n=1
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0
    actor_rollout_ref.rollout.val_kwargs.top_k=-1
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0
)

TRAINER_CONFIG=(
    # Logger Configuration
    trainer.logger='["console"]'
    # Project Settings
    trainer.project_name="${project_name}"
    trainer.experiment_name="${exp_name}"
    # Hardware Configuration
    trainer.nnodes="${NNODES}"
    trainer.n_gpus_per_node="${NPUS_PER_NODE}"
    trainer.device='npu'
    # Training Schedule
    trainer.total_epochs=1
    trainer.val_before_train=False
    trainer.test_freq=-1
    trainer.save_freq=-1
    trainer.total_training_steps=1
)

# profiling configuration
PROF_CONFIG=(
    global_profiler.tool=npu
    global_profiler.steps=null
    global_profiler.save_path=/profpath
    actor_rollout_ref.actor.profiler.enable=True
    actor_rollout_ref.actor.profiler.ranks="[0]"
    actor_rollout_ref.actor.profiler.all_ranks=False
    actor_rollout_ref.actor.profiler.tool_config.npu.discrete=True
    actor_rollout_ref.actor.profiler.tool_config.npu.contents=['npu','cpu']
    actor_rollout_ref.actor.profiler.tool_config.npu.level=level0
    actor_rollout_ref.actor.profiler.tool_config.npu.analysis=True
    actor_rollout_ref.rollout.profiler.enable=True
    actor_rollout_ref.rollout.profiler.ranks="[0]"
    actor_rollout_ref.rollout.profiler.all_ranks=False
)

python3 -m verl.trainer.main_ppo \
    --config-path=config \
    --config-name='ppo_trainer.yaml' \
    model_engine=mindspeed \
    "${DATA_CONFIG[@]}" \
    "${MODEL_CONFIG[@]}" \
    "${ACTOR_CONFIG[@]}" \
    "${REF_CONFIG[@]}" \
    "${ROLLOUT_CONFIG[@]}" \
    "${ALGORITHM_CONFIG[@]}" \
    "${TRAINER_CONFIG[@]}" \
    "${PROF_CONFIG[@]}" \
    "$@"

# clean up
if [[ "$USE_DUMMY_MODEL" == "True" ]]; then
    rm -rf $DUMMY_MODEL_PATH
    if [[ "$USE_DIST_CKPT" == "True" ]]; then
        rm -rf $DIST_CKPT_PATH
    fi
fi