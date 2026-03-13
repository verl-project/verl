#!/usr/bin/env bash
set -xeuo pipefail

################################################### environment ###################################################
### # 1. use docker image `verlai/verl:vllm015.dev`` and install correct dependencies:
# pip install nvidia-modelopt
# MAX_JOBS=32 pip install git+https://github.com/Dao-AILab/causal-conv1d.git --no-build-isolation --no-cache-dir
# MAX_JOBS=32 pip install git+https://github.com/state-spaces/mamba.git --no-build-isolation --no-cache-dir
# pip install --no-deps git+https://github.com/NVIDIA-NeMo/Megatron-Bridge 
# pip install --no-deps git+https://github.com/NVIDIA/Megatron-LM.git@core_dev_r0.16.0
# unset ROCR_VISIBLE_DEVICES
# unset PYTORCH_CUDA_ALLOC_CONF

################################################### quick config ###################################################

rollout_mode="async"
return_raw_chat="False"
rollout_name="vllm" # sglang or vllm
if [ "$rollout_mode" = "async" ]; then
    export VLLM_USE_V1=1
    return_raw_chat="True"
fi
dtype="bfloat16"

project_name='DAPO'
exp_name='nano_30b'

adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 2))
enable_overlong_buffer=True
overlong_buffer_len=$((1024 * 1))
overlong_penalty_factor=1.0

loss_agg_mode="token-mean"

train_prompt_bsz=32
n_resp_per_prompt=16
train_prompt_mini_bsz=32

# Ray
RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}
WORKING_DIR=${WORKING_DIR:-"${PWD}"}
RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/verl/verl/trainer/runtime_env.yaml"}
NNODES=${NNODES:-1}
# Paths
MODEL_PATH=${MODEL_PATH:-"${RAY_DATA_HOME}/models/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"}
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/dapo-math-17k.parquet"}
TEST_FILE=${TEST_FILE:-"${RAY_DATA_HOME}/data/aime-2024.parquet"}

# Algorithm
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
val_top_p=0.7

# Performance Related Parameter
use_dynamic_bsz=True
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 10 / 10))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 1))
offload=True
gen_tp=8
train_tp=8
train_pp=1
EP=8
ETP=1

################################################### start of config ###################################################

FP8=(
    # train
    # +actor_rollout_ref.actor.megatron.override_transformer_config.fp8="e4m3" # e4m3 or hybrid
    # +actor_rollout_ref.actor.megatron.override_transformer_config.fp8_recipe="blockwise"
    # +actor_rollout_ref.actor.optim.override_optimizer_config.fp8_recipe="blockwise"
    # rollout
    actor_rollout_ref.actor.megatron.dtype=${dtype}
    actor_rollout_ref.rollout.dtype=${dtype}
    # +actor_rollout_ref.rollout.quantization="fp8"
)

DATA=(
    data.train_files="${TRAIN_FILE}"
    data.val_files="${TEST_FILE}"
    data.prompt_key=prompt
    data.return_raw_chat=$return_raw_chat
    data.truncation='left'
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.train_batch_size=${train_prompt_bsz}
)

REWARD_MODEL=(
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer}
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len}
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor}
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=False
    +reward_model.reward_kwargs.max_resp_len=${max_response_length}
    reward_model.reward_manager=dapo
)

PERF_OPT=(
    +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=True
    actor_rollout_ref.model.use_fused_kernels=False
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype=fp32
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1 \
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True \
)

ACTOR=(
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss}
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef}
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low}
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high}
    actor_rollout_ref.actor.clip_ratio_c=10.0
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz}
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len}
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.optim.lr_warmup_steps=10
    actor_rollout_ref.actor.optim.weight_decay=0.1
    actor_rollout_ref.actor.optim.clip_grad=1.0
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz}
    actor_rollout_ref.actor.megatron.param_offload=${offload}
    actor_rollout_ref.actor.megatron.optimizer_offload=${offload}
    actor_rollout_ref.actor.megatron.grad_offload=${offload}
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${train_pp}
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${train_tp}
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=$EP
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=$ETP
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode}
    actor_rollout_ref.actor.megatron.use_mbridge=True
    actor_rollout_ref.actor.megatron.vanilla_mbridge=False
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=${rollout_name}
    actor_rollout_ref.rollout.mode=${rollout_mode}
    actor_rollout_ref.rollout.gpu_memory_utilization=0.70
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp}
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length))
    actor_rollout_ref.rollout.temperature=${temperature}
    actor_rollout_ref.rollout.top_p=${top_p}
    actor_rollout_ref.rollout.top_k=${top_k}
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature}
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p}
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k}
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.rollout.val_kwargs.n=1
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.n=${n_resp_per_prompt}
)

TRAINER=(
    trainer.logger=['console','wandb']
    trainer.project_name="${project_name}"
    trainer.experiment_name="${exp_name}"
    trainer.n_gpus_per_node=8
    trainer.nnodes="${NNODES}"
    trainer.val_before_train=False
    trainer.test_freq=10
    trainer.save_freq=-1
    trainer.total_epochs=10
    trainer.default_local_dir="${CKPTS_DIR}"
    trainer.resume_mode=auto
    trainer.log_val_generations=10
)

FORWARD_ONLY_SETS=(
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz}
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len}
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len}
)

MODEL=(
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.model.trust_remote_code=True
)

ALGORITHM=(
    algorithm.adv_estimator=${adv_estimator}
    algorithm.use_kl_in_reward=${use_kl_in_reward}
    algorithm.kl_ctrl.kl_coef=${kl_coef}
)
################################################### start script ###################################################

python3 -m verl.trainer.main_ppo \
    --config-path=config \
    --config-name='ppo_megatron_trainer.yaml' \
    "${DATA[@]}" \
    "${ALGORITHM[@]}" \
    "${MODEL[@]}" \
    "${ROLLOUT[@]}" \
    "${ACTOR[@]}" \
    "${REWARD_MODEL[@]}" \
    "${FP8[@]}" \
    "${PERF_OPT[@]}" \
    "${TRAINER[@]}" \
    "${FORWARD_ONLY_SETS[@]}" \