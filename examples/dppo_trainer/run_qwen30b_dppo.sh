# run Qwen3-30B-A3B-Base on dapo-math-17k dataset
set -x

# ================================ DPPO Specific Parameters ===========================

# Why from GRPO to DPPO?
"""
The ratio clipping mechanism in GRPO/PPO is structurally ill-suited due to the large, 
long-tailed vocabularies inherent to LLMs. It over-penalizes low-probability tokens 
and under-penalizes high-probability ones, leading to training inefficiency and 
instability. For example, increasing a rare token’s probability from 1e−5 to 1e−3 
generates a massive ratio of 100 that triggers clipping, even though the actual 
divergence is negligible. Conversely, small ratio changes on high-probability tokens 
can make catastrophic shifts in probability mass (e.g., a drop from 0.99 to 0.8), yet 
it often remains unpenalized by the clipping mechanism.

DPPO addresses this issue by using a divergence-based clipping mechanism, achieving 
superior training stability and final performance compared to existing methods.

DPPO paper: https://arxiv.org/pdf/2602.04879
"""

LOSS_MODE=${LOSS_MODE:-"dppo_tv"}

if [[ $LOSS_MODE == "dppo_kl" ]]; then
    # The KL divergence threshold for DPPO.
    clip_ratio=0.05
    clip_ratio_low=${CLIP_LOW:-0.05}
    clip_ratio_high=${CLIP_HIGH:-0.05}
elif [[ $LOSS_MODE == "dppo_tv" ]]; then
    # The TV divergence threshold for DPPO.
    clip_ratio=0.15
    clip_ratio_low=${CLIP_LOW:-0.15}
    clip_ratio_high=${CLIP_HIGH:-0.15}
elif [[ $LOSS_MODE == "vanilla" ]]; then
    # GRPO baseline
    clip_ratio=0.2
    clip_ratio_low=${CLIP_LOW:-0.2}
    clip_ratio_high=${CLIP_HIGH:-0.28}
else
    echo "Invalid loss mode: $LOSS_MODE"
    exit 1
fi

# Disable dual-clip PPO and TIS for a fair comparison between GRPO and DPPO.
clip_ratio_c=10000.0

# ===================================== Algorithm =====================================
adv_estimator=grpo

# We recommand directly clipping the ratio/divergence with respect to the original 
# rollout policy (implemented by bypass_mode=True), instead of the recomputed one. 
# This can not only save the computation cost, but also improve the training stability 
# for both GRPO and DPPO by controlling the training-inference mismatch at a low level.
# See Section 5.2 in https://arxiv.org/pdf/2602.04879 for more details.
bypass_mode=True

# We recommand using Dr.GRPO to remove the length and difficulty bias in original GRPO.
# See Section 3.1 in https://arxiv.org/pdf/2503.20783 for more details.
norm_adv_by_std_in_grpo=False               # remove the difficulty bias
loss_agg_mode="seq-mean-token-sum-norm"     # remove the length bias

# reference policy
use_kl_in_reward=False
kl_coef=0.001
use_kl_loss=False
kl_loss_coef=0.001

actor_lr=1e-6
critic_lr=2e-6
gae_gamma=1.0
gae_lam=0.95
critic_warmup=0


# ================================== Data/Model/Config =================================

# Node Info
NNODES=${NNODES:-2}

# wandb
backend=megatron # fsdp, fsdp2, megatron
project_name=Qwen3-30B-A3B-Base-dapo-math-17k
experiment_name="${backend}-${NNODES}nodes-${LOSS_MODE}-low${clip_ratio_low}-high${clip_ratio_high}"

# Paths
DATA_ROOT=${DATA_ROOT:-"${HOME}/verl"}
CKPTS_DIR=${CKPTS_DIR:-"${DATA_ROOT}/ckpts/${project_name}/${experiment_name}"}
MODEL_PATH=${MODEL_PATH:-"${DATA_ROOT}/models/Qwen3-30B-A3B-Base"}
TRAIN_FILE=${TRAIN_FILE:-"${DATA_ROOT}/data/dapo-math-17k.parquet"}
TEST_FILE=${TEST_FILE:-"${DATA_ROOT}/data/aime-2024.parquet"}


actor_model_path=$MODEL_PATH
critic_model_path=$MODEL_PATH

max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 8))
enable_overlong_buffer=False
overlong_buffer_len=$((1024 * 4))
overlong_penalty_factor=1.0

train_batch_size=256
ppo_mini_batch_size=32
n_resp_per_prompt=16
n_resp_per_prompt_val=1

# ===================================== Training ======================================
actor_max_token_len_per_gpu=$(((max_prompt_length + max_response_length) * 1))
critic_max_token_len_per_gpu=$(((max_prompt_length + max_response_length) * 1))

# FSDP parallelism config
USP_SIZE=4
ACTOR_FSDP_CONFIG="
    actor_rollout_ref.actor.fsdp_config.strategy=$backend \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$USP_SIZE"

# Megatron parallelism config
TP_SIZE=2
CP_SIZE=1
PP_SIZE=1
VPP_SIZE=null
EP_SIZE=8
ETP_SIZE=1
ACTOR_MEGATRON_CONFIG="
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=$TP_SIZE \
    actor_rollout_ref.actor.megatron.context_parallel_size=$CP_SIZE \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=$PP_SIZE \
    actor_rollout_ref.actor.megatron.virtual_pipeline_model_parallel_size=$VPP_SIZE \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=$EP_SIZE \
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=$ETP_SIZE \
    actor_rollout_ref.actor.megatron.param_offload=True \
    actor_rollout_ref.actor.megatron.grad_offload=True \
    actor_rollout_ref.actor.megatron.optimizer_offload=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype=fp32 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=True \
    actor_rollout_ref.actor.megatron.use_mbridge=True"

# Actor model config
ACTOR_CONFIG="
    actor_rollout_ref.actor.optim.lr=$actor_lr \
    actor_rollout_ref.model.path=$actor_model_path \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_kl_loss=$use_kl_loss \
    actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
    actor_rollout_ref.actor.clip_ratio=$clip_ratio \
    actor_rollout_ref.actor.clip_ratio_low=$clip_ratio_low \
    actor_rollout_ref.actor.clip_ratio_high=$clip_ratio_high \
    actor_rollout_ref.actor.clip_ratio_c=$clip_ratio_c \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.calculate_entropy=True \
    actor_rollout_ref.actor.policy_loss.loss_mode=${LOSS_MODE} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$actor_max_token_len_per_gpu"

# Critic model config
CIRITC_CONFIG="
    critic.optim.lr=$critic_lr \
    critic.model.path=$critic_model_path \
    critic.model.use_remove_padding=True \
    critic.ppo_max_token_len_per_gpu=$critic_max_token_len_per_gpu \
    critic.ulysses_sequence_parallel_size=$USP_SIZE"

CRITIC_FSDP_CONFIG="${ACTOR_FSDP_CONFIG//actor_rollout_ref.actor/critic.model}"
CRITIC_MEGATRON_CONFIG="${ACTOR_MEGATRON_CONFIG//actor_rollout_ref.actor/critic}"

if [[ $backend == "megatron" ]]; then
    CONFIG_NAME=ppo_megatron_trainer
    ACTOR_CONFIG="$ACTOR_CONFIG $ACTOR_MEGATRON_CONFIG"
    if [[ $adv_estimator == "gae" ]]; then
        CIRITC_CONFIG="$CIRITC_CONFIG $CRITIC_MEGATRON_CONFIG"
    else
        CIRITC_CONFIG=""
    fi
else # fsdp, fsdp2
    CONFIG_NAME=ppo_trainer
    ACTOR_CONFIG="$ACTOR_CONFIG $ACTOR_FSDP_CONFIG"
    if [[ $adv_estimator == "gae" ]]; then
        CIRITC_CONFIG="$CIRITC_CONFIG $CRITIC_FSDP_CONFIG"
    else
        CIRITC_CONFIG=""
    fi
fi

# ===================================== Inference =====================================
rollout_name=vllm
if [ "$rollout_name" = "vllm" ]; then
    export VLLM_USE_V1=1
fi
infer_tp=4
infer_dp=1
infer_ep=1
gpu_memory_utilization=0.7

ROLLOUT_CONFIG="
    actor_rollout_ref.rollout.name=$rollout_name \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$infer_tp \
    actor_rollout_ref.rollout.data_parallel_size=$infer_dp \
    actor_rollout_ref.rollout.expert_parallel_size=$infer_ep \
    actor_rollout_ref.rollout.gpu_memory_utilization=$gpu_memory_utilization \
    actor_rollout_ref.rollout.n=$n_resp_per_prompt \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n=$n_resp_per_prompt_val"

# ===================================== Reward =====================================
REWARD_CONFIG="
    reward.reward_manager.name=dapo \
    +reward.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
    +reward.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
    +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
    +reward.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward.reward_kwargs.max_resp_len=${max_response_length}"

python3 -m verl.trainer.main_ppo \
    --config-path=./config \
    --config-name=$CONFIG_NAME \
    algorithm.adv_estimator=$adv_estimator \
    algorithm.use_kl_in_reward=$use_kl_in_reward \
    algorithm.kl_ctrl.kl_coef=$kl_coef \
    algorithm.gamma=$gae_gamma \
    algorithm.lam=$gae_lam \
    algorithm.rollout_correction.bypass_mode=$bypass_mode \
    algorithm.norm_adv_by_std_in_grpo=$norm_adv_by_std_in_grpo \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$TEST_FILE" \
    data.return_raw_chat=True \
    data.train_batch_size=$train_batch_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=False \
    data.filter_overlong_prompts_workers=64 \
    data.truncation='error' \
    trainer.use_legacy_worker_impl=disable \
    trainer.critic_warmup=$critic_warmup \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.default_local_dir=$CKPTS_DIR \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=$NNODES \
    trainer.val_before_train=False \
    trainer.log_val_generations=100 \
    trainer.save_freq=-1 \
    trainer.test_freq=10 \
    trainer.total_epochs=10 \
    trainer.total_training_steps=500 \
    $ACTOR_CONFIG \
    $CIRITC_CONFIG \
    $ROLLOUT_CONFIG \
    $REWARD_CONFIG
