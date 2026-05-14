set -x

# ===================================== Environment & Paths =====================================
export CUDA_DEVICE_MAX_CONNECTIONS=1  # For megatron communication/computation overlapping

MODEL_PATH=/data/models/Qwen3-VL-30B-A3B-Instruct
TRAIN_FILE=$HOME/data/geo3k/train.parquet
TEST_FILE=$HOME/data/geo3k/test.parquet
CKPTS_DIR=/ckpts

# ===================================== Rollout Mode =====================================
rollout_mode="async"
rollout_name="vllm"  # sglang or vllm

return_raw_chat="False"
if [ "$rollout_mode" = "async" ]; then
    export VLLM_USE_V1=1
    return_raw_chat="True"
fi

# ===================================== GPU Allocation =====================================
n_gpus_rollout=16
n_gpus_training=16
n_nodes_rollout=2
n_nodes_train=2

# ===================================== Data =====================================
train_prompt_bsz=0
gen_prompt_bsz=1
n_resp_per_prompt=16
max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 8))

DATA_CONFIG="
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.val_batch_size=512 \
    data.return_raw_chat=${return_raw_chat}"

# ===================================== Actor Model & Optim =====================================
use_kl_loss=False
kl_loss_coef=0.0
clip_ratio_low=0.2
clip_ratio_high=0.28
use_dynamic_bsz=True
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) / 4))
train_prompt_mini_bsz=32
actor_offload=False
loss_agg_mode="token-mean"
sp_size=8 
fsdp_size=$((n_gpus_training * n_nodes_train)) 

ACTOR_CONFIG="
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${actor_offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${actor_offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${fsdp_size} \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.fsdp_config.use_torch_compile=False \
    critic.strategy=fsdp2 \
    actor_rollout_ref.nccl_timeout=7200 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.hybrid_engine=False \
    +actor_rollout_ref.model.override_config.max_position_embeddings=32768 \
    actor_rollout_ref.model.path=${MODEL_PATH} "

# ===================================== Ref Config =====================================
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) / 4))

REF_CONFIG="
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.ref.use_torch_compile=False \
    actor_rollout_ref.ref.fsdp_config.use_torch_compile=False"

# ===================================== Rollout Config =====================================
gen_tp=2
enforce_eager=False
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
val_top_p=0.7

ROLLOUT_CONFIG="
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.80 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.expert_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.enforce_eager=${enforce_eager} \
    +actor_rollout_ref.rollout.enable_sleep_mode=False \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.name=${rollout_name} \
    actor_rollout_ref.rollout.mode=${rollout_mode} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len}"

# ===================================== Algorithm =====================================
use_kl_in_reward=False
adv_estimator=grpo
kl_coef=0.0

ALGORITHM_CONFIG="
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    algorithm.rollout_correction.bypass_mode=False"

# ===================================== Trainer =====================================
total_epochs=200
test_freq=5
total_rollout_steps=$(( 512 * 100 ))

TRAINER_CONFIG="
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_grpo_example_geo3k_fully_async' \
    trainer.experiment_name='910B-Fsdp2-tp2sp8-async' \
    trainer.val_before_train=False \
    trainer.test_freq=${test_freq} \
    trainer.default_local_dir=${CKPTS_DIR} \
    trainer.resume_mode=disable \
    trainer.nnodes=${n_nodes_train} \
    trainer.n_gpus_per_node=${n_gpus_training} \
    trainer.save_freq=5 \
    rollout.nnodes=${n_nodes_rollout} \
    rollout.n_gpus_per_node=${n_gpus_rollout} \
    rollout.total_rollout_steps=${total_rollout_steps} "

# ===================================== Reward Model =====================================
enable_overlong_buffer=True
overlong_buffer_len=$((1024 * 4))
overlong_penalty_factor=1.0

REWARD_MODEL_CONFIG="
    reward_model.reward_manager=dapo \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward_model.reward_kwargs.max_resp_len=${max_response_length}"

# ===================================== Async Training =====================================
staleness_threshold=0.6 
trigger_parameter_sync_step=16
require_batches=1
partial_rollout=True

ASYNC_CONFIG="
    async_training.staleness_threshold=${staleness_threshold} \
    async_training.trigger_parameter_sync_step=${trigger_parameter_sync_step} \
    async_training.partial_rollout=${partial_rollout}"

# ===================================== Launch =====================================
python3 -m verl.experimental.fully_async_policy.fully_async_main \
    --config-path=config \
    --config-name='fully_async_ppo_trainer.yaml' \
    $DATA_CONFIG \
    $ACTOR_CONFIG \
    $REF_CONFIG \
    $ROLLOUT_CONFIG \
    $ALGORITHM_CONFIG \
    $TRAINER_CONFIG \
    $REWARD_MODEL_CONFIG \
    $ASYNC_CONFIG