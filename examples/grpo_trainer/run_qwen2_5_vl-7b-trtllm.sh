set -x

# Clean all slurm / MPI / PMIx env to avoid pmix mismatch error
for v in $(env | awk -F= '/^(PMI|PMIX|MPI|OMPI|SLURM)_/{print $1}'); do
    unset "$v"
done

export RAY_DEDUP_LOGS=0

# -----
# Config
# -----
TP=${1:-4}
PROJECT_NAME=${PROJECT_NAME:-"verl_grpo_example_gsm8k_math"}
EXP_NAME=trtllm-qwen2-7b-tp${TP}-8gpus${EXP_NAME_SUFFIX:+"-"}${EXP_NAME_SUFFIX}

if [ $TP -eq 4 ]; then
    MAX_BATCH_SIZE=1024
else
    MAX_BATCH_SIZE=384
fi

# -----
# Data
# -----
DATADIR=${DATADIR:-$PWD/data}
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen2.5-VL-7B-Instruct"}

GEO3K_TRAIN_PATH=${DATADIR}/geo3k/train.parquet
GEO3K_TEST_PATH=${DATADIR}/geo3k/test.parquet
TRAIN_FILES="['$GEO3K_TRAIN_PATH']"
TEST_FILES="['$GEO3K_TEST_PATH']"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.rollout_correction.rollout_is_threshold=2.0 \
    data.train_files="$TRAIN_FILES" \
    data.val_files="$TEST_FILES" \
    data.train_batch_size=512 \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.trust_remote_code=True \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    +actor_rollout_ref.model.override_config.attn_implementation=eager \
    +actor_rollout_ref.ref.model.override_config.attn_implementation=eager \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${TP} \
    actor_rollout_ref.rollout.name=trtllm \
    actor_rollout_ref.rollout.mode="async" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.max_num_seqs=${MAX_BATCH_SIZE} \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    +actor_rollout_ref.rollout.engine_kwargs.trtllm.batch_wait_timeout_iters=32 \
    +actor_rollout_ref.rollout.engine_kwargs.trtllm.batch_wait_max_tokens_ratio=0.5 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name=${EXP_NAME} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.resume_mode=disable \
    trainer.total_epochs=10 $@