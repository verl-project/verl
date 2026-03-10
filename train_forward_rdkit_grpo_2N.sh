
set -e

# 1. 设置Ray和其他环境变量
ulimit -n 65535
export RAY_DEDUP_LOGS=0

# 2. 定义模型和数据路径
ACTOR_MODEL_PATH="/mnt/shared-storage-user/mineru4s/dingruiyi/scripts_cot/output/0129b_v2_cot_with_True_think/v0-20260212-122544/checkpoint-0129b_v2_cot_with_True_think_10e"
FORWARD_MODEL_PATH="/mnt/shared-storage-user/mineru4s/dingruiyi/scripts_cot/output/0129b_smiles2smiles_forward/v0-20260213-221047/checkpoint-0129b_smiles2smiles_forward_10e"
VAL_DATA="/mnt/shared-storage-user/mineru4s/dingruiyi/WanjuanTraining/data/USPTO-50k_SL_COT_V1_test_data_rl.parquet"
TRAIN_DATA="/mnt/shared-storage-user/mineru4s/dingruiyi/USPTO_50k_train_SL_COT_rl.parquet"

# 3. 输出目录
OUTPUT_DIR="outputs/forward_rdkit_grpo_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"
cp "$0" "$OUTPUT_DIR/"

# 4. 训练参数
NUM_GPUS_PER_NODE=8
NNODES=2
TRAIN_BATCH_SIZE=1024
LEARNING_RATE=1e-5
TOTAL_EPOCHS=10

export TIKTOKEN_ENCODINGS_BASE=/root/encoder
export TIKTOKEN_RS_CACHE_DIR=/root/encoder

cd /mnt/shared-storage-user/mineru4s/dingruiyi/WanjuanTraining
# 5. 启动训练
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN_DATA" \
    data.val_files="$VAL_DATA" \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="$ACTOR_MODEL_PATH" \
    actor_rollout_ref.actor.optim.lr=${LEARNING_RATE} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.max_model_len=2048 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
    algorithm.use_kl_in_reward=False \
    reward.reward_manager.name=forward_rdkit \
    reward.reward_manager.source=register \
    reward.num_workers=8 \
    reward.reward_model.enable=False \
    +reward.forward_model.model_path="$FORWARD_MODEL_PATH" \
    +reward.forward_model.num_gpus=2 \
    +reward.beam_search_config.beam_width=3 \
    +reward.beam_search_config.max_tokens=512\
    +reward.beam_search_config.max_model_len=3176 \
    +reward.beam_search_config.gpu_memory_utilization=0.3 \
    +reward.beam_search_config.dtype=bfloat16 \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name='forward_rdkit_grpo' \
    trainer.experiment_name='qwen3.5_4b_forward_rdkit_grpo' \
    trainer.n_gpus_per_node=${NUM_GPUS_PER_NODE} \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=5 \
    trainer.test_freq=-1 \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.resume_mode=disable \
    trainer.default_local_dir="$OUTPUT_DIR" \
    trainer.val_before_train=False $@

echo "训练完成! 输出目录: $OUTPUT_DIR"

