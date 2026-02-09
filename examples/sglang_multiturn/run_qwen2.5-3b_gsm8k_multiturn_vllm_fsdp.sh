# run on Ascend 910
# make sure your current working directory is the root of the project

set -x
ulimit -n 65535

#set vllm v1 env
export VLLM_USE_V1=1

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"

TRAIN_BATCH_SIZE=32
MICRO_BATCH_SIZE=8

python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='gsm8k_multiturn_grpo' \
    actor_rollout_ref.rollout.name=vllm \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="Qwen/Qwen2.5-3B-Instruct" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${TRAIN_BATCH_SIZE} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9\
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.project_name='gsm8k_async_rl' \
    trainer.experiment_name='qwen2.5-3b_function_rm-gsm8k-sgl-multi-w-tool-verify-n16' \
    trainer.n_gpus_per_node=16 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=20 \
    trainer.logger='["console"]' \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    trainer.total_epochs=15 \
    actor_rollout_ref.rollout.trace.token2text=False \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.multi_turn.enable=true \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$PROJECT_DIR/examples/sglang_multiturn/config/tool_config/gsm8k_tool_config.yaml" \
    actor_rollout_ref.rollout.free_cache_engine=True