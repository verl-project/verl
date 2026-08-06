# ---- user-adjustable ----
TRAIN_FILE=${TRAIN_FILE:-data/full_hh_rlhf/rl/train.parquet}
TEST_FILE=${TEST_FILE:-data/full_hh_rlhf/rl/train.parquet} # no use
MODEL_PATH=${MODEL_PATH:-mistralai/Mistral-Nemo-Instruct-2407}
REWARD_MODEL_PATH=${REWARD_MODEL_PATH:-Skywork/Skywork-Reward-Llama-3.1-8B}
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-512}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-128}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-10}
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-10}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-4096}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}

ACTOR_LR=${ACTOR_LR:-1e-6}
ROLLOUT_TP=${ROLLOUT_TP:-4}
ROLLOUT_N=${ROLLOUT_N:-5}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.7}
REWARD_NUM_WORKERS=${REWARD_NUM_WORKERS:-8}
REWARD_ROLLOUT_GPU_MEM_UTIL=${REWARD_ROLLOUT_GPU_MEM_UTIL:-0.8}
REWARD_ROLLOUT_TP=${REWARD_ROLLOUT_TP:-1}
REWARD_PROMPT_LENGTH=${REWARD_PROMPT_LENGTH:-8192}
REWARD_RESPONSE_LENGTH=${REWARD_RESPONSE_LENGTH:-4096}

ADV_ESTIMATOR=${ADV_ESTIMATOR:-grpo}
PROJECT_NAME=${PROJECT_NAME:-verl_full_hh_rlhf_examples}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-grpo_mistral13B-skyworkLlama8b-hhrlhf}
SAVE_FREQ=${SAVE_FREQ:-10}
TEST_FREQ=${TEST_FREQ:--1}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-5}
# ---- end user-adjustable ----
########################### parameter arrays ###########################

DATA=(
    algorithm.adv_estimator=${ADV_ESTIMATOR}
    data.train_files="${TRAIN_FILE}"
    data.val_files="${TEST_FILE}"
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.prompt_key="prompt"
    data.return_raw_chat=True
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.filter_overlong_prompts=True
    data.truncation='error'
    algorithm.use_kl_in_reward=False
)

MODEL=(
    actor_rollout_ref.model.path=${MODEL_PATH}
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.actor.use_kl_loss=False
)

ROLLOUT=(
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}
)

REWARD=(
    reward.num_workers=${REWARD_NUM_WORKERS}
    reward.reward_model.enable=True
    reward.reward_model.model_path=${REWARD_MODEL_PATH}
    reward.reward_model.rollout.name=vllm
    reward.reward_model.rollout.gpu_memory_utilization=${REWARD_ROLLOUT_GPU_MEM_UTIL}
    reward.reward_model.rollout.tensor_model_parallel_size=${REWARD_ROLLOUT_TP}
    reward.reward_model.rollout.prompt_length=${REWARD_PROMPT_LENGTH}
    reward.reward_model.rollout.response_length=${REWARD_RESPONSE_LENGTH}
)

TRAINER=(
    trainer.logger='["console","wandb"]'
    trainer.val_before_train=False
    trainer.project_name=${PROJECT_NAME}
    trainer.experiment_name=${EXPERIMENT_NAME}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.save_freq=${SAVE_FREQ}
    trainer.test_freq=${TEST_FREQ}
    trainer.total_epochs=${TOTAL_EPOCHS}
)

EXTRA=(
)

########################### launch ###########################
python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REWARD[@]}" \
    "${TRAINER[@]}" \
    "${EXTRA[@]}" \
    "$@"