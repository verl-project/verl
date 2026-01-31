set -x
libero_train_path=$HOME/data/libero_rl/train.parquet
libero_test_path=$HOME/data/libero_rl/test.parquet

train_files=$libero_train_path
test_files=$libero_test_path

OUTPUT_DIR=${MLP_MODEL_OUTPUT:-"$HOME/models/vla_libero_grpo"}
VIDEO_OUTPUT=${MLP_MODEL_OUTPUT:-"$HOME"}/video
SFT_MODEL_PATH=${SFT_MODEL_PATH:-"$HOME/data/pi05_libero_torch"}
TOKENIZER_PATH="$SFT_MODEL_PATH"

# Physical Node Config
NUM_NODES=1                                    # number of nodes
NUM_GPUS=4                                     # total number of gpus per node

# Role Config
NUM_ENV_GPUS=2                                 # number of gpus for env workers per node
NUM_ROLLOUT_GPUS=$((NUM_GPUS - NUM_ENV_GPUS))  # number of gpus for rollout workers per node

# Rollout Config
# NOTE: TRAIN_BATCH_SIZE * ROLLOUT_N == NUM_ENV_GPUS * NUM_STAGE * NUM_ENV
TRAIN_BATCH_SIZE=32                            # batch size for dataloaders per step
ROLLOUT_N=1                                    # response number for each prompt (for GRPO)
NUM_STAGE=2                                    # number of pipeline stages
NUM_ENV=8                                      # number of envs per env worker

NUM_ACTION_CHUNKS=10                           # number of action chunks
MAX_EPISODE_STEPS=512                          # max episode steps for each env
                                               # max_interactions = MAX_EPISODE_STEPS / num_action_chunks

# Training Config
MINI_BATCH_SIZE=128                            # mini batch size (batch size per GPU, automatically multiplied by ROLLOUT_N)
                                               # invalid in SAC, currently
                                               # In SAC, it equal to (max_interactions - 1) * TRAIN_BATCH_SIZE * ROLLOUT_N / NUM_ROLLOUT_GPUS
MICRO_BATCH_SIZE=8                             # micro batch size (per GPU, for gradient accumulation, should divide MINI_BATCH_SIZE)



# isaac or libero
# libero means original libero benchmark with mujoco sim
# isaac means libero benchmark using isaac sim
SIM_TYPE=${SIM_TYPE:-"libero"}
PROJECT_NAME="vla_libero_RL"
EXPERIMENT_NAME="${SIM_TYPE}_reinforce_plus_plus"

ISSC_PYTHON="/workspace/isaaclab/_isaac_sim/python.sh"
PYTHON=python
if [ -f "$ISSC_PYTHON" ]; then
    PYTHON=$ISSC_PYTHON
fi

# avoiding warnings
mkdir /root/LIBERO/libero/libero/../datasets
gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -n 1)

# force osmesa in Hopper
if echo "$gpu_name" | grep "NVIDIA H"; then
    echo "enable MUJOCO_GL=osmesa in Hopper"
    export MUJOCO_GL=osmesa
fi

export VERL_LOGGING_LEVEL=INFO

$PYTHON -m verl.experimental.vla.main_sac \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.val_batch_size=4 \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    env.train.num_envs=$NUM_ENV \
    data.max_prompt_length=256 \
    data.max_response_length=128 \
    env.rollout.pipeline_stage_num=$NUM_STAGE \
    env.train.simulator_type=$SIM_TYPE \
    env.actor.model.num_action_chunks=$NUM_ACTION_CHUNKS \
    env.actor.model.action_dim=7 \
    env.train.only_eval=False \
    env.train.max_episode_steps=$MAX_EPISODE_STEPS \
    env.train.video_cfg.save_video=True \
    env.train.video_cfg.video_base_dir=${VIDEO_OUTPUT} \
    env.train.seed=42 \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[SiglipEncoderLayer,GemmaDecoderLayerWithExpert] \
    actor_rollout_ref.model.path=$SFT_MODEL_PATH \
    actor_rollout_ref.model.tokenizer_path=$TOKENIZER_PATH \
    actor_rollout_ref.rollout.mode=async_envloop \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.actor.optim.warmup_style=constant \
    actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.strategy=fsdp2 \
    critic.strategy=fsdp2 \
    actor_rollout_ref.actor.grad_clip=1 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.num_images_in_input=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.trust_remote_code=False \
    +actor_rollout_ref.model.override_config.attn_implementation=eager \
    actor_rollout_ref.actor.entropy_coeff=0. \
    actor_rollout_ref.rollout.temperature=1.6 \
    actor_rollout_ref.rollout.prompt_length=512 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=hf \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    +actor_rollout_ref.algorithm='sac' \
    algorithm.kl_ctrl.kl_coef=0.00 \
    trainer.logger=['console'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=$OUTPUT_DIR \
    trainer.n_gpus_per_node=$NUM_GPUS \
    +trainer.n_env_gpus_per_node=$NUM_ENV_GPUS \
    +trainer.n_rollout_gpus_per_node=$NUM_ROLLOUT_GPUS \
    trainer.nnodes=$NUM_NODES \
    trainer.save_freq=30 \
    trainer.test_freq=-1 \
    trainer.total_epochs=100 \
    trainer.val_only=False \
    trainer.val_before_train=False
