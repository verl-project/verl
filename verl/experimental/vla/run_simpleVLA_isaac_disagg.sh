#!/bin/bash
set -x

echo "remember to set ray param < --resources='{\"sim\"/\"actor_rollout\":1}' > if using disagg sim"

libero_train_path=$HOME/data/libero_rl/train.parquet
libero_test_path=$HOME/data/libero_rl/test.parquet

train_files=$libero_train_path
test_files=$libero_test_path

OUTPUT_DIR=${MLP_MODEL_OUTPUT:-"$HOME/models/vla_libero_grpo"}
VIDEO_OUTPUT=${MLP_MODEL_OUTPUT:-"$HOME"}/video
SFT_MODEL_PATH=${SFT_MODEL_PATH:-"$HOME/data/Openvla-oft-SFT-libero10-trajall"}

# for rollout and train
NUM_NODES=1
# for simulator
SIM_NODES=1
NUM_ENV_GPUS=8
NUM_ROLLOUT_GPUS=8
STAGE_NUM=2
BATCH_SIZE=16
# rollout.n should equal to num_envs for isaac env
ROLLOUT_N=8

# 512 is required for libero benchmark, but you can reduce it in debugging to run faster
MAX_EPISODE_STEPS=512

# isaac or libero
# libero means original libero benchmark with mujoco sim
# isaac means libero benchmark using isaac sim
SIM_TYPE=${SIM_TYPE:-"isaac"}
PROJECT_NAME="vla-disagg-issac"
EXPERIMENT_NAME="${SIM_TYPE}_rl"

ISSC_PYTHON="/workspace/isaaclab/_isaac_sim/python.sh"
PYTHON=python
if [ -f "$ISSC_PYTHON" ]; then
    PYTHON=$ISSC_PYTHON
fi

# avoiding warnings
mkdir /root/LIBERO/libero/libero/../datasets


$PYTHON -m verl.experimental.vla.main_ppo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=${BATCH_SIZE} \
    data.val_batch_size=${BATCH_SIZE} \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    env.train.num_envs=$ROLLOUT_N \
    data.max_prompt_length=256 \
    data.max_response_length=128 \
    env.rollout.pipeline_stage_num=$STAGE_NUM \
    env.train.simulator_type=$SIM_TYPE \
    env.actor.model.num_action_chunks=8 \
    env.actor.model.action_dim=7 \
    env.train.only_eval=False \
    env.train.max_episode_steps=$MAX_EPISODE_STEPS \
    env.train.video_cfg.save_video=True \
    env.train.video_cfg.video_base_dir=${VIDEO_OUTPUT} \
    env.train.seed=42 \
    env.disagg_sim.enable=True \
    env.disagg_sim.nnodes=$SIM_NODES \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.model.path=$SFT_MODEL_PATH \
    actor_rollout_ref.rollout.mode=async_envloop \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.actor.optim.warmup_style=constant \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.grad_clip=1 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.num_images_in_input=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.trust_remote_code=False \
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
    algorithm.kl_ctrl.kl_coef=0.00 \
    trainer.logger=['console'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=$OUTPUT_DIR \
    trainer.n_gpus_per_node=$NUM_ROLLOUT_GPUS \
    +trainer.n_env_gpus_per_node=$NUM_ENV_GPUS \
    +trainer.n_rollout_gpus_per_node=$NUM_ROLLOUT_GPUS \
    trainer.nnodes=$NUM_NODES \
    trainer.save_freq=30 \
    trainer.test_freq=-1 \
    trainer.total_epochs=20 \
    trainer.val_only=False \
    trainer.total_training_steps=10000 \
    algorithm.adv_estimator=reinforce_plus_plus \
    trainer.val_before_train=False  $@


