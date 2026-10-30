#!/bin/bash

# 定义变量
project_name="bootcamp_example_project"
experiment_name="default_experiment_name"
internbootcamp_path="/path/to/bootcamp"
actor_model="/path/to/actor_model"
verl_path="/path/to/verl"
export WANDB_API_KEY="your_wandb_api_key"

# 安装依赖
pip install -e "$internbootcamp_path"

# 设置环境变量
export VERL_PPO_LOGGING_LEVEL=DEBUG
# Do not use these in new version of Verl
# export HYDRA_FULL_ERROR=1
# export VLLM_ATTENTION_BACKEND=XFORMERS

# 定义文件路径数组
train_files=(
    "examples/bootcamp_generator_outputs/<time_stamp>_for_verl_merged/train/bootcamps.parquet"
)
test_files=(
    "examples/bootcamp_generator_outputs/<time_stamp>_for_verl_merged/test/bootcamps.parquet"
    "examples/bootcamp_generator_outputs/<time_stamp>_for_verl/test/aime.parquet"
    "examples/bootcamp_generator_outputs/<time_stamp>_for_verl/test/math.parquet"
)

# 构建 train_files 和 test_files 字符串
build_json_array() {
  local array=("$@")
  if [[ ${#array[@]} -eq 0 ]]; then
    echo "[]"
  else
    local result="["
    for item in "${array[@]}"; do
      result+="\"$item\","
    done
    result="${result%,}]"
    echo "$result"
  fi
}

train_files_str=$(build_json_array "${train_files[@]}")
test_files_str=$(build_json_array "${test_files[@]}")


# 调试模式
set -x

# 运行训练脚本
python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files_str" \
    data.val_files="$test_files_str" \
    +data.no_chat_template=False \
    data.train_batch_size=64 \
    data.val_batch_size=64 \
    data.truncation=right \
    data.max_prompt_length=4096 \
    data.max_response_length=8192 \
    actor_rollout_ref.model.path=$actor_model \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=$verl_path/ckpts/$experiment_name \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=20000 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.disable_log_stats=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.max_num_seqs=32 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.65 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.total_epochs=1 $@