set -x

# profiling configuration
PROFILE_STEPS="[2,4]"
PROFILE_RANKS_ALL=False
DISCRETE=True
PROFILE_RANKS="[1,2]"

# profiling NPU options
SAVE_PATH="$HOME/profile_data"
LEVEL="level0"
CONTENTS=['npu','cpu']
ANALYSIS=True

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.train_batch_size=32 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.optim.lr=5e-8 \
    actor_rollout_ref.actor.ppo_mini_batch_size=2 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.profiler.enable=True \
    actor_rollout_ref.actor.profiler.ranks=$PROFILE_RANKS \
    actor_rollout_ref.actor.profiler.all_ranks=$PROFILE_RANKS_ALL \
    actor_rollout_ref.actor.profiler.tool_config.npu.discrete=$DISCRETE \
    actor_rollout_ref.actor.profiler.tool_config.npu.contents=$CONTENTS \
    actor_rollout_ref.actor.profiler.tool_config.npu.level=$LEVEL \
    actor_rollout_ref.actor.profiler.tool_config.npu.analysis=$ANALYSIS \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.profiler.enable=True \
    actor_rollout_ref.ref.profiler.ranks=$PROFILE_RANKS \
    actor_rollout_ref.ref.profiler.all_ranks=$PROFILE_RANKS_ALL \
    actor_rollout_ref.ref.profiler.tool_config.npu.discrete=$DISCRETE \
    actor_rollout_ref.ref.profiler.tool_config.npu.contents=$CONTENTS \
    actor_rollout_ref.ref.profiler.tool_config.npu.level=$LEVEL \
    actor_rollout_ref.ref.profiler.tool_config.npu.analysis=$ANALYSIS \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=console \
    trainer.project_name='verl_grpo_example_gsm8k' \
    trainer.experiment_name='qwen2_5_7b_function_rm' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=5 \
    global_profiler.tool=npu \
    global_profiler.steps=$PROFILE_STEPS \
    global_profiler.save_path=$SAVE_PATH
    $@