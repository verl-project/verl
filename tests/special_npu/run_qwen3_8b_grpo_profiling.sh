set -x

# Some models are optimized by vllm ascend. While in some case, e.g. rlhf training, 
# the optimized model may not be suitable. In this case, set this value to 0 to disable the optimized model.

MODEL_ID=${MODEL_ID:-Qwen/Qwen3-8B}
MODEL_PATH=${MODEL_PATH:-${HOME}/.cache/models/${MODEL_ID}}
SAVE_PATH=${SAVE_PATH:-./profiler_data-dis}
rm -rf "$SAVE_PATH"

LEVEL=${LEVEL:-"level0"}
CONTENTS=${CONTENTS:-['npu','cpu']}
ANALYSIS=${ANALYSIS:-True}
PROFILE_STEPS=${PROFILE_STEPS:-[1]}
PROFILE_RANKS_ALL=${PROFILE_RANKS_ALL:-False}
PROFILE_RANKS=${PROFILE_RANKS:-[2,4]}
DISCRETE=${DISCRETE:-True}
TRAIN_FILE=${TRAIN_FILE:-${HOME}/data/gsm8k/train.parquet}
TEST_FILE=${TEST_FILE:-${HOME}/data/gsm8k/test.parquet}
total_training_steps=${total_training_steps:-1}

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.train_batch_size=32 \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    data.shuffle=False \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=2 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    actor_rollout_ref.rollout.max_num_batched_tokens=4000 \
    actor_rollout_ref.rollout.max_num_seqs=64 \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path="${MODEL_PATH}" \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=1 \
    critic.ulysses_sequence_parallel_size=2 \
    critic.fsdp.param_offload=True \
    critic.fsdp.optimizer_offload=True \
    critic.use_dynamic_bsz=True \
    trainer.critic_warmup=0 \
    trainer.logger=console \
    trainer.project_name='verl_example_ppo_gsm8k' \
    trainer.experiment_name='qwen3_8b_fsdp' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.val_before_train=False \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.max_critic_ckpt_to_keep=1 \
    trainer.total_training_steps="$total_training_steps" \
    actor_rollout_ref.actor.profiler.enable=True \
    actor_rollout_ref.actor.profiler.all_ranks="$PROFILE_RANKS_ALL" \
    actor_rollout_ref.actor.profiler.ranks="$PROFILE_RANKS" \
    actor_rollout_ref.actor.profiler.tool_config.npu.discrete="$DISCRETE" \
    actor_rollout_ref.actor.profiler.tool_config.npu.contents="$CONTENTS" \
    actor_rollout_ref.actor.profiler.tool_config.npu.level="$LEVEL" \
    actor_rollout_ref.actor.profiler.tool_config.npu.analysis="$ANALYSIS" \
    actor_rollout_ref.ref.profiler.enable=True \
    actor_rollout_ref.ref.profiler.all_ranks="$PROFILE_RANKS_ALL" \
    actor_rollout_ref.ref.profiler.ranks="$PROFILE_RANKS" \
    actor_rollout_ref.ref.profiler.tool_config.npu.discrete="$DISCRETE" \
    actor_rollout_ref.ref.profiler.tool_config.npu.contents="$CONTENTS" \
    actor_rollout_ref.ref.profiler.tool_config.npu.level="$LEVEL" \
    actor_rollout_ref.ref.profiler.tool_config.npu.analysis="$ANALYSIS" \
    global_profiler.tool=npu \
    global_profiler.steps="$PROFILE_STEPS" \
    global_profiler.save_path="$SAVE_PATH" $@

