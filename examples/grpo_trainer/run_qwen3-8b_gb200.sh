# Tested on GB200 NVL4 (1 node, 4x B200 192GB, aarch64)
# Supports both SGLang and vLLM rollout backends.
# Based on run_qwen3-8b.sh adapted for GB200.
#
# Key GB200-specific settings vs the standard script:
#   - enforce_eager=True (required for Blackwell)
#   - ray_kwargs.ray_init.num_gpus=N (Docker --privileged bypasses GPU auto-detection)
#   - fsdp_config.model_dtype=bfloat16 (FSDP actor defaults to fp32, breaks FlashAttn)
#   - SGLang only: attention_backend=flashinfer (FA3 unsupported on SM>90)

set -x

NNODES=${NNODES:-1}
NGPUS_PER_NODES=${NGPUS_PER_NODES:-4}

rollout_name="sglang" # sglang or vllm

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.train_batch_size=1024 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=Qwen/Qwen3-8B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.name=${rollout_name} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${NGPUS_PER_NODES} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    $([ "${rollout_name}" = "sglang" ] && echo "+actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=flashinfer") \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_grpo_example_gsm8k' \
    trainer.experiment_name='qwen3_8b_function_rm_gb200' \
    trainer.n_gpus_per_node=${NGPUS_PER_NODES} \
    +ray_kwargs.ray_init.num_gpus=${NGPUS_PER_NODES} \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 $@
