set -x
ENGINE=${1:-vllm}

export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"
export VLLM_USE_V1=1
export VLLM_VERSION=0.13.0
export VLLM_ASCEND_ENABLE_NZ=0
export HCCL_BUFFSIZE=610
export CKPT_DIR="./ckpt30b"

TRAIN_FILE=dapo-math-17k.parquet
TEST_FILE=aime-2024.parquet
max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 8))
rollout_max_num_seqs=$((128))


python3 -m verl.trainer.main_ppo \
    model_engine=veomni \
    algorithm.adv_estimator=grpo \
    data.train_files=${TRAIN_FILE} \
    data.val_files=${TEST_FILE} \
    data.train_batch_size=16 \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=False \
    data.truncation='error' \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_cache_gb=0 \
    actor_rollout_ref.model.path=/Qwen3-30B-MoE-merge \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.veomni.param_offload=True \
    actor_rollout_ref.actor.veomni.optimizer_offload=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.veomni.fsdp_size=-1 \
    actor_rollout_ref.actor.veomni.expert_parallel_size=1 \
    actor_rollout_ref.ref.veomni.param_offload=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.data_parallel_size=8 \
    actor_rollout_ref.rollout.expert_parallel_size=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enable_prefix_caching=True \
    actor_rollout_ref.rollout.max_model_len=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((1024)) \
    actor_rollout_ref.rollout.max_num_seqs=${rollout_max_num_seqs} \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.veomni.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.use_legacy_worker_impl=disable \
    trainer.critic_warmup=0 \
    trainer.logger=console \
    trainer.project_name='verl_qwen3_veomni' \
    trainer.experiment_name='qwen3_30b_veomni' \
    trainer.n_gpus_per_node=16 \
    trainer.nnodes=1 \
    trainer.device=npu \
    trainer.save_freq=100 \
    trainer.default_local_dir=$CKPT_DIR \
    trainer.test_freq=-1 \
    trainer.total_training_steps=100 \
    2>&1 | tee "logs/veomni-30b_$(date +%Y%m%d_%H%M).log"
