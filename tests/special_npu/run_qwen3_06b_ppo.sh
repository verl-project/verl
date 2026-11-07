set -x

MODEL_ID=${MODEL_ID:-Qwen/Qwen3-0.6B}
MODEL_PATH=${MODEL_PATH:-${HOME}/.cache/modelscope/hub/models/${MODEL_ID}}

# Override these on smaller nodes, e.g. a single A2 card. The default rollout
# memory utilization (0.8) OOMs on a single card once cudagraph capture starts;
# 0.5 is verified there:
# NDEVICES_PER_NODE=1 SP_SIZE=1 ROLLOUT_GPU_MEM_UTIL=0.5 bash tests/special_npu/run_qwen3_06b_ppo.sh
NDEVICES_PER_NODE=${NDEVICES_PER_NODE:-8}
SP_SIZE=${SP_SIZE:-2}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.8}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-2}

# train_batch_size must be a multiple of the data-parallel size
# (NDEVICES_PER_NODE / SP_SIZE): the trainer asserts equal-size chunks per
# rank. Floor the default 16 accordingly (16 for all standard 8/16-device
# setups).
dp_size=$(( NDEVICES_PER_NODE / SP_SIZE ))
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-$(( (16 / dp_size) * dp_size ))}

SKIP_ENABLE=True
SKIP_DUMP_DIR=${SKIP_DUMP_DIR:-${HOME}/data/rollout_dump_sync}
SKIP_STEPS='[1]'
SKIP_ACTION=cache

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=512 \
    data.max_response_length=128 \
    data.shuffle=False \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${TRAIN_BATCH_SIZE} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${SP_SIZE} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode="FULL_AND_PIECEWISE" \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.weight_nz_mode=0 \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path="${MODEL_PATH}" \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=1 \
    critic.ulysses_sequence_parallel_size=${SP_SIZE} \
    critic.fsdp.param_offload=True \
    critic.fsdp.optimizer_offload=True \
    critic.use_dynamic_bsz=True \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name='verl_ppo_example_gsm8k_qwen3' \
    trainer.experiment_name='qwen3_06b_fsdp' \
    trainer.n_gpus_per_node=${NDEVICES_PER_NODE} \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=${TOTAL_TRAINING_STEPS} \
    skip.rollout.enable=${SKIP_ENABLE} \
    skip.rollout.dump_dir=${SKIP_DUMP_DIR} \
    skip.rollout.steps=${SKIP_STEPS} \
    skip.rollout.action=${SKIP_ACTION} \
    "$@"
