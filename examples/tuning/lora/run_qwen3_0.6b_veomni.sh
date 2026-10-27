#!/usr/bin/env bash
# Qwen3 Dense native LoRA + VeOmni actor + merged-weight vLLM rollout.
# Requires matching VeOmni checkout with VeOmniLoraModel.disable_adapter().
set -euo pipefail

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-0.6B}
TRAIN_FILES=${TRAIN_FILES:-${HOME}/data/gsm8k/train.parquet}
VAL_FILES=${VAL_FILES:-${HOME}/data/gsm8k/test.parquet}
NUM_GPUS=${NUM_GPUS:-2}
OUTPUT_DIR=${OUTPUT_DIR:-checkpoints/qwen3-0.6b-veomni-lora}
if [ "$NUM_GPUS" = 1 ]; then
    INIT_DEVICE=${INIT_DEVICE:-cuda}
else
    INIT_DEVICE=${INIT_DEVICE:-meta}
fi

python3 -m verl.trainer.main_ppo \
    ray_kwargs.ray_init.runtime_env.py_executable=null \
    model_engine=veomni \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN_FILES" \
    data.val_files="$VAL_FILES" \
    data.train_batch_size=8 \
    data.max_prompt_length=256 \
    data.max_response_length=256 \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.lora_rank=8 \
    actor_rollout_ref.model.lora_alpha=16 \
    actor_rollout_ref.model.target_modules='[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]' \
    actor_rollout_ref.model.lora.merge=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-4 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.veomni.fsdp_size="$NUM_GPUS" \
    actor_rollout_ref.actor.veomni.init_device="$INIT_DEVICE" \
    actor_rollout_ref.actor.veomni.ulysses_parallel_size=1 \
    actor_rollout_ref.actor.veomni.expert_parallel_size=1 \
    actor_rollout_ref.actor.veomni.param_offload=False \
    actor_rollout_ref.actor.veomni.optimizer_offload=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.max_model_len=512 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    trainer.logger=console \
    trainer.project_name=veomni-dense-lora \
    trainer.experiment_name=qwen3-0.6b-grpo \
    trainer.n_gpus_per_node="$NUM_GPUS" \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.save_freq=2 \
    trainer.test_freq=-1 \
    trainer.total_training_steps=4 \
    trainer.default_local_dir="$OUTPUT_DIR" \
    "$@"
