#!/usr/bin/env bash
set -euo pipefail

# Run in the prepared uv environment on an existing two-node Ray cluster.
# Both engines use the same BF16 checkpoint derived from the FP8 source.
# Model-specific Transformers/vLLM support must be installed separately.
# Set VERL_NO_PLACEMENT_MMAP_DIR on Ray workers to node-local disk shared by that node's ranks.
: "${MODEL_PATH:?set the shared BF16 checkpoint}"
: "${TRAIN_FILE:?set the DAPO-Math training parquet}"
: "${VAL_FILE:?set the held-out AIME2024 parquet}"
: "${OUTPUT_DIR:?set the checkpoint directory}"

exec uv run --active --no-sync python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="['${TRAIN_FILE}']" \
    data.val_files="['${VAL_FILE}']" \
    data.train_batch_size=8 \
    data.max_prompt_length=4096 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.trust_remote_code=True \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_liger=False \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.model.lora_rank=16 \
    actor_rollout_ref.model.lora_alpha=32 \
    'actor_rollout_ref.model.target_modules=[q_proj,k_proj,v_proj,o_proj,in_proj_a,in_proj_b,in_proj_qkv,in_proj_z,out_proj,gate_proj,up_proj,down_proj]' \
    '+actor_rollout_ref.model.target_parameters=[mlp.experts.gate_up_proj,mlp.experts.down_proj]' \
    actor_rollout_ref.model.lora.merge=False \
    actor_rollout_ref.model.mtp.enable=True \
    actor_rollout_ref.model.mtp.enable_train=False \
    actor_rollout_ref.model.mtp.enable_rollout=True \
    actor_rollout_ref.model.mtp.method=mtp \
    actor_rollout_ref.model.mtp.num_speculative_tokens=3 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    +actor_rollout_ref.actor.checkpoint.save_lora_only=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    'actor_rollout_ref.actor.optim.betas=[0.9,0.98]' \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.76 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.seed=17 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.max_model_len=8192 \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.max_num_seqs=64 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8192 \
    actor_rollout_ref.rollout.enforce_eager=False \
    "+actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config='{\"inductor_compile_config\":{\"triton.autotune_at_compile_time\":false}}'" \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.language_model_only=True \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.enable_flashinfer_autotune=False \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.gdn_prefill_backend=triton \
    actor_rollout_ref.rollout.checkpoint_engine.backend=naive \
    trainer.logger='["console"]' \
    trainer.project_name=qwen38_lora \
    trainer.experiment_name=bf16_fsdp16_tp4 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=2 \
    trainer.total_training_steps=200 \
    trainer.val_before_train=True \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.default_local_dir="$OUTPUT_DIR" \
    +ray_kwargs.ray_init.address="${RAY_ADDRESS:-auto}" \
    "$@"
