#!/usr/bin/env bash
#
# GRPO (DAPO-Math-17K, AIME-2024 eval) | MiniMax M3.1 NVFP4 | FSDP2 + LoRA | vLLM rollout | 4 nodes x 4 GPUs
#
# The trainer runs the verl-native MiniMax M3 text model (verl/models/transformers/minimax_m3): NVFP4 expert
# weights decoded to BF16, emulated activation quantization on the routed experts (NVFP4, FC2 per rollout TP
# shard) and on the other projections (MXFP8), the adapter delta added in FP32. Only LoRA adapters are trained and synced (r32/alpha64 on
# attention, dense MLP, shared and routed experts; routed experts share their outer factors). The rollout
# records the routed experts and the trainer replays them (R3).
#
# One optimizer step per batch, so the update forward is the old policy: bypass mode skips the old-log-prob
# pass and trains REINFORCE with token-level IS against the rollout log-probs (same gradient as decoupled
# PPO + token IS). Until verl-project/verl#8348 lands the actor workers ignore
# algorithm.rollout_correction, so the bypass loss is also set on the actor here.
#
# Needs a vLLM build with MiniMax M3 LoRA serving (routed-expert LoRA on the FlashInfer TRTLLM NVFP4 MoE is
# vllm-project/vllm#63024; MiniMax M3 LoRA support is vllm-project/vllm#63323), a Ray cluster with
# nnodes x 4 GPUs, the model on every node, and DAPO-Math-17K / AIME-2024 parquet files with data_source
# math_dapo / aime_2024.
#
# Reference run: 4 nodes x 4 GB300, 200 steps; AIME-2024 mean@8 36.3% at step 10, 67.5% at step 200. Steps 1-100
# ran before the bypass settings reached the actor workers (decoupled PPO + token IS with the old-log-prob pass; same
# gradient with one update per batch); steps 101-200 ran this configuration, resumed from step 100. That run rounded
# the routed-expert FC2 product before adding the adapter delta (the model code here adds it first), and its
# server quantized the FC1 input with a row rule instead of vLLM's per-token FlashInfer quantizer used here.

set -xeuo pipefail

# ---- user-adjustable ----
model_path=${model_path:-$HOME/models/MiniMax-M3.1-NVFP4}
train_file=${train_file:-$HOME/data/dapo-math-17k.parquet}
val_file=${val_file:-$HOME/data/aime-2024.parquet}
ckpt_dir=${ckpt_dir:-$HOME/ckpt/minimax_m31_dapo_lora}
nnodes=${nnodes:-4}
gpus_per_node=${gpus_per_node:-4}
rollout_tp=${rollout_tp:-4}
total_steps=${total_steps:-200}

export VLLM_ALLOW_RUNTIME_LORA_UPDATING=1

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=false \
    algorithm.rollout_correction.bypass_mode=true \
    algorithm.rollout_correction.loss_type=reinforce \
    algorithm.rollout_correction.rollout_is=token \
    algorithm.rollout_correction.rollout_is_threshold=2.0 \
    data.seed=42 \
    data.train_files="$train_file" \
    data.val_files="$val_file" \
    data.train_batch_size=32 \
    data.max_prompt_length=2048 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=false \
    data.truncation=error \
    data.shuffle=true \
    data.trust_remote_code=true \
    actor_rollout_ref.model.path="$model_path" \
    actor_rollout_ref.model.trust_remote_code=true \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    +actor_rollout_ref.model.override_config.text_config.expert_activation_qat=true \
    +actor_rollout_ref.model.override_config.text_config.linear_activation_qat=true     +actor_rollout_ref.model.override_config.text_config.expert_qat_tp_shards="$rollout_tp" \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=64 \
    "actor_rollout_ref.model.target_modules=[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,w1,w2,w3]" \
    actor_rollout_ref.model.lora.rank=32 \
    actor_rollout_ref.model.lora.alpha=64 \
    actor_rollout_ref.model.lora.merge=false \
    ++actor_rollout_ref.model.lora.experts_shared_outer_loras=true \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.seed=42 \
    actor_rollout_ref.actor.fsdp_config.use_torch_compile=false \
    actor_rollout_ref.actor.fsdp_config.router_replay.mode=R3 \
    actor_rollout_ref.actor.policy_loss.loss_mode=bypass_mode \
    ++actor_rollout_ref.actor.policy_loss.rollout_correction.bypass_mode=true \
    ++actor_rollout_ref.actor.policy_loss.rollout_correction.loss_type=reinforce \
    ++actor_rollout_ref.actor.policy_loss.rollout_correction.rollout_is=token \
    ++actor_rollout_ref.actor.policy_loss.rollout_correction.rollout_is_threshold=2.0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=true \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.calculate_entropy=true \
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=true \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    "actor_rollout_ref.actor.optim.betas=[0.9,0.98]" \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
    actor_rollout_ref.actor.optim.clip_grad=1.0 \
    actor_rollout_ref.actor.optim.lr_scheduler_type=constant \
    +actor_rollout_ref.actor.optim.override_optimizer_config.foreach=false \
    ++actor_rollout_ref.actor.checkpoint.save_lora_only=true \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.seed=42 \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$rollout_tp" \
    actor_rollout_ref.rollout.data_parallel_size=1 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n=8 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=true \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.max_num_seqs=64 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.max_model_len=6144 \
    "actor_rollout_ref.rollout.cudagraph_capture_sizes=[1,2,4,8,16,32,64]" \
    actor_rollout_ref.rollout.calculate_log_probs=true \
    actor_rollout_ref.rollout.enable_rollout_routing_replay=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=true \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=32768 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.block_size=128 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_dtype=nvfp4 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.attention_config.backend=FLASHINFER \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.attention_config.indexer_kv_dtype=nvfp4 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.attention_config.minimax_m3_msa_decode_backend=cutlass \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.kernel_config.enable_flashinfer_autotune=false \
    trainer.nnodes="$nnodes" \
    trainer.n_gpus_per_node="$gpus_per_node" \
    trainer.total_training_steps="$total_steps" \
    trainer.val_before_train=false \
    trainer.test_freq=10 \
    trainer.save_freq=10 \
    trainer.max_actor_ckpt_to_keep=20 \
    trainer.resume_mode=auto \
    "trainer.logger=[console,tensorboard]" \
    trainer.project_name=minimax_m31_dapo \
    trainer.experiment_name=grpo_lora_r32 \
    trainer.default_local_dir="$ckpt_dir" "$@"
