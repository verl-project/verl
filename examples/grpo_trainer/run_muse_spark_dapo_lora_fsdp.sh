#!/usr/bin/env bash
#
# GRPO (DAPO-Math-17K, AIME-2024 eval) | Muse Spark 816B-A42B NVFP4 | FSDP2 + LoRA | vLLM rollout | 4 nodes x 4 GPUs
#
# vLLM serves the NVFP4 checkpoint; the trainer runs verl/models/transformers/muse_spark on the same checkpoint
# (routed experts dequantized to BF16 at load) and trains LoRA only, with R3. Needs a vLLM build that serves Muse Spark
# with LoRA, and DAPO-Math-17K / AIME-2024 parquet files.

set -xeuo pipefail

# ---- user-adjustable ----
model_path=${model_path:-$HOME/models/Muse-Spark-Open-816B-A42B-NVFP4}
train_file=${train_file:-$HOME/data/dapo-math-17k.parquet}
val_file=${val_file:-$HOME/data/aime-2024.parquet}
ckpt_dir=${ckpt_dir:-$HOME/ckpt/muse_spark_dapo_lora}
nnodes=${nnodes:-4}
gpus_per_node=${gpus_per_node:-4}
total_steps=${total_steps:-150}

export VERL_USE_EXTERNAL_MODULES=verl.models.transformers.muse_spark.register
export VERL_FSDP2_SHARDED_INIT=1  # each rank reads only its FSDP2 shards
export VLLM_USE_V2_MODEL_RUNNER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# training config of the text model (the checkpoint config nests it under text_config)
text_config=${ckpt_dir}/text-config
mkdir -p "$ckpt_dir"
python3 - "$model_path" "$text_config" <<'PY'
import json
import sys

from verl.models.transformers.muse_spark.configuration_muse_spark import MuseSparkTextConfig

config = json.load(open(sys.argv[1] + "/config.json"))
MuseSparkTextConfig.from_checkpoint_config(config).save_pretrained(sys.argv[2])
PY

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_file" \
    data.val_files="$val_file" \
    data.train_batch_size=32 \
    data.max_prompt_length=1024 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=true \
    data.seed=42 \
    actor_rollout_ref.model.path="$model_path" \
    actor_rollout_ref.model.hf_config_path="$text_config" \
    actor_rollout_ref.model.use_remove_padding=false \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=64 \
    "actor_rollout_ref.model.target_modules=[q_proj,k_proj,v_proj,o_proj,gate_proj,pre_expert_proj,post_expert_proj]" \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    +actor_rollout_ref.model.override_config.expert_lora_rank=32 \
    +actor_rollout_ref.model.override_config.expert_lora_alpha=64 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.actor.fsdp_config.router_replay.mode=R3 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.use_dynamic_bsz=true \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    "actor_rollout_ref.actor.optim.betas=[0.9,0.98]" \
    ++actor_rollout_ref.actor.checkpoint.save_lora_only=true \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.n=8 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=true \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.60 \
    actor_rollout_ref.rollout.max_num_seqs=64 \
    actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
    actor_rollout_ref.rollout.max_model_len=5120 \
    actor_rollout_ref.rollout.enable_rollout_routing_replay=true \
    "+actor_rollout_ref.rollout.engine_kwargs.vllm.limit_mm_per_prompt={image:0,video:0,audio:0}" \
    trainer.nnodes="$nnodes" \
    trainer.n_gpus_per_node="$gpus_per_node" \
    trainer.total_training_steps="$total_steps" \
    trainer.test_freq=10 \
    trainer.save_freq=10 \
    "trainer.logger=[console,tensorboard]" \
    trainer.project_name=muse_spark_dapo \
    trainer.experiment_name=grpo_lora_r32 \
    trainer.default_local_dir="$ckpt_dir" "$@"
