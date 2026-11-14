#!/usr/bin/env bash
# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to a local Qwen3-1.7B checkpoint}"
: "${DATA_DIR:?Set DATA_DIR to prepared ALFWorld parquet files (see the handoff)}"
# Run at the repository root. ALFWorld's CPU service must already be listening on localhost:8765.
# Two training GPUs plus one separate rollout GPU. TP=2, PP=2 or CP=2 replaces DP=2.
TP=${TP:-1}
PP=${PP:-1}
CP=${CP:-1}

python -m verl.trainer.main_ppo \
    model_engine=megatron \
    trainer.use_v1=True \
    trainer.v1.trainer_mode=group_delivery \
    actor_rollout_ref.hybrid_engine=False \
    algorithm.adv_estimator=grpo \
    data.train_files="${DATA_DIR}/train.parquet" \
    data.val_files="${DATA_DIR}/test.parquet" \
    data.train_batch_size=8 \
    data.val_batch_size=8 \
    data.max_prompt_length=2048 \
    data.max_response_length=6144 \
    '+data.apply_chat_template_kwargs.enable_thinking=False' \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size="${TP}" \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size="${PP}" \
    actor_rollout_ref.actor.megatron.context_parallel_size="${CP}" \
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.nnodes=1 \
    actor_rollout_ref.rollout.n_gpus_per_node=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.max_model_len=8192 \
    actor_rollout_ref.rollout.max_num_seqs=16 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8192 \
    actor_rollout_ref.rollout.checkpoint_engine.backend=nccl \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=50 \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=50 \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=2048 \
    actor_rollout_ref.rollout.agent.agent_loop_config_path=examples/group_delivery/agents.yaml \
    actor_rollout_ref.rollout.agent.num_workers=4 \
    actor_rollout_ref.rollout.val_kwargs.n=2 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=2 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=2 \
    trainer.val_before_train=False \
    trainer.test_freq=-1 \
    trainer.save_freq=1 \
    trainer.logger=console \
    trainer.project_name=verl_group_delivery \
    trainer.experiment_name=group_delivery_alfworld \
    "$@"
