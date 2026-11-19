#!/bin/bash
# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
set -euo pipefail
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
: "${MODEL_PATH:?Set MODEL_PATH to the cached Qwen3-0.6B checkpoint}"
: "${SMOKE_DIR:?Set SMOKE_DIR to a writable directory for inputs and logs}"
export MODEL_PATH SMOKE_DIR
python3 tests/special_distributed/delta_grpo_smoke/prepare.py
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
export HF_HUB_OFFLINE=1
python3 -m verl.experimental.one_step_off_policy.main_ppo \
 "actor_rollout_ref.model.path=$MODEL_PATH" \
 "data.train_files=$SMOKE_DIR/train.parquet" "data.val_files=$SMOKE_DIR/val.parquet" \
 data.train_batch_size=4 data.max_prompt_length=128 data.max_response_length=32 \
 data.filter_overlong_prompts=True data.truncation=error \
 actor_rollout_ref.actor.strategy=fsdp2 actor_rollout_ref.hybrid_engine=False \
 actor_rollout_ref.actor.ppo_mini_batch_size=4 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
 actor_rollout_ref.actor.optim.lr=1e-5 actor_rollout_ref.actor.use_kl_loss=False \
 actor_rollout_ref.actor.fsdp_config.use_torch_compile=False \
 actor_rollout_ref.model.use_remove_padding=False actor_rollout_ref.model.enable_gradient_checkpointing=True \
 +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
 actor_rollout_ref.rollout.name=sglang actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
 actor_rollout_ref.rollout.n=2 actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
 actor_rollout_ref.rollout.max_model_len=256 actor_rollout_ref.rollout.agent.num_workers=2 \
 +actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=triton \
 +actor_rollout_ref.rollout.engine_kwargs.sglang.disable_cuda_graph=True \
 actor_rollout_ref.rollout.checkpoint_engine.backend=delta_sharded \
 +actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.delta_sharded.encoding=indices \
 +actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.delta_sharded.verify_every=1 \
 +actor_rollout_ref.rollout.engine_kwargs.sglang.log_level=warning \
 algorithm.adv_estimator=grpo algorithm.use_kl_in_reward=False \
 reward.num_workers=2 "reward.custom_reward_function.path=$REPO_ROOT/tests/special_distributed/delta_grpo_smoke/reward.py" \
 reward.custom_reward_function.name=compute_score \
 trainer.nnodes=1 trainer.n_gpus_per_node=1 rollout.nnodes=1 rollout.n_gpus_per_node=1 \
 trainer.total_epochs=2 trainer.total_training_steps=10 trainer.val_before_train=False \
 trainer.save_freq=-1 trainer.test_freq=-1 trainer.logger='[console]' \
 trainer.project_name=delta-l20-validation trainer.experiment_name=fsdp2-sglang \
 ray_kwargs.ray_init.num_cpus=32 "$@"
