#!/usr/bin/env bash
# GRPO with prefix-tree (MAGI) attention — public model + public dataset, CP=4.
#
# Prerequisites:
#   - GSM8K data preprocessed to parquet: python3 examples/data_preprocess/gsm8k.py
#   - magi_attention package installed (for magi backend)
#   - 8 GPUs (1 node, TP=1, PP=2, CP=4)
#
# Usage:
#   bash examples/grpo_trainer/run_grpo_prefix_tree_magi_megatron.sh
#   # or override model/data:
#   HF_MODEL_PATH=Qwen/Qwen2.5-7B-Instruct bash examples/grpo_trainer/run_grpo_prefix_tree_magi_megatron.sh

set -xeuo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=1
export VLLM_USE_V1=1
export HYDRA_FULL_ERROR=1

########################### Config ###########################

HF_MODEL_PATH=${HF_MODEL_PATH:-Qwen/Qwen3-4B-Instruct-2507}
GSM8K_TRAIN=${GSM8K_TRAIN:-$HOME/data/gsm8k/train.parquet}
GSM8K_TEST=${GSM8K_TEST:-$HOME/data/gsm8k/test.parquet}
REWARD_FN=${REWARD_FN:-verl/utils/reward_score/gsm8k.py}

TS=$(date +%Y%m%d_%H%M%S)
OUTDIR=${OUTDIR:-/tmp/verl_grpo_prefix_tree/${TS}}
mkdir -p "$OUTDIR"

########################### Launch ###########################

python3 -m verl.trainer.main_ppo \
    model_engine=megatron \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    \
    data.train_files="['${GSM8K_TRAIN}']" \
    data.val_files="['${GSM8K_TEST}']" \
    data.val_max_samples=32 \
    data.train_batch_size=64 \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.prompt_key=prompt \
    \
    actor_rollout_ref.model.path="${HF_MODEL_PATH}" \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_prefix_tree=True \
    actor_rollout_ref.model.prefix_tree_attention=magi \
    \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_decay_style=constant \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096 \
    \
    # NOTE: prefix-tree does NOT support sequence parallelism (SP) with TP>1.
    # This run uses TP=1 (SP is auto-disabled). If you raise TP above 1, also
    # disable SP explicitly:
    #   actor_rollout_ref.actor.megatron.sequence_parallel=False
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=1 \
    actor_rollout_ref.actor.megatron.context_parallel_size=4 \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=2 \
    actor_rollout_ref.actor.megatron.use_mbridge=True \
    actor_rollout_ref.actor.megatron.vanilla_mbridge=True \
    actor_rollout_ref.actor.megatron.use_megatron_fsdp=True \
    '+actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=False' \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.max_model_len=4096 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8192 \
    \
    reward.custom_reward_function.path="${REWARD_FN}" \
    reward.num_workers=2 \
    \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.total_training_steps=50 \
    trainer.save_freq=-1 \
    trainer.test_freq=10 \
    trainer.val_before_train=False \
    trainer.balance_batch=True \
    'trainer.logger=["console","tensorboard"]' \
    trainer.project_name=grpo_prefix_tree_magi \
    trainer.experiment_name="grpo_magi_${TS}" \
    2>&1 | tee "${OUTDIR}/run.log"

echo "================================================================"
echo "Done -> ${OUTDIR}"
echo "================================================================"
