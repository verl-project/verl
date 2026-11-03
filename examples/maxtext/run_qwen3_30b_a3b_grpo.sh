#!/usr/bin/env bash
# GRPO on GSM8K with a MaxText (JAX) trainer and a vLLM rollout.
#
# Topology: the trainer is one Ray actor that owns every GPU on its node. The
# rollout is vLLM on a separate node, one process per GPU. Weights go from the
# trainer to the rollout through veRL's NCCL checkpoint engine. There is no
# reference model or critic: a second MaxText process could not share the
# trainer's GPUs.
#
# Prerequisites
#   * An image with MaxText, vLLM and veRL: docker/rocm/Dockerfile.rocm.maxtext.
#   * A running Ray cluster of two 8-GPU nodes (trainer + rollout); run this
#     script on the head node with RAY_ADDRESS set, or pass
#     +ray_kwargs.ray_init.address=<head>:<port>. veRL must be importable by
#     the Ray workers on both nodes (pip install -e . in the image, or
#     PYTHONPATH=<verl repo> when starting Ray).
#   * The default MAXTEXT_CONFIG is an absolute path under this checkout. The trainer
#     node needs that file at the same path, or MAXTEXT_CONFIG must point to a file
#     that exists there.
#   * MODEL_PATH: a local Hugging Face checkpoint of Qwen3-30B-A3B.
#   * GSM8K parquet files from examples/data_preprocess/gsm8k.py.
#
# The MaxText model and parallelism come from MAXTEXT_CONFIG, a normal MaxText
# config. The compiled step consumes num_devices * per_device_batch_size *
# gradient_accumulation_steps sequences (8 * 1 * 4 = 32 for the default
# config), which must equal TRAIN_BATCH_SIZE * ROLLOUT_N.
#
# Optional
#   Checkpoints (fp32 weights only, no optimizer state): SAVE_FREQ=10
#     CKPT_DIR=/path RESUME_MODE=auto. A restarted job resumes from the latest.
#     CKPT_DIR must be an absolute path on storage shared by both nodes: the trainer
#     task runner writes the tracker and data state, the MaxText actor writes weights.
#   Weights & Biases: TRAINER_LOGGER='[console,wandb]' and WANDB_API_KEY.
#   Extra Hydra overrides go after the script arguments. The MaxText optimizer is
#   configured by MaxText keys, not actor.optim.*; for example, to change the
#   learning rate (1e-6 in the shipped config):
#     bash run_qwen3_30b_a3b_grpo.sh +actor_rollout_ref.actor.maxtext.overrides.learning_rate=2e-6
#   Extra MaxText keys: +actor_rollout_ref.actor.maxtext.overrides.<key>=<value>
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to a local Qwen3-30B-A3B checkpoint}"
DATA_DIR="${DATA_DIR:-${HOME}/data/gsm8k}"
MAXTEXT_CONFIG="${MAXTEXT_CONFIG:-${ROOT}/examples/maxtext/qwen3_30b_a3b_gsm8k.yml}"
PROJECT_NAME="${PROJECT_NAME:-verl_maxtext}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3_30b_a3b_gsm8k}"

if [ "${SAVE_FREQ:--1}" != "-1" ] || [ "${RESUME_MODE:-disable}" != "disable" ]; then
  case "${CKPT_DIR:-}" in
    /*) ;;
    *) echo "Checkpointing needs CKPT_DIR set to an absolute path on storage shared by both nodes." >&2; exit 1 ;;
  esac
fi

RAY_ARGS=()
if [ -n "${RAY_ADDRESS:-}" ]; then
  RAY_ARGS=(+ray_kwargs.ray_init.address="${RAY_ADDRESS}")
fi

python3 -m verl.trainer.main_ppo \
  model_engine=maxtext \
  trainer.v1.trainer_mode=separate_async \
  trainer.v1.separate_async.parameter_sync_step=1 \
  trainer.v1.separate_async.num_warmup_batches=1 \
  actor_rollout_ref.hybrid_engine=false \
  actor_rollout_ref.actor.maxtext.config_path="${MAXTEXT_CONFIG}" \
  +actor_rollout_ref.actor.maxtext.overrides.tokenizer_path="${MODEL_PATH}" \
  actor_rollout_ref.actor.use_kl_loss=false \
  actor_rollout_ref.actor.use_dynamic_bsz=false \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE:-8}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE:-4}" \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.n="${ROLLOUT_N:-4}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP:-8}" \
  actor_rollout_ref.rollout.nnodes=1 \
  actor_rollout_ref.rollout.n_gpus_per_node=8 \
  actor_rollout_ref.rollout.checkpoint_engine.backend=nccl \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.8}" \
  data.train_files="${TRAIN_FILES:-${DATA_DIR}/train.parquet}" \
  data.val_files="${VAL_FILES:-${DATA_DIR}/test.parquet}" \
  data.train_batch_size="${TRAIN_BATCH_SIZE:-8}" \
  data.max_prompt_length="${MAX_PROMPT_LENGTH:-256}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH:-384}" \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=false \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node=8 \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.total_epochs=100 \
  trainer.total_training_steps="${TOTAL_TRAINING_STEPS:-100}" \
  trainer.test_freq=0 \
  trainer.val_before_train=false \
  trainer.save_freq="${SAVE_FREQ:--1}" \
  trainer.default_local_dir="${CKPT_DIR:-checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}}" \
  trainer.resume_mode="${RESUME_MODE:-disable}" \
  trainer.max_actor_ckpt_to_keep="${MAX_CKPT_KEEP:-2}" \
  trainer.logger="${TRAINER_LOGGER:-[console]}" \
  "${RAY_ARGS[@]}" \
  "$@"
