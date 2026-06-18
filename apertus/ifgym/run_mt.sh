#!/usr/bin/env bash
# IFGym multi-turn instruction-following RL.
#
# Port of swiss-ai/if-gym train/mt/run_mt.sh onto this verl fork. The monkey
# patches from the original (sglang http_server, per-turn rm_scores in
# agent_loop.py, advantage-estimator auto-loading in core_algos.py) are no longer
# needed: per-turn credit assignment is handled in agent_loop._postprocess, and
# the estimators / agent loop are registered by apertus.ifgym.main_ifgym.
#
# Required env vars:
#   MODEL_PATH    path to the base model
#   EXP_NAME      wandb experiment name
#   DATA_DIR      dir with train.parquet / test.parquet (see prepare_ifgym_mt_data.py)
# Optional:
#   ALGO          perturn_grpo (default) | trajectory_grpo | trajectory_rloo |
#                 perturn_rloo | trajectory_gspo | perturn_gspo
#   CKPT_DIR, TP_SIZE, MICRO_BSZ, GPU_MEM_UTIL, N_GPUS_PER_NODE, NNODES
set -x

ulimit -n 65535 || true

# Repo root = two levels up from this script (apertus/ifgym/run_mt.sh).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH=${REPO_ROOT}:${PYTHONPATH}

: "${MODEL_PATH:?must set MODEL_PATH}"
: "${EXP_NAME:?must set EXP_NAME}"
: "${DATA_DIR:?must set DATA_DIR (with train.parquet/test.parquet)}"
: "${TP_SIZE:=1}"
: "${MICRO_BSZ:=4}"
: "${GPU_MEM_UTIL:=0.6}"
: "${N_GPUS_PER_NODE:=4}"
: "${NNODES:=1}"
: "${CKPT_DIR:=${SCRIPT_DIR}/checkpoints/${EXP_NAME}}"

# Algorithm selector -> hydra overrides. Mirrors the original run_mt.sh.
ALGO=${ALGO:-perturn_grpo}
case "$ALGO" in
  perturn_grpo)
    EXTRA_ARGS="algorithm.adv_estimator=ifgym_per_turn_grpo"
    ;;
  trajectory_grpo)
    EXTRA_ARGS="algorithm.adv_estimator=grpo"
    ;;
  trajectory_rloo)
    EXTRA_ARGS="algorithm.adv_estimator=rloo_vectorized"
    ;;
  perturn_rloo)
    EXTRA_ARGS="algorithm.adv_estimator=ifgym_per_turn_rloo"
    ;;
  trajectory_gspo)
    EXTRA_ARGS="algorithm.adv_estimator=grpo \
actor_rollout_ref.actor.policy_loss.loss_mode=gspo \
actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean"
    ;;
  perturn_gspo)
    EXTRA_ARGS="algorithm.adv_estimator=ifgym_per_turn_grpo \
actor_rollout_ref.actor.policy_loss.loss_mode=gspo \
actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean"
    ;;
  *)
    echo "[run_mt] unknown ALGO=$ALGO (perturn_grpo|trajectory_grpo|trajectory_rloo|perturn_rloo|trajectory_gspo|perturn_gspo)" >&2
    exit 1
    ;;
esac
echo "[run_mt] ALGO=$ALGO  EXTRA_ARGS=$EXTRA_ARGS"

python3 -m apertus.ifgym.main_ifgym \
    ${EXTRA_ARGS} \
    data.train_files=${DATA_DIR}/train.parquet \
    data.val_files=${DATA_DIR}/test.parquet \
    data.train_batch_size=64 \
    data.max_prompt_length=16384 \
    data.max_response_length=16384 \
    data.filter_overlong_prompts=True \
    data.truncation='right' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO_BSZ} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${MICRO_BSZ} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${TP_SIZE} \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM_UTIL} \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.multi_turn.enable=true \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=10 \
    actor_rollout_ref.rollout.agent.default_agent_loop=ifgym_agent \
    actor_rollout_ref.rollout.agent.agent_loop_config_path=${SCRIPT_DIR}/ifgym_agent.yaml \
    actor_rollout_ref.rollout.agent.num_workers=8 \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${MICRO_BSZ} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='ifgym_mt' \
    trainer.experiment_name=${EXP_NAME} \
    trainer.default_local_dir=${CKPT_DIR} \
    trainer.max_actor_ckpt_to_keep=3 \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=25 \
    trainer.test_freq=25 \
    trainer.resume_mode=auto \
    trainer.total_epochs=1 "$@"
