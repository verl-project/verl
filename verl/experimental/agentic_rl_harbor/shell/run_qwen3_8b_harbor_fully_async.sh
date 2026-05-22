#!/usr/bin/env bash
# Train Qwen3-8B with Harbor + VeRL fully_async_policy on Harbor task datasets.
#
# Defaults to TaskTrove v2 "easy" buckets (curriculum-easy / exercism-python-v2)
# because Qwen3-8B's pass@1 on competition-grade datasets like CodeContests is
# essentially zero, which collapses GRPO advantages to zero (no gradient signal)
# and still pays the full sandbox cost. Override HARBOR_TRAIN/HARBOR_VAL to point
# at a harder split when you have a stronger model.
#
# Prereqs (versions this recipe has been tested with):
#   pip install 'harbor[daytona]==0.6.1' 'litellm==1.82.6'
#   python verl/experimental/agentic_rl_harbor/prepare_harbor_dataset.py \
#       --dataset DCAgent/exp_rpt_curriculum-easy
#   python verl/experimental/agentic_rl_harbor/prepare_harbor_dataset.py \
#       --dataset laion/exp_rpt_exercism-python-v2
#
# Sandbox credentials (Harbor runs trials on a remote sandbox; pick one):
#   export DAYTONA_API_KEY=...    # default https://app.daytona.io/api
#   export DAYTONA_API_URL=...    # get an API key from https://app.daytona.io/dashboard/keys

set -xeuo pipefail
export VLLM_USE_V1=1

#----------------------- Paths -----------------------
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
HARBOR_PKG=${HARBOR_PKG:-$(cd "$SCRIPT_DIR/.." && pwd)}

MODEL_PATH=${MODEL_PATH:-/root/Qwen3-8B}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-$(basename "$MODEL_PATH")}

DATA_DIR=${DATA_DIR:-$HOME/data/harbor}
HARBOR_TRAIN=${HARBOR_TRAIN:-$DATA_DIR/exp_rpt_curriculum-easy}
HARBOR_VAL=${HARBOR_VAL:-$DATA_DIR/exp_rpt_exercism-python-v2}
TRAIN_FILES="['$HARBOR_TRAIN']"
VAL_FILES="['$HARBOR_VAL']"

HARBOR_DATASET_PY=${HARBOR_DATASET_PY:-$HARBOR_PKG/harbor_dataset.py}
HARBOR_AGENT_LOOP_YAML=${HARBOR_AGENT_LOOP_YAML:-$HARBOR_PKG/config/harbor_agent.yaml}
# Default Qwen3 chat template strips <think> blocks from non-last assistant turns,
# breaking the prefix invariant HarborAgentLoop's _merge_stepwise relies on. The
# accumulate-thinking variant keeps assistant content as-is across turns so a
# trajectory merges into a single training group instead of being split.
CHAT_TEMPLATE_PATH=${CHAT_TEMPLATE_PATH:-$HARBOR_PKG/templates/qwen3_acc_thinking.jinja2}

RUN_NAME=${RUN_NAME:-qwen3_8b_harbor_fully_async}
LOG_ROOT=${LOG_ROOT:-/root/logs/$RUN_NAME}
CKPT_DIR=${CKPT_DIR:-$LOG_ROOT/ckpts}

#----------------------- Algorithm -----------------------
ADV_ESTIMATOR=${ADV_ESTIMATOR:-grpo}
N_RESP_PER_PROMPT=${N_RESP_PER_PROMPT:-2}
N_RESP_PER_PROMPT_VAL=${N_RESP_PER_PROMPT_VAL:-1}

PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-2}
GEN_BATCH_SIZE=${GEN_BATCH_SIZE:-1}
TOTAL_ROLLOUT_STEPS=${TOTAL_ROLLOUT_STEPS:-32}
TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:-32}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
LOG_VAL_GENERATIONS=${LOG_VAL_GENERATIONS:-10}

MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-4096}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-$((32768 - MAX_PROMPT_LENGTH))}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}

ACTOR_LR=${ACTOR_LR:-1e-6}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.85}
OFFLOAD=${OFFLOAD:-True}

#----------------------- Async knobs (cf. fully_async_policy README) -----------------------
# Harbor trials run in a remote sandbox whose state is not serializable, so
# partial_rollout (Mode 4 in fully_async_policy/README.md) is not supported.
#
# Sandbox concurrency cap (IMPORTANT -- Daytona free tier defaults to 10 CPUs):
#   Each trial launches a Daytona sandbox that takes override_cpus=1 (see
#   config/harbor_agent.yaml). Exceeding the tier's total CPU quota is rejected
#   server-side with:
#     "Failed to create sandbox: Total CPU limit exceeded. Maximum allowed: 10."
#   The number of concurrent trials follows FullyAsyncRollouter (see
#   fully_async_rollouter.py:529-541):
#     max_required_samples   = ppo_mini_batch_size * require_batches
#                              * (staleness_threshold + 1) * trigger_parameter_sync_step
#     max_concurrent_samples = min(replicas * 16, max_required_samples)
#     concurrent_trials      ~= max_concurrent_samples * N_RESP_PER_PROMPT
#
#   Hard constraints we can't relax further:
#     N_RESP_PER_PROMPT=2          -- GRPO minimum (n=1 -> advantage is always 0)
#     PPO_MINI_BATCH_SIZE=2        -- must be divisible by train DP = N_GPUS_TRAIN/TRAIN_SP = 2
#     REQUIRE_BATCHES=1            -- streaming granularity, Mode 3 demo
#     TRIGGER_PARAM_SYNC_STEP=2    -- Mode 3 definition requires > 1
#   That leaves STALENESS_THRESHOLD as the only knob to keep concurrent_trials <= 10
#   while preserving Mode 3 (which requires staleness > 0).
#
#   Trade-off table at the current other defaults (ppo_mini=2, require=1, trigger=2, n=2):
#     staleness=1.0 -> max_req=8  -> 16 trials (tripped Daytona quota; was the old default)
#     staleness=0.5 -> max_req=6  -> 12 trials (still over the 10-CPU cap)
#     staleness=0.25-> max_req=5  -> 10 trials (exactly at cap, no headroom -> teardown race risk)
#     staleness=0.1 -> max_req=4  ->  8 trials (2 CPU headroom; Mode 3 preserved)  <-- current default
#     staleness=0.0 -> max_req=4  ->  8 trials (would degrade to Mode 2)
#
#   After upgrading the Daytona tier (https://app.daytona.io/dashboard/limits) or
#   switching to a local sandbox, raise STALENESS_THRESHOLD back to 1.0 to fully
#   exercise Mode 3 -- the "truly async" tier where trainer consumes samples from
#   the previous weight version while rollouter is already producing for the next.
STALENESS_THRESHOLD=${STALENESS_THRESHOLD:-0.1}
TRIGGER_PARAM_SYNC_STEP=${TRIGGER_PARAM_SYNC_STEP:-2}
REQUIRE_BATCHES=${REQUIRE_BATCHES:-1}
PARTIAL_ROLLOUT=${PARTIAL_ROLLOUT:-False}
TEST_FREQ=${TEST_FREQ:--1}

#----------------------- Resources -----------------------
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
N_GPUS_ROLLOUT=${N_GPUS_ROLLOUT:-4}
N_GPUS_TRAIN=${N_GPUS_TRAIN:-$((NGPUS_PER_NODE - N_GPUS_ROLLOUT))}
INFER_TP=${INFER_TP:-2}
TRAIN_SP=${TRAIN_SP:-2}
FSDP_SIZE=${FSDP_SIZE:-$N_GPUS_TRAIN}

ACTOR_MAX_TOKEN_LEN_PER_GPU=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
LOG_PROB_MAX_TOKEN_LEN_PER_GPU=$((ACTOR_MAX_TOKEN_LEN_PER_GPU * 4))

python3 -m verl.experimental.fully_async_policy.fully_async_main \
    algorithm.adv_estimator=$ADV_ESTIMATOR \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    data.train_files="$TRAIN_FILES" \
    data.val_files="$VAL_FILES" \
    data.return_raw_chat=True \
    data.train_batch_size=0 \
    data.gen_batch_size=$GEN_BATCH_SIZE \
    data.train_max_samples=$TRAIN_MAX_SAMPLES \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.custom_cls.path=$HARBOR_DATASET_PY \
    data.custom_cls.name=HarborTaskDataset \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.optim.lr=$ACTOR_LR \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$ACTOR_MAX_TOKEN_LEN_PER_GPU \
    actor_rollout_ref.actor.fsdp_config.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=$FSDP_SIZE \
    actor_rollout_ref.actor.fsdp_config.param_offload=$OFFLOAD \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$OFFLOAD \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$TRAIN_SP \
    actor_rollout_ref.actor.use_rollout_log_probs=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$LOG_PROB_MAX_TOKEN_LEN_PER_GPU \
    critic.strategy=fsdp2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$INFER_TP \
    actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEM_UTIL \
    actor_rollout_ref.rollout.n=$N_RESP_PER_PROMPT \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.disable_log_stats=False \
    actor_rollout_ref.rollout.prometheus.enable=True \
    actor_rollout_ref.rollout.prometheus.served_model_name=$SERVED_MODEL_NAME \
    actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n=$N_RESP_PER_PROMPT_VAL \
    actor_rollout_ref.rollout.agent.agent_loop_config_path=$HARBOR_AGENT_LOOP_YAML \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.chat_template=$CHAT_TEMPLATE_PATH \
    trainer.logger=['console'] \
    trainer.project_name=harbor \
    trainer.experiment_name=$RUN_NAME \
    trainer.val_before_train=False \
    trainer.test_freq=$TEST_FREQ \
    trainer.save_freq=-1 \
    trainer.default_local_dir=$CKPT_DIR \
    trainer.log_val_generations=$LOG_VAL_GENERATIONS \
    trainer.nnodes=$NNODES \
    trainer.n_gpus_per_node=$N_GPUS_TRAIN \
    rollout.nnodes=$NNODES \
    rollout.n_gpus_per_node=$N_GPUS_ROLLOUT \
    rollout.total_rollout_steps=$TOTAL_ROLLOUT_STEPS \
    trainer.total_epochs=$TOTAL_EPOCHS \
    async_training.staleness_threshold=$STALENESS_THRESHOLD \
    async_training.trigger_parameter_sync_step=$TRIGGER_PARAM_SYNC_STEP \
    async_training.require_batches=$REQUIRE_BATCHES \
    async_training.partial_rollout=$PARTIAL_ROLLOUT \
    "$@"
