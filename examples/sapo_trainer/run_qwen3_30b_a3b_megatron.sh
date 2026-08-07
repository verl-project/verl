#!/usr/bin/env bash
# SAPO | Qwen3-30B-A3B (MoE) | Megatron training | vLLM rollout | GPU or Ascend NPU
# SAPO replaces ratio clipping with a smooth tau-parameterized surrogate (arXiv:2511.20347).
#
# Platform and inference backend are runtime toggles, not separate scripts:
#   DEVICE=npu INFER_BACKEND=vllm NDEVICES_PER_NODE=16 bash examples/sapo_trainer/run_qwen3_30b_a3b_megatron.sh

set -xeuo pipefail

########################### user-adjustable ###########################
# DEVICE is auto-detected by probing torch_npu; override only for special cases.
DEVICE=${DEVICE:-$(python3 -c 'import torch_npu' 2>/dev/null && echo npu || echo gpu)}
INFER_BACKEND=${INFER_BACKEND:-vllm}

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-30B-A3B-Base}
# Optional pre-converted Megatron dist checkpoint. Produce it with:
#   python3 scripts/converter_hf_to_mcore.py --hf_model_path "$MODEL_PATH" \
#       --output_path "$MCORE_MODEL_PATH" --use_cpu_initialization
# Leave empty to let mbridge load the HF weights directly.
MCORE_MODEL_PATH=${MCORE_MODEL_PATH:-}

NNODES=${NNODES:-1}
NDEVICES_PER_NODE=${NDEVICES_PER_NODE:-}

# SAPO smoothing temperatures (paper defaults for Qwen3-30B-A3B-Base).
TAU_POS=${TAU_POS:-1.0}
TAU_NEG=${TAU_NEG:-1.05}

# Megatron parallelism. EP is not bounded by DP; the constraint is
# EP * ETP == world_size (with PP=CP=1). Larger EP spreads the 128 experts over
# more ranks, which is the main lever on expert memory.
TP=${TP:-4}
PP=${PP:-1}
CP=${CP:-1}
EP=${EP:-4}
ETP=${ETP:-4}

# Activation recomputation: "full" recomputes every layer (max memory saving,
# slowest backward), "selective" recomputes only the cheap ops, "none" disables
# it. Megatron *rejects* recompute_method/recompute_num_layers when granularity
# is selective, so each mode emits its own flag set rather than sharing one.
RECOMPUTE=${RECOMPUTE:-full}

# Fraction of optimizer state held on the host. 1 offloads all of it, which a
# 30B MoE needs on 8 devices; lower it to trade device memory back for speed
# once you know how much headroom you have.
OPTIMIZER_OFFLOAD_FRACTION=${OPTIMIZER_OFFLOAD_FRACTION:-1}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-32}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-8192}

ACTOR_LR=${ACTOR_LR:-1e-6}
ENTROPY_COEFF=${ENTROPY_COEFF:-0}

ROLLOUT_N=${ROLLOUT_N:-8}

TRAIN_FILE=${TRAIN_FILE:-$HOME/data/dapo-math-17k/train.parquet}
VAL_FILE=${VAL_FILE:-$HOME/data/aime-2024/test.parquet}

PROJECT_NAME=${PROJECT_NAME:-verl_sapo_qwen3_moe}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_30b_a3b_megatron}
SAVE_FREQ=${SAVE_FREQ:-50}

# What goes into each checkpoint. The default ['model','optimizer','extra']
# writes ~374 GB for this model on 16 ranks -- roughly 57 GB of weights plus
# ~317 GB of Adam state. Dropping 'optimizer' keeps checkpoints at weight size;
# the cost is that a resumed run restarts the optimizer from scratch.
#
# Weight-only is necessary but not sufficient on a networked filesystem. A
# 16-rank run measured ~25 MiB/s aggregate to JuiceFS and did not speed up with
# more concurrent writers, so even 57 GB takes ~37 minutes. One rank then ran
# ~4x slower than its peers: the other 15 finished, entered the collective, and
# died on the 30-minute gloo barrier timeout with 59 of 61 GB on disk and no
# .metadata -- an unloadable checkpoint and a lost run. If your shared
# filesystem is anywhere near that slow, point trainer.default_local_dir at
# node-local disk and copy the finished checkpoint out afterwards.
SAVE_CONTENTS=${SAVE_CONTENTS:-'["model","extra"]'}

# Node-local disk is far smaller than a shared filesystem, so bound retention.
MAX_ACTOR_CKPT_TO_KEEP=${MAX_ACTOR_CKPT_TO_KEEP:-2}

# Profiling is opt-in and costs nothing when off. Discrete mode splits the
# trace per role (rollout / actor_compute_log_prob / actor_update /
# ref_compute_log_prob), which is what turns "the step is slow" into "this
# stage is slow". Step 1 pays one-off compilation and cache warmup, so the
# default window starts at step 2.
PROFILE=${PROFILE:-0}
PROFILE_STEPS=${PROFILE_STEPS:-"[2,3]"}
PROFILE_RANKS=${PROFILE_RANKS:-"[0]"}
PROFILE_ALL_RANKS=${PROFILE_ALL_RANKS:-False}
PROFILE_DISCRETE=${PROFILE_DISCRETE:-True}
# NPU-only knobs; ignored when DEVICE=gpu selects the torch profiler.
PROFILE_LEVEL=${PROFILE_LEVEL:-level1}
PROFILE_ANALYSIS=${PROFILE_ANALYSIS:-True}
PROFILE_SAVE_PATH=${PROFILE_SAVE_PATH:-./profile_data}

# Prompt filtering runs to completion before any device work starts, and the
# shipped data config pins it to a single process (see
# trainer/config/data/legacy_data.yaml), so filtering 1.79M samples costs ~28
# minutes with every accelerator idle. The code default is cpu_count()//4;
# restore that here so the wait scales with the machine.
FILTER_WORKERS=${FILTER_WORKERS:-$(python3 -c 'import os; print(max(1, os.cpu_count() // 4))' 2>/dev/null || echo 8)}
TEST_FREQ=${TEST_FREQ:--1}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-10}
########################### end user-adjustable ###########################

########################### per-device defaults ###########################
case "${DEVICE}" in
    gpu)
        export CUDA_DEVICE_MAX_CONNECTIONS=1  # for megatron comm/compute overlap
        n_devices_per_node=${NDEVICES_PER_NODE:-8}
        gen_tp=${GEN_TP:-4}
        rollout_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.8}
        ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-20480}
        profile_tool=torch
        profile_contents=${PROFILE_CONTENTS:-"['cuda','cpu']"}
        ;;
    npu)
        export CUDA_DEVICE_MAX_CONNECTIONS=1
        export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-1500}
        export HCCL_OP_EXPANSION_MODE=${HCCL_OP_EXPANSION_MODE:-AIV}  # more streams than FFTS+
        export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
        export TASK_QUEUE_ENABLE=${TASK_QUEUE_ENABLE:-1}
        n_devices_per_node=${NDEVICES_PER_NODE:-16}
        gen_tp=${GEN_TP:-4}
        # Rollout and training share device memory; leave headroom for the
        # offload traffic Megatron generates on Ascend.
        rollout_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.5}
        ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-10240}
        profile_tool=npu
        profile_contents=${PROFILE_CONTENTS:-"['npu','cpu']"}
        ;;
    *)
        echo "Unsupported DEVICE=${DEVICE}. Expected 'gpu' or 'npu'." >&2
        exit 1
        ;;
esac

########################### parameter arrays ###########################

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="${TRAIN_FILE}"
    data.val_files="${VAL_FILE}"
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.filter_overlong_prompts=True
    data.filter_overlong_prompts_workers=${FILTER_WORKERS}
    data.truncation='error'
)

MODEL=(
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.model.use_remove_padding=True
)

# SAPO: tau_pos/tau_neg are ActorConfig fields, NOT policy_loss fields.
# compute_policy_loss_sapo reads config.tau_pos off ActorConfig, so overriding
# them under policy_loss silently has no effect.
ACTOR=(
    actor_rollout_ref.actor.policy_loss.loss_mode=sapo
    actor_rollout_ref.actor.tau_pos=${TAU_POS}
    actor_rollout_ref.actor.tau_neg=${TAU_NEG}
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    # SAPO drops ratio clipping, and the paper trains without a KL penalty.
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF}
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${TP}
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${PP}
    actor_rollout_ref.actor.megatron.context_parallel_size=${CP}
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${EP}
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${ETP}
    actor_rollout_ref.actor.megatron.use_mbridge=True
    actor_rollout_ref.actor.megatron.param_offload=True
    actor_rollout_ref.actor.megatron.optimizer_offload=True
    actor_rollout_ref.actor.megatron.grad_offload=True
    # megatron.optimizer_offload alone does not move the distributed optimizer
    # state off-device: without these, Adam lazily allocates exp_avg/exp_avg_sq
    # on the accelerator during the first step() and a 30B MoE runs out of
    # memory there.
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=${OPTIMIZER_OFFLOAD_FRACTION}
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=True
    # 128 experts without fp32 routing is numerically fragile (Megatron warns).
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype=fp32
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=${INFER_BACKEND}
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp}
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_mem_util}
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
)

REF=(
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${TP}
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${PP}
    actor_rollout_ref.ref.megatron.context_parallel_size=${CP}
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=${EP}
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=${ETP}
    actor_rollout_ref.ref.megatron.use_mbridge=True
    actor_rollout_ref.ref.megatron.param_offload=True
)

TRAINER=(
    trainer.critic_warmup=0
    trainer.logger='["console"]'
    trainer.project_name="${PROJECT_NAME}"
    trainer.experiment_name="${EXPERIMENT_NAME}"
    trainer.nnodes=${NNODES}
    trainer.n_gpus_per_node=${n_devices_per_node}
    trainer.device=${DEVICE}
    trainer.val_before_train=False
    trainer.save_freq=${SAVE_FREQ}
    trainer.max_actor_ckpt_to_keep=${MAX_ACTOR_CKPT_TO_KEEP}
    actor_rollout_ref.actor.checkpoint.save_contents=${SAVE_CONTENTS}
    trainer.test_freq=${TEST_FREQ}
    trainer.total_epochs=${TOTAL_EPOCHS}
)

# Trailing extras array; stays non-empty-safe under `set -u`.
EXTRA=(
    model_engine=megatron
)

# Activation recomputation. Megatron validates these against each other:
# selective granularity requires recompute_num_layers/method to be unset, so
# the modes cannot share one flag set.
case "${RECOMPUTE}" in
    full)
        EXTRA+=(
            +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
            +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
            +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
        )
        ;;
    selective)
        EXTRA+=(
            +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=selective
        )
        ;;
    none)
        ;;
    *)
        echo "Unsupported RECOMPUTE=${RECOMPUTE}. Expected 'full', 'selective' or 'none'." >&2
        exit 1
        ;;
esac

# Profiling. All three roles are traced so the per-stage split is complete;
# tracing only the actor tells you update_actor is slow but not what it is
# competing with. The tool_config keys are tool-specific, so each profiler
# gets its own literal block rather than an interpolated key path -- these
# must stay greppable and checkable against the config schema.
if [ "${PROFILE}" != 0 ]; then
    EXTRA+=(
        global_profiler.tool=${profile_tool}
        global_profiler.steps=${PROFILE_STEPS}
        global_profiler.save_path="${PROFILE_SAVE_PATH}"
        actor_rollout_ref.actor.profiler.enable=True
        actor_rollout_ref.actor.profiler.ranks=${PROFILE_RANKS}
        actor_rollout_ref.actor.profiler.all_ranks=${PROFILE_ALL_RANKS}
        actor_rollout_ref.rollout.profiler.enable=True
        actor_rollout_ref.rollout.profiler.ranks=${PROFILE_RANKS}
        actor_rollout_ref.rollout.profiler.all_ranks=${PROFILE_ALL_RANKS}
        actor_rollout_ref.ref.profiler.enable=True
        actor_rollout_ref.ref.profiler.ranks=${PROFILE_RANKS}
        actor_rollout_ref.ref.profiler.all_ranks=${PROFILE_ALL_RANKS}
    )
    if [ "${profile_tool}" = npu ]; then
        EXTRA+=(
            actor_rollout_ref.actor.profiler.tool_config.npu.discrete=${PROFILE_DISCRETE}
            actor_rollout_ref.actor.profiler.tool_config.npu.contents=${profile_contents}
            actor_rollout_ref.actor.profiler.tool_config.npu.level=${PROFILE_LEVEL}
            actor_rollout_ref.actor.profiler.tool_config.npu.analysis=${PROFILE_ANALYSIS}
            actor_rollout_ref.rollout.profiler.tool_config.npu.discrete=${PROFILE_DISCRETE}
            actor_rollout_ref.rollout.profiler.tool_config.npu.contents=${profile_contents}
            actor_rollout_ref.rollout.profiler.tool_config.npu.level=${PROFILE_LEVEL}
            actor_rollout_ref.rollout.profiler.tool_config.npu.analysis=${PROFILE_ANALYSIS}
            actor_rollout_ref.ref.profiler.tool_config.npu.discrete=${PROFILE_DISCRETE}
            actor_rollout_ref.ref.profiler.tool_config.npu.contents=${profile_contents}
            actor_rollout_ref.ref.profiler.tool_config.npu.level=${PROFILE_LEVEL}
            actor_rollout_ref.ref.profiler.tool_config.npu.analysis=${PROFILE_ANALYSIS}
        )
    else
        EXTRA+=(
            actor_rollout_ref.actor.profiler.tool_config.torch.discrete=${PROFILE_DISCRETE}
            actor_rollout_ref.actor.profiler.tool_config.torch.contents=${profile_contents}
            actor_rollout_ref.rollout.profiler.tool_config.torch.discrete=${PROFILE_DISCRETE}
            actor_rollout_ref.rollout.profiler.tool_config.torch.contents=${profile_contents}
            actor_rollout_ref.ref.profiler.tool_config.torch.discrete=${PROFILE_DISCRETE}
            actor_rollout_ref.ref.profiler.tool_config.torch.contents=${profile_contents}
        )
    fi
fi

# Load from a pre-converted Megatron dist checkpoint when one is supplied.
if [ -n "${MCORE_MODEL_PATH}" ]; then
    EXTRA+=(
        actor_rollout_ref.actor.megatron.use_dist_checkpointing=True
        actor_rollout_ref.actor.megatron.dist_checkpointing_path="${MCORE_MODEL_PATH}"
        actor_rollout_ref.ref.megatron.use_dist_checkpointing=True
        actor_rollout_ref.ref.megatron.dist_checkpointing_path="${MCORE_MODEL_PATH}"
    )
fi

if [ "${DEVICE}" = npu ]; then
    EXTRA+=(
        actor_rollout_ref.actor.use_torch_compile=False
        actor_rollout_ref.ref.use_torch_compile=False
    )
fi

########################### launch ###########################
python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${TRAINER[@]}" \
    "${EXTRA[@]}" \
    "$@"
