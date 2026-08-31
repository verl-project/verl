#!/usr/bin/env bash
set -xeuo pipefail


SAVE_PATH=tests/utils/ci/profiler_data
rm -rf "$SAVE_PATH"

PROFILE_STEPS=[1]
PROFILE_RANKS_ALL=False
PROFILE_RANKS=[0]
DISCRETE=True

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FINISH_HOOK="$SCRIPT_DIR/profiler_finish_hook_marker.sh"

# The finish hook is dispatched by every profiled worker when profiling stops, so a run with
# profiling enabled must leave at least one marker behind. Markers live under finish_hook_markers/
# (not $SAVE_PATH directly) so their role-derived names are not mistaken for profiler stage
# deliverables by test_check_profiler_output.py (see profiler_finish_hook_marker.sh).
assert_finish_hook_ran() {
    if ! compgen -G "$SAVE_PATH/finish_hook_markers/finish_hook_ran_*" > /dev/null; then
        echo "global_profiler.finish_hook_cmd never ran: no marker file under $SAVE_PATH/finish_hook_markers"
        ls -la "$SAVE_PATH" || true
        exit 1
    fi
    ls "$SAVE_PATH"/finish_hook_markers/finish_hook_ran_*
}

# Download model if not exists
MODEL_ID=${MODEL_ID:-Qwen/Qwen2.5-0.5B-Instruct}
MODEL_PATH=${MODEL_PATH:-${HOME}/models/${MODEL_ID}}
#huggingface-cli download "${MODEL_ID}" --local-dir "${MODEL_PATH}"

TRAIN_FILES=${TRAIN_FILES:-${HOME}/data/gsm8k/train.parquet}
VAL_FILES=${VAL_FILES:-${HOME}/data/gsm8k/test.parquet}
MAX_PROMPT_LEN=${MAX_PROMPT_LEN:-512}
MAX_RESPONSE_LEN=${MAX_RESPONSE_LEN:-128}
# vLLM rejects enable_chunked_prefill=False when max_num_batched_tokens < max_model_len.
# Qwen2.5 / Qwen3-VL default max_model_len to max_position_embeddings (32k / 128k),
# so pin it to the actual prompt+response budget used by this E2E script.
MAX_MODEL_LEN=${MAX_MODEL_LEN:-$((MAX_PROMPT_LEN + MAX_RESPONSE_LEN))}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-True}
NUM_GPUS=${NUM_GPUS:-8}
FSDP_SIZE=${FSDP_SIZE:-4}
SP_SIZE=${SP_SIZE:-2}
EP_SIZE=${EP_SIZE:-1}
MODEL_NAME_ONLY=${MODEL_ID##*/}
VERL_EXP_NAME=${VERL_EXP_NAME:-${MODEL_NAME_ONLY}-function-reward-minimal-fsdp-size${FSDP_SIZE}}

device_name=$(python3 - <<'EOF'
from verl.utils.device import get_device_name
print(get_device_name())
EOF
)

common_params=(
    model_engine=veomni \
    algorithm.adv_estimator=grpo \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.train_batch_size=16 \
    data.max_prompt_length="${MAX_PROMPT_LEN}" \
    data.max_response_length="${MAX_RESPONSE_LEN}" \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.veomni.param_offload=True \
    actor_rollout_ref.actor.veomni.optimizer_offload=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.veomni.fsdp_size="${FSDP_SIZE}" \
    actor_rollout_ref.actor.veomni.ulysses_parallel_size="${SP_SIZE}" \
    actor_rollout_ref.actor.veomni.expert_parallel_size="${EP_SIZE}" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.veomni.param_offload=True \
    actor_rollout_ref.ref.use_torch_compile=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN}" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.ref.veomni.optimizer_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=console \
    trainer.project_name='verl_veomni_test' \
    trainer.experiment_name="${VERL_EXP_NAME}" \
    trainer.n_gpus_per_node="${NUM_GPUS}" \
    trainer.val_before_train="${VAL_BEFORE_TRAIN}" \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=1 \
    actor_rollout_ref.actor.profiler.enable=True \
    actor_rollout_ref.actor.profiler.all_ranks=$PROFILE_RANKS_ALL \
    actor_rollout_ref.actor.profiler.ranks=$PROFILE_RANKS \
    actor_rollout_ref.ref.profiler.enable=True \
    actor_rollout_ref.ref.profiler.all_ranks=$PROFILE_RANKS_ALL \
    actor_rollout_ref.ref.profiler.ranks=$PROFILE_RANKS \
    global_profiler.steps=$PROFILE_STEPS \
    global_profiler.save_path="$SAVE_PATH" \
    global_profiler.finish_hook_cmd="$FINISH_HOOK" \
)

if [ -n "$device_name" ] && [ "$device_name" == "cuda" ]; then
    CONTENTS=['cuda']
    python3 -m verl.trainer.main_ppo \
        "${common_params[@]}" \
        actor_rollout_ref.actor.profiler.tool_config.torch.discrete=$DISCRETE \
        actor_rollout_ref.actor.profiler.tool_config.torch.contents=$CONTENTS \
        actor_rollout_ref.ref.profiler.tool_config.torch.discrete=$DISCRETE \
        actor_rollout_ref.ref.profiler.tool_config.torch.contents=$CONTENTS \
        global_profiler.tool=torch $@

    assert_finish_hook_ran
    python3 "tests/utils/test_check_profiler_output.py" --profiler_dir="$SAVE_PATH" --device="gpu"
    
elif [ -n "$device_name" ] && [ "$device_name" == "npu" ]; then
    CONTENTS=['npu','cpu']
    python3 -m verl.trainer.main_ppo \
        "${common_params[@]}" \
        actor_rollout_ref.actor.profiler.tool_config.npu.discrete=$DISCRETE \
        actor_rollout_ref.actor.profiler.tool_config.npu.contents=$CONTENTS \
        actor_rollout_ref.ref.profiler.tool_config.npu.discrete=$DISCRETE \
        actor_rollout_ref.ref.profiler.tool_config.npu.contents=$CONTENTS \
        global_profiler.tool=npu $@

    assert_finish_hook_ran
    python3 "tests/utils/test_check_profiler_output.py" --profiler_dir="$SAVE_PATH" --device="npu"
else
    echo "Unknown device: $device_name"
    exit 1
fi

rm -rf "$SAVE_PATH"
