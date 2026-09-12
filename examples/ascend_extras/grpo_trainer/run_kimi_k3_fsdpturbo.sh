#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "$SCRIPT_DIR/kimi_k3_runtime.sh"
PROJECT_ROOT=$BACKEND_ROOT
run_mode="${KIMI_RUN_MODE:-single}"
start_time="${VERL_RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
project_name="${VERL_PROJECT_NAME:-GRPO-Kimi-K3-A3B}"

model_path=${MODEL_PATH:-/mnt/share/w00848461/weights/K3-top16-pruned_bf16_countbench_128_n4}

case "$run_mode" in
    single)
        nnodes=1
        npus_per_node=${NPUS_PER_NODE:-8}
        ;;
    multinode)
        nnodes=${NNODES:?set by ray_start_multi_nodes.sh}
        ((nnodes >= 2)) || { echo "multinode mode requires at least two nodes" >&2; exit 2; }
        npus_per_node=${NPUS_PER_NODE:-16}
        ;;
    *) echo "Unsupported KIMI_RUN_MODE: $run_mode" >&2; exit 2 ;;
esac

turbo_fsdp_size=$((nnodes * npus_per_node))
turbo_ep_size=${TURBO_EP_SIZE:-$npus_per_node}
rollout_tp_size=${ROLLOUT_TP:-$npus_per_node}
if ((turbo_ep_size <= 0 || rollout_tp_size <= 0 || turbo_fsdp_size % turbo_ep_size != 0 || turbo_fsdp_size % rollout_tp_size != 0 || turbo_ep_size != rollout_tp_size)); then
    echo "FSDP requires actor EP = rollout TP > 0, dividing the actor world size" >&2
    exit 2
fi
turbo_efsdp_size=$((turbo_fsdp_size / turbo_ep_size))
rollout_ep_size=$rollout_tp_size
visible_devices_default=$(seq -s, 0 $((npus_per_node - 1)))
max_response_length=${VERL_MAX_RESPONSE_LENGTH:-512}
rollout_max_num_seqs=${VERL_VLLM_MAX_NUM_SEQS:-32}
rollout_max_num_batched_tokens=${VERL_VLLM_MAX_NUM_BATCHED_TOKENS:-4096}
train_batch_size=${TRAIN_BATCH_SIZE:-16}
rollout_n=${ROLLOUT_N:-4}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-16}
exp_name=${VERL_EXPERIMENT_NAME:-kimi_k3_fsdpturbo_${nnodes}nodes_${start_time}}
total_training_steps=${VERL_TOTAL_TRAINING_STEPS:-5}
save_freq_default=-1
profiler_log_level_default=WARN

save_freq="${VERL_SAVE_FREQ:-${save_freq_default}}"
rollout_enforce_eager="${VERL_VLLM_ENFORCE_EAGER:-False}"
max_actor_ckpt_to_keep="${VERL_MAX_ACTOR_CKPT_TO_KEEP:-2}"
resume_mode="${VERL_RESUME_MODE:-disable}"
resume_from_path="${VERL_RESUME_FROM_PATH:-}"
train_file="${TRAIN_FILE:-/mnt/share/w00848461/datasets/countbenchqa_lite/train_448.parquet}"
test_file="${TEST_FILE:-/mnt/share/w00848461/datasets/countbenchqa_lite/train.parquet}"

verl_path="${PROJECT_ROOT}/verl"
fsdp_turbo_path="${PROJECT_ROOT}/FSDPTurbo"
vllm_path="${PROJECT_ROOT}/vllm"
vllm_ascend_path="${PROJECT_ROOT}/vllm-ascend"
runtime_site="${PROJECT_ROOT}/.runtime-site"
cann_python_site=/usr/local/Ascend/cann-9.0.1/python/site-packages

# The checkpoint's Python files are executable remote code. Keep actor and
# rollout on the reviewed FSDP-Turbo implementation for every launch.
model_source=$model_path
model_path="$PROJECT_ROOT/models/fsdpturbo-$(basename "$model_source")"
python3 "$SCRIPT_DIR/prepare_kimi_k3_model.py" "$model_source" "$model_path" --code "$fsdp_turbo_path/fsdp_turbo/models/kimi"

export PYTHONPATH="${PROJECT_ROOT}/.python_deps/modelopt-0.46.0:${runtime_site}:${verl_path}:${vllm_path}:${vllm_ascend_path}:${fsdp_turbo_path}:${fsdp_turbo_path}/examples:${cann_python_site}"
export FSDP_TURBO_ROOT="${fsdp_turbo_path}"
export VLLM_ASCEND_PATH="${vllm_ascend_path}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-${visible_devices_default}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NON_MEGATRON="${NON_MEGATRON:-true}"
export RAY_ENABLE_UV_RUN_RUNTIME_ENV="${RAY_ENABLE_UV_RUN_RUNTIME_ENV:-0}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-${profiler_log_level_default}}"
export VERL_LOGGING_LEVEL="${VERL_LOGGING_LEVEL:-${profiler_log_level_default}}"
export VLLM_DISABLE_COMPILE_CACHE="${VLLM_DISABLE_COMPILE_CACHE:-1}"
export VLLM_BATCH_INVARIANT="${VLLM_BATCH_INVARIANT:-0}"
export VLLM_ALLOW_RUNTIME_LORA_UPDATING="${VLLM_ALLOW_RUNTIME_LORA_UPDATING:-true}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export VLLM_ASCEND_ENABLE_FLASHCOMM="${VLLM_ASCEND_ENABLE_FLASHCOMM:-1}"
export VLLM_ASCEND_ENABLE_NZ="${VLLM_ASCEND_ENABLE_NZ:-0}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export MULTI_STREAM_MEMORY_REUSE="${MULTI_STREAM_MEMORY_REUSE:-1}"
export CPU_AFFINITY_CONF="${CPU_AFFINITY_CONF:-1}"
export HCCL_OP_EXPANSION_MODE="${HCCL_OP_EXPANSION_MODE:-AIV}"
export HCCL_SOCKET_IFNAME="${HCCL_SOCKET_IFNAME:-enp48s3u1u1}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-${HCCL_SOCKET_IFNAME}}"
export HCCL_HOST_SOCKET_PORT_RANGE="${HCCL_HOST_SOCKET_PORT_RANGE:-auto}"
export HCCL_NPU_SOCKET_PORT_RANGE="${HCCL_NPU_SOCKET_PORT_RANGE:-auto}"
export HCCL_IF_BASE_PORT="${HCCL_IF_BASE_PORT:-50000}"
export HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-17340}"
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-7200}"
export HCCL_ASYNC_ERROR_HANDLING="${HCCL_ASYNC_ERROR_HANDLING:-0}"
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-256}"
export P2P_HCCL_BUFFSIZE="${P2P_HCCL_BUFFSIZE:-20}"
export VLLM_VERSION="${VLLM_VERSION:-0.26.0}"
export VERL_KIMI_MLP_CHUNK_TOKENS="${VERL_KIMI_MLP_CHUNK_TOKENS:-128}"

export NPUS_PER_NODE=$npus_per_node
KIMI_RAY_ARGS=()
if [[ "$run_mode" == single ]]; then
    kimi_single_ray_args
else
    : "${RAY_ADDRESS:?multinode mode must be started by ray_start_multi_nodes.sh}"
fi

rollout_data_dir="${VERL_ROLLOUT_DATA_DIR:-${PROJECT_ROOT}/rollout_dump/${start_time}}"
training_log_dir="${PROJECT_ROOT}/verl/logs"
mkdir -p "${rollout_data_dir}" "${training_log_dir}"
training_log="${training_log_dir}/${exp_name}.log"

max_prompt_length=1024
rollout_multimodal_token_margin="${VERL_VLLM_MULTIMODAL_TOKEN_MARGIN:-16384}"
rollout_max_model_len="${VERL_VLLM_MAX_MODEL_LEN:-$((max_prompt_length + max_response_length + rollout_multimodal_token_margin))}"
ppo_max_token_len=$(((max_prompt_length + max_response_length) / 2))

RAY_CONFIG=()
while IFS= read -r name; do
    case "$name" in
        PYTHON*|NON_MEGATRON|FSDP_TURBO*|VLLM*|VERL*|ASCEND*|HCCL*|GLOO*|KIMI*|OMP*|TOKENIZERS*|TMPDIR|XDG_CACHE_HOME|HF_HOME|HF_MODULES_CACHE|TRITON_CACHE_DIR|TORCHINDUCTOR_CACHE_DIR)
            RAY_CONFIG+=("++ray_kwargs.ray_init.runtime_env.env_vars.$name=\"${!name}\"") ;;
    esac
done < <(compgen -e)

DATA_CONFIG=(
    data.train_files="${train_file}"
    data.val_files="${test_file}"
    "data.train_batch_size=${train_batch_size}"
    "data.max_prompt_length=${max_prompt_length}"
    "data.max_response_length=${max_response_length}"
    data.filter_overlong_prompts=True
    data.filter_overlong_prompts_workers=null
    data.truncation=error
    data.image_key=images
    data.shuffle=False
    data.validation_shuffle=False
    data.trust_remote_code=True
)

TRAINER_CONFIG=(
    trainer.critic_warmup=0
    'trainer.logger=["console"]'
    trainer.project_name="${project_name}"
    trainer.experiment_name="${exp_name}"
    "trainer.n_gpus_per_node=${npus_per_node}"
    "trainer.nnodes=${nnodes}"
    trainer.balance_batch=True
    trainer.resume_from_path="${resume_from_path}"
    "trainer.resume_mode=${resume_mode}"
    "trainer.save_freq=${save_freq}"
    trainer.default_local_dir="${PROJECT_ROOT}/checkpoints/${project_name}/${exp_name}"
    "trainer.max_actor_ckpt_to_keep=${max_actor_ckpt_to_keep}"
    trainer.test_freq=-1
    trainer.val_before_train=False
    trainer.total_epochs=10
    "trainer.total_training_steps=${total_training_steps}"
    trainer.rollout_data_dir="${rollout_data_dir}"
)

MODEL_CONFIG=(
    "actor_rollout_ref.model.path=${model_path}"
    actor_rollout_ref.model.trust_remote_code=True
    actor_rollout_ref.model.use_remove_padding=False
    actor_rollout_ref.model.enable_gradient_checkpointing=False
)

ACTOR_CONFIG=(
    actor_rollout_ref.actor.checkpoint.save_contents="['model','optimizer','extra']"
    actor_rollout_ref.actor.checkpoint.load_contents="['model','optimizer','extra']"
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.optim.lr_scheduler_type=constant
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0
    actor_rollout_ref.actor.optim.weight_decay=0.01
    actor_rollout_ref.actor.optim.clip_grad=1.0
    actor_rollout_ref.actor.optim.optimizer=AdamW
    "actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}"
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.kl_loss_coef=0.01
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.use_dynamic_bsz=True
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len}"
)

ROLLOUT_CONFIG=(
    +actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.kimi_training_backend=fsdpturbo
    +actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.kimi_training_causal_conv1d=True
    +actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.kimi_kda_oproj_fp32_reduce=True
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    "actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp_size}"
    +actor_rollout_ref.rollout.engine_kwargs.vllm.mm_encoder_tp_mode=data
    "actor_rollout_ref.rollout.expert_parallel_size=${rollout_ep_size}"
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.ignore_eos=False
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6
    actor_rollout_ref.rollout.enable_rollout_routing_replay=True
    "actor_rollout_ref.rollout.max_model_len=${rollout_max_model_len}"
    "actor_rollout_ref.rollout.max_num_seqs=${rollout_max_num_seqs}"
    "actor_rollout_ref.rollout.n=${rollout_n}"
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    "actor_rollout_ref.rollout.max_num_batched_tokens=${rollout_max_num_batched_tokens}"
    actor_rollout_ref.rollout.free_cache_engine=True
    "actor_rollout_ref.rollout.enforce_eager=${rollout_enforce_eager}"
    actor_rollout_ref.rollout.enable_prefix_caching=False
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=256
    actor_rollout_ref.rollout.load_format=dummy
    actor_rollout_ref.rollout.logprobs_mode=processed_logprobs
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.temperature=1.0
    actor_rollout_ref.rollout.top_p=1.0
    actor_rollout_ref.rollout.top_k=-1
)

# Kimi MoE reload keeps the graph-captured runtime storage address stable, so
# graph mode can survive weight updates without recapture. Eager remains an
# explicit diagnostic fallback.
case "${rollout_enforce_eager,,}" in
    false|0|no|off)
        ROLLOUT_CONFIG+=(
            +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode=FULL_DECODE_ONLY
            '+actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_capture_sizes=[1,2,4,8,16]'
            +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.max_cudagraph_capture_size=16
            +actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.enable_sleep_mode_extra_cleanup=False
        )
        ;;
esac

# The actor and reference use the same model and sharding topology.
fsdp_modules='{vision_tower.encoder.blocks.\{*\}:{},mm_projector:{},language_model.model.embed_tokens:{},language_model.model.layers.\{*\}.self_attn:{},language_model.model.layers.\{*\}.mlp:{},language_model.model.layers.\{*\}.block_sparse_moe:{},language_model.lm_head:{}}'
FSDP_CONFIG=()
for role in actor ref; do
    prefix="actor_rollout_ref.$role"
    FSDP_CONFIG+=(
        "$prefix.strategy=fsdp_turbo"
        "$prefix.fsdp_config.model_dtype=bfloat16"
        "$prefix.fsdp_config.dtype=bfloat16"
        "+$prefix.fsdp_config.turbo_config.model.attn_implementation=eager"
        "+$prefix.fsdp_config.turbo_config.distributed.fsdp_plan.ignored_modules=[]"
        "++$prefix.fsdp_config.turbo_config.distributed.fsdp_plan.apply_modules=$fsdp_modules"
        "$prefix.fsdp_config.turbo_config.distributed.fsdp_plan.reduce_dtype=bf16"
        "$prefix.fsdp_config.turbo_config.distributed.fsdp_plan.fsdp_implementation=native"
        "$prefix.fsdp_config.turbo_config.distributed.fully_shard_parallel_size=$turbo_fsdp_size"
        "$prefix.fsdp_config.turbo_config.distributed.ulysses_parallel_size=1"
        "$prefix.fsdp_config.turbo_config.distributed.expert_parallel_size=$turbo_ep_size"
        "+$prefix.fsdp_config.turbo_config.distributed.ep_plan.apply_modules=['language_model.model.layers.{*}.block_sparse_moe.experts']"
        "+$prefix.fsdp_config.turbo_config.distributed.ep_plan.dispatcher=custom_ep_forward_fused"
        "$prefix.fsdp_config.turbo_config.distributed.expert_fully_shard_parallel_size=$turbo_efsdp_size"
        "+$prefix.fsdp_config.turbo_config.distributed.ep_plan.apply_efsdp_modules=['language_model.model.layers.{*}.block_sparse_moe.experts']"
        "$prefix.fsdp_config.reshard_after_forward=True"
        "$prefix.fsdp_config.offload_policy=False"
        "$prefix.fsdp_config.param_offload=False"
        "$prefix.entropy_from_logits_with_chunking=True"
        "$prefix.use_torch_compile=False"
    )
done
FSDP_CONFIG+=(
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    actor_rollout_ref.actor.fsdp_config.turbo_config.distributed.fsdp_plan.num_to_forward_prefetch=1
    actor_rollout_ref.actor.fsdp_config.turbo_config.distributed.fsdp_plan.num_to_backward_prefetch=1
    +actor_rollout_ref.actor.fsdp_config.turbo_config.distributed.ep_plan.fixed_router=False
    +actor_rollout_ref.actor.fsdp_config.turbo_config.memory.recompute=True
    '+actor_rollout_ref.actor.fsdp_config.turbo_config.memory.recompute_plan=["language_model.model.layers.{*}","vision_tower.encoder.blocks.{*}"]'
    actor_rollout_ref.ref.fsdp_config.wrap_policy.min_num_params=0
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$ppo_max_token_len"
)

echo "Kimi run: mode=${run_mode} experiment=${exp_name} model=${model_path} steps=${total_training_steps}"
echo "Kimi offload: actor_policy=False actor_param=False actor_optimizer=False ref_config_policy=False ref_forward_only_runtime_policy=True"
echo "Kimi topology: nodes=${nnodes} NPU/node=${npus_per_node} FSDP=${turbo_fsdp_size} EP=${turbo_ep_size} eFSDP=${turbo_efsdp_size} rollout_TP=${rollout_tp_size}"
echo "Kimi sequence: prompt=${max_prompt_length} response=${max_response_length} multimodal_margin=${rollout_multimodal_token_margin} model_len=${rollout_max_model_len} actor_ref_tokens_per_gpu=${ppo_max_token_len}"
echo "Kimi batches: train=${train_batch_size} rollout_n=${rollout_n} effective=$((train_batch_size * rollout_n)) ppo_mini=${ppo_mini_batch_size} micro_per_gpu=1 max_num_seqs=${rollout_max_num_seqs} max_batched_tokens=${rollout_max_num_batched_tokens}"
echo "Kimi training log: ${training_log}"

cd "${verl_path}"
python3 -m verl.trainer.main_ppo \
    --config-path=config \
    --config-name=ppo_trainer.yaml \
    model_engine=dp \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.rollout_correction.bypass_mode=False \
    "${DATA_CONFIG[@]}" \
    "${MODEL_CONFIG[@]}" \
    "${ACTOR_CONFIG[@]}" \
    "${ROLLOUT_CONFIG[@]}" \
    "${FSDP_CONFIG[@]}" \
    "${TRAINER_CONFIG[@]}" \
    "${RAY_CONFIG[@]}" \
    "${KIMI_RAY_ARGS[@]}" \
    "$@" 2>&1 | tee "${training_log}"
