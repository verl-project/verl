#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="${KIMI_BACKEND_ROOT:-/mnt/share/g00815994/kimi-k31}"
run_mode="${KIMI_RUN_MODE:-single}"
start_time="${VERL_RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
project_name="${VERL_PROJECT_NAME:-GRPO-Kimi-K3-A3B}"

model_path=${MODEL_PATH:-/mnt/share/g00815994/kimi-k31/models/Kimi-K3.1-top16-pruned-bf16-full}

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
rollout_dp_size=${ROLLOUT_DP:-1}
rollout_ep_size=$((rollout_tp_size * rollout_dp_size))
if ((turbo_ep_size <= 0 || rollout_tp_size <= 0 || rollout_dp_size <= 0 || turbo_fsdp_size % turbo_ep_size != 0 || turbo_fsdp_size % rollout_ep_size != 0)); then
    echo "Actor EP and rollout TP*DP must be positive and divide world size" >&2
    exit 2
fi
turbo_efsdp_size=$((turbo_fsdp_size / turbo_ep_size))
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
rollout_enforce_eager="${VERL_VLLM_ENFORCE_EAGER:-True}"
max_actor_ckpt_to_keep="${VERL_MAX_ACTOR_CKPT_TO_KEEP:-2}"
resume_mode="${VERL_RESUME_MODE:-disable}"
resume_from_path="${VERL_RESUME_FROM_PATH:-}"
train_file="${TRAIN_FILE:-/mnt/share/g00815994/dsv41/rl_data/countbenchqa_lite_train.parquet}"
test_file="${TEST_FILE:-/mnt/share/g00815994/dsv41/rl_data/countbenchqa_lite_val.parquet}"

kimi_fsdp_turbo_path="${PROJECT_ROOT}/source/FSDPTurbo"
kimi_actor_ep_dispatcher=custom_ep_forward_mc2
kimi_ref_ep_dispatcher=custom_ep_forward_fused

verl_path="${PROJECT_ROOT}/source/verl"
fsdp_turbo_path="$kimi_fsdp_turbo_path"
vllm_path="${PROJECT_ROOT}/source/vllm"
vllm_ascend_path="${PROJECT_ROOT}/source/vllm-ascend"
runtime_site="${PROJECT_ROOT}/source/python-site"
cann_python_site=/mnt/share/l00804615/cann/9.2.0/cann-9.2.0/python/site-packages

# Use the prepared four-layer checkpoint and its tokenizer directly.
tokenizer_path="$model_path"

export PYTHONPATH="${verl_path}:${fsdp_turbo_path}:${PYTHONPATH:+${PYTHONPATH}:}${KIMI_TORCH_NPU_SITE:+${KIMI_TORCH_NPU_SITE}:}${PROJECT_ROOT}/source/triton-ascend-kernels/src:${PROJECT_ROOT}/.python_deps/modelopt-0.46.0:${runtime_site}:${verl_path}:${vllm_path}:${vllm_ascend_path}:${fsdp_turbo_path}:${fsdp_turbo_path}/examples:${cann_python_site}"
export FSDP_TURBO_ROOT="${fsdp_turbo_path}"
export VLLM_ASCEND_PATH="${vllm_ascend_path}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-${visible_devices_default}}"
# Retain free actor buffers between reference inference and actor update.
export VERL_REF_TO_ACTOR_EMPTY_CACHE="${VERL_REF_TO_ACTOR_EMPTY_CACHE:-0}"
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
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export VLLM_ASCEND_ENABLE_FLASHCOMM1="${VLLM_ASCEND_ENABLE_FLASHCOMM1:-1}"
export VLLM_ASCEND_ENABLE_NZ="${VLLM_ASCEND_ENABLE_NZ:-0}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export MULTI_STREAM_MEMORY_REUSE="${MULTI_STREAM_MEMORY_REUSE:-1}"
export CPU_AFFINITY_CONF="${CPU_AFFINITY_CONF:-0}"
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
export VLLM_VERSION="${VLLM_VERSION:-0.30.0}"
export VERL_KIMI_MLP_CHUNK_TOKENS="${VERL_KIMI_MLP_CHUNK_TOKENS:-128}"

export NPUS_PER_NODE=$npus_per_node
KIMI_RAY_ARGS=()
if [[ "$run_mode" == single ]]; then
    KIMI_RAY_ARGS=(
        '++ray_kwargs.ray_init.address=local'
        "++ray_kwargs.ray_init.num_cpus=${RAY_NUM_CPUS:-64}"
        "++ray_kwargs.ray_init.resources.NPU=$npus_per_node"
        "++ray_kwargs.ray_init.object_store_memory=${RAY_OBJECT_STORE_MEMORY:-8589934592}"
    )
else
    : "${RAY_ADDRESS:?start the multinode Ray cluster first}"
    KIMI_RAY_ARGS=("++ray_kwargs.ray_init.address=${RAY_ADDRESS}")
fi

rollout_data_dir="${VERL_ROLLOUT_DATA_DIR:-${PROJECT_ROOT}/rollout_dump/${start_time}}"
training_log_dir="${PROJECT_ROOT}/logs"
mkdir -p "${rollout_data_dir}" "${training_log_dir}"
training_log="${training_log_dir}/${exp_name}.log"

max_prompt_length=${VERL_MAX_PROMPT_LENGTH:-1024}
rollout_multimodal_token_margin="${VERL_VLLM_MULTIMODAL_TOKEN_MARGIN:-16384}"
rollout_max_model_len="${VERL_VLLM_MAX_MODEL_LEN:-$((max_prompt_length + max_response_length + rollout_multimodal_token_margin))}"
ppo_max_token_len=${VERL_MAX_TOKEN_LEN_PER_GPU:-$(((max_prompt_length + max_response_length) / 2))}

RAY_CONFIG=()
while IFS= read -r name; do
    case "$name" in
        TASK_QUEUE_ENABLE|PYTHON*|NON_MEGATRON|FSDP_TURBO*|VLLM*|VERL*|ASCEND*|HCCL*|GLOO*|KIMI*|OMP*|CPU_AFFINITY_CONF|TOKENIZERS*|TMPDIR|XDG_CACHE_HOME|HF_HOME|HF_MODULES_CACHE|HF_DATASETS_CACHE|TRITON_CACHE_DIR|TORCHINDUCTOR_CACHE_DIR)
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
    +data.apply_chat_template_kwargs.thinking=False
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
    "trainer.total_epochs=${VERL_TOTAL_EPOCHS:-10}"
    "trainer.total_training_steps=${total_training_steps}"
    trainer.rollout_data_dir="${rollout_data_dir}"
)

MODEL_CONFIG=(
    +actor_rollout_ref.model.override_config.mm_merge_chunk_tokens=256
    "actor_rollout_ref.model.path=${model_path}"
    "actor_rollout_ref.model.tokenizer_path=${tokenizer_path}"
    actor_rollout_ref.model.trust_remote_code=True
    "actor_rollout_ref.model.use_remove_padding=${KIMI_USE_TND:-False}"
    actor_rollout_ref.model.enable_gradient_checkpointing=False
    # Training-only HF overrides; rollout KV cache remains enabled in vLLM.
    "+actor_rollout_ref.model.override_config.text_config.dsa_config.implementation=${VERL_DSA_IMPLEMENTATION:-ascendc}"
    +actor_rollout_ref.model.override_config.text_config.kda_implementation=ascendc
    +actor_rollout_ref.model.override_config.text_config.causal_conv1d_implementation=ascendc
    +actor_rollout_ref.model.override_config.text_config.situ_glu_implementation=ascendc
    +actor_rollout_ref.model.override_config.text_config.attn_res_implementation=ascendc
    +actor_rollout_ref.model.override_config.text_config.skip_kda_recompute=True
    +actor_rollout_ref.model.override_config.text_config.use_cache=False
    # skip_flash_attn_recompute is not wired in the NPU FA backend; do not set a no-op flag.
)

ACTOR_CONFIG=(
    actor_rollout_ref.actor.checkpoint.save_contents="['model','optimizer','extra']"
    actor_rollout_ref.actor.checkpoint.load_contents="['model','optimizer','extra']"
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.optim.lr_scheduler_type=constant
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0
    actor_rollout_ref.actor.optim.weight_decay=0.01
    actor_rollout_ref.actor.optim.clip_grad=1.0
    "actor_rollout_ref.actor.optim.optimizer=${VERL_OPTIMIZER:-AdamW}"
    "actor_rollout_ref.actor.optim.optimizer_impl=${VERL_OPTIMIZER_IMPL:-torch.optim}"
    "actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}"
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${VERL_PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.kl_loss_coef=0.01
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    "actor_rollout_ref.actor.use_dynamic_bsz=${KIMI_USE_DYNAMIC_BSZ:-False}"
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len}"
)

ROLLOUT_CONFIG=(
    +actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.rl_config.enabled=True
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${KIMI_USE_DYNAMIC_BSZ:-False}"
    "actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp_size}"
    "actor_rollout_ref.rollout.data_parallel_size=${rollout_dp_size}"
    +actor_rollout_ref.rollout.engine_kwargs.vllm.mm_encoder_tp_mode=data
    "actor_rollout_ref.rollout.expert_parallel_size=${rollout_ep_size}"
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.ignore_eos=False
    actor_rollout_ref.rollout.gpu_memory_utilization=0.35
    +actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_memory_bytes=2147483648
    "actor_rollout_ref.rollout.enable_rollout_routing_replay=${KIMI_ENABLE_R3:-False}"
    "actor_rollout_ref.rollout.max_model_len=${rollout_max_model_len}"
    "actor_rollout_ref.rollout.max_num_seqs=${rollout_max_num_seqs}"
    "actor_rollout_ref.rollout.n=${rollout_n}"
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    "actor_rollout_ref.rollout.max_num_batched_tokens=${rollout_max_num_batched_tokens}"
    actor_rollout_ref.rollout.free_cache_engine=True
    "actor_rollout_ref.rollout.enforce_eager=${rollout_enforce_eager}"
    "actor_rollout_ref.rollout.enable_prefix_caching=${VERL_VLLM_ENABLE_PREFIX_CACHING:-False}"
    "actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=${VERL_WEIGHT_SYNC_BUCKET_MB:-1024}"
    actor_rollout_ref.rollout.load_format=dummy
    actor_rollout_ref.rollout.logprobs_mode=processed_logprobs
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.temperature=1.0
    actor_rollout_ref.rollout.top_p=1.0
    actor_rollout_ref.rollout.top_k=-1
)

# Optional reference-inference profile. Actor/FSDP settings remain independent.
if [[ "${KIMI_INFERENCE_REFERENCE_PROFILE:-0}" == 1 ]]; then
    ROLLOUT_CONFIG+=(
        +actor_rollout_ref.rollout.engine_kwargs.vllm.tokenizer_mode=kimi_k3
        +actor_rollout_ref.rollout.engine_kwargs.vllm.block_size=1024
        +actor_rollout_ref.rollout.engine_kwargs.vllm.mamba_ssm_cache_dtype=float32
        +actor_rollout_ref.rollout.engine_kwargs.vllm.cpu_offload_gb=0
        +actor_rollout_ref.rollout.engine_kwargs.vllm.enable_ep_weight_filter=True
    )
fi

# Kimi MoE reload keeps the graph-captured runtime storage address stable, so
# graph mode can survive weight updates without recapture. Eager remains an
# explicit diagnostic fallback.
case "${rollout_enforce_eager,,}" in
    false|0|no|off)
        ROLLOUT_CONFIG+=(
            +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode=FULL_DECODE_ONLY
            "+actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_capture_sizes=${KIMI_GRAPH_CAPTURE_SIZES:-[1,2,4,8,16,32]}"
            "+actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.max_cudagraph_capture_size=${KIMI_GRAPH_MAX_CAPTURE_SIZE:-32}"
            "+actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.rl_config.sleep_mode_extra_cleanup=${VERL_VLLM_SLEEP_EXTRA_CLEANUP:-True}"
        )
        ;;
esac

# The actor and reference use the same model and sharding topology.
fsdp_modules='{vision_tower.encoder.blocks.\{*\}:{},mm_projector:{},language_model.model.embed_tokens:{},language_model.model.layers.\{*\}.self_attn:{},language_model.model.layers.\{*\}.mlp:{},language_model.model.layers.\{*\}.block_sparse_moe:{},language_model.lm_head:{}}'
FSDP_CONFIG=()
for role in actor ref; do
    prefix="actor_rollout_ref.$role"
    ep_dispatcher=$kimi_ref_ep_dispatcher
    if [[ "$role" == actor ]]; then ep_dispatcher=$kimi_actor_ep_dispatcher; fi
    FSDP_CONFIG+=(
        "$prefix.strategy=fsdp_turbo"
        "$prefix.fsdp_config.model_dtype=bfloat16"
        "$prefix.fsdp_config.dtype=bfloat16"
        "+$prefix.fsdp_config.turbo_config.model.attn_implementation=flash_attention_2"
        "+$prefix.fsdp_config.turbo_config.distributed.fsdp_plan.ignored_modules=[]"
        "++$prefix.fsdp_config.turbo_config.distributed.fsdp_plan.apply_modules=$fsdp_modules"
        "$prefix.fsdp_config.turbo_config.distributed.fsdp_plan.reduce_dtype=bf16"
        "$prefix.fsdp_config.turbo_config.distributed.fsdp_plan.num_to_forward_prefetch=1"
        "$prefix.fsdp_config.turbo_config.distributed.fsdp_plan.num_to_backward_prefetch=1"
        "+$prefix.fsdp_config.turbo_config.distributed.fsdp_plan.pin_memory=${KIMI_TRAIN_PIN_MEMORY:-False}"
        "+$prefix.fsdp_config.turbo_config.data.pin_memory=${KIMI_TRAIN_PIN_MEMORY:-False}"
        "$prefix.fsdp_config.turbo_config.distributed.fsdp_plan.fsdp_implementation=native"
        "$prefix.fsdp_config.turbo_config.distributed.fully_shard_parallel_size=$turbo_fsdp_size"
        "$prefix.fsdp_config.turbo_config.distributed.ulysses_parallel_size=1"
        "$prefix.fsdp_config.turbo_config.distributed.expert_parallel_size=$turbo_ep_size"
        "+$prefix.fsdp_config.turbo_config.distributed.ep_plan.apply_modules=['language_model.model.layers.{*}.block_sparse_moe.experts']"
        "+$prefix.fsdp_config.turbo_config.distributed.ep_plan.dispatcher=$ep_dispatcher"
        "++$prefix.fsdp_config.turbo_config.distributed.ep_plan.efsdp_shard_placements=[[gate_up_proj,2],[down_proj,1]]"
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
    +actor_rollout_ref.actor.fsdp_config.turbo_config.distributed.fsdp_plan.prefetch_wait_a2a=True
    "actor_rollout_ref.actor.fsdp_config.offload_policy=${KIMI_CPU_OFFLOAD:-True}"
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    +actor_rollout_ref.actor.fsdp_config.turbo_config.distributed.ep_plan.reuse_dispatch_splits=True
    +actor_rollout_ref.actor.fsdp_config.turbo_config.distributed.ep_plan.fixed_router=False
    +actor_rollout_ref.actor.fsdp_config.turbo_config.memory.recompute=True
    "+actor_rollout_ref.actor.fsdp_config.turbo_config.memory.swap_activation=${KIMI_SWAP_ACTIVATION:-True}"
    '+actor_rollout_ref.actor.fsdp_config.turbo_config.memory.swap_activation_plan.apply_modules=["language_model.model.layers.{*}"]'
    '+actor_rollout_ref.actor.fsdp_config.turbo_config.memory.recompute_plan=["language_model.model.layers.{*}","vision_tower.encoder.blocks.{*}"]'
    actor_rollout_ref.ref.fsdp_config.wrap_policy.min_num_params=0
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$ppo_max_token_len"
    "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$ppo_max_token_len"
    "actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${KIMI_USE_DYNAMIC_BSZ:-False}"
)

# Explicit policy also controls forward-only reference residency in FSDP-Turbo.
if [[ -n "${KIMI_CPU_OFFLOAD:-}" ]]; then
    for role in actor ref; do
        FSDP_CONFIG+=("+actor_rollout_ref.$role.fsdp_config.turbo_config.distributed.fsdp_plan.cpu_offload=${KIMI_CPU_OFFLOAD}")
    done
fi

echo "Kimi run: mode=${run_mode} experiment=${exp_name} model=${model_path} steps=${total_training_steps}"
echo "Kimi offload: actor_policy=${KIMI_CPU_OFFLOAD:-True} param=False optimizer=False activation_swap=${KIMI_SWAP_ACTIVATION:-True} ref_cpu_policy=${KIMI_CPU_OFFLOAD:-auto}"
echo "Kimi topology: nodes=${nnodes} NPU/node=${npus_per_node} FSDP=${turbo_fsdp_size} EP=${turbo_ep_size} eFSDP=${turbo_efsdp_size} rollout_TP=${rollout_tp_size} rollout_DP=${rollout_dp_size} rollout_EP=${rollout_ep_size}"
echo "Kimi sequence: prompt=${max_prompt_length} response=${max_response_length} multimodal_margin=${rollout_multimodal_token_margin} model_len=${rollout_max_model_len} actor_ref_tokens_per_gpu=${ppo_max_token_len}"
echo "Kimi batches: train=${train_batch_size} rollout_n=${rollout_n} effective=$((train_batch_size * rollout_n)) ppo_mini=${ppo_mini_batch_size} micro_per_gpu=1 max_num_seqs=${rollout_max_num_seqs} max_batched_tokens=${rollout_max_num_batched_tokens}"
echo "Kimi training log: ${training_log}"

cd "${verl_path}"
kimi_trainer_command() {
    if [[ "${KIMI_CONFIG_DRY_RUN:-0}" == 1 ]]; then
        printf '%s\0' "$@" > "${KIMI_CONFIG_DRY_RUN_OUTPUT:?set dry-run output}"
        return 0
    fi
    python3 "$@"
}
kimi_trainer_command -m verl.trainer.main_ppo \
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
