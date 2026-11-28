#!/usr/bin/env bash
# GRPO | Qwen3-30B-A3B (MoE) | DAPO-Math-17k | Megatron training | MXFP8 end to end | AMD Instinct MI350 / MI355 (gfx950)
#
# ROCm version of examples/mxfp8/run_qwen3_30b_a3b_megatron.sh for one 4-GPU node. PRECISION=mxfp8 trains with
# Transformer Engine's MXFP8 recipe and serves the rollout in MXFP8, experts included; PRECISION=bf16 runs the same
# job in BF16 as the baseline. See docs/low_precision/mxfp8.md and docs/low_precision/mxfp8_rocm.md.
# Run inside the docker/rocm/Dockerfile.rocm103 image (vLLM 0.34 with its ROCm patches), from the verl repo root.
#
# On gfx950, Transformer Engine runs MXFP8 GEMMs on HipKittens when every GEMM dimension that carries tokens or
# features is a multiple of 256 (K a multiple of 128). Any GEMM that misses it falls back to hipBLASLt, which has no
# prebuilt kernel for the MXFP8 scale layout TE passes, and rocRoller then generates kernels on the CPU at run time
# (minutes each, per process). The MXFP8 settings below keep every training GEMM on HipKittens:
#   - expert tensor parallel 1 (ACTOR_ETP): with the default ETP = TP = 2 each rank holds 768 / 2 = 384 columns of
#     every expert, and the expert dgrad / wgrad GEMMs go to rocRoller whatever the token padding;
#   - each expert's tokens padded to 256 (moe_router_padding_for_fp8; verl raises Megatron's MXFP8 alignment from
#     32 to 256 on ROCm, apply_patch_rocm_mxfp8_expert_alignment in verl/models/mcore/rocm_patch.py);
#   - the packed sequence padded to a multiple of PAD_TO_LENGTH_BUCKET tokens (actor pad_to_length) for the dense
#     attention GEMMs; verl alone pads it to 128;
#   - experts on HipKittens' grouped MXFP8 GEMM (NVTE_USE_HIPKITTENS_GROUPED_GEMM=1) instead of one GEMM per expert.
# Qwen3-30B-A3B's widths fit at TP=1 / ETP=1 (hidden 2048, qkv 5120, attention output 4096, expert 1536 (gate+up)
# and 768) and at TP=2 (qkv 2560 and attention output 2048 per rank). Set ROCROLLER_SAVE_ASSEMBLY=1 to have
# rocRoller write a .s file in the worker's working directory for every kernel it generates; none should appear.
#
# Other ROCm settings: NVTE_ROCM_ENABLE_MXFP8=1 enables TE's MXFP8 path; Megatron attention_backend=fused matches the
# image's NVTE_FLASH_ATTN=0 / NVTE_FUSED_ATTN=1; VLLM_ROCM_MXFP8_BLOCK32_SCALES=0 (also the image's default) keeps
# vLLM's per-row MXFP8 scales so weight refit works; system python (the uv extras in uv.lock are CUDA builds). The
# MXFP8 MoE rollout uses vLLM's Triton kernel; export VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MOE=1 before running
# to put it on AITER.

set -xeuo pipefail
export CUDA_DEVICE_MAX_CONNECTIONS=1

########################### user-adjustable ###########################
INFER_BACKEND=${INFER_BACKEND:-vllm}
PRECISION=${PRECISION:-mxfp8}

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-30B-A3B}
TRAIN_FILE=${TRAIN_FILE:-$HOME/data/dapo-math-17k/train.parquet}
# only read when TEST_FREQ > 0
VAL_FILE=${VAL_FILE:-$TRAIN_FILE}
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-4}

train_batch_size=${TRAIN_BATCH_SIZE:-64}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-32}
max_prompt_length=${MAX_PROMPT_LENGTH:-1024}
max_response_length=${MAX_RESPONSE_LENGTH:-8192}
max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU:-20480}
# token budget per GPU for the forward-only log-prob passes (old log-probs, reference); defaults to the training one
logprob_max_token_len_per_gpu=${LOGPROB_MAX_TOKEN_LEN_PER_GPU:-$max_token_len_per_gpu}
# data shuffle seed; set it to pair steps across runs (null: unseeded, as upstream)
seed=${SEED:-null}

actor_lr=${ACTOR_LR:-1e-6}
kl_loss_coef=${KL_LOSS_COEF:-0.001}
entropy_coeff=${ENTROPY_COEFF:-0}

# 4 GPUs: dense layers TP1 x DP4, experts EP4 x ETP1 (32 experts per GPU)
actor_tp=${ACTOR_TP:-1}
actor_pp=${ACTOR_PP:-1}
actor_ep=${ACTOR_EP:-4}
actor_etp=${ACTOR_ETP:-1}
# round every packed micro-batch up to a multiple of this many tokens (MXFP8 only)
pad_to_length_bucket=${PAD_TO_LENGTH_BUCKET:-512}
# offload parameters and optimizer state while the rollout runs
offload=${OFFLOAD:-True}
# keep the whole optimizer on the CPU (on 4x MI355X, Adam's FP32 state on the GPU runs update_actor out of memory)
optimizer_cpu_offload=${OPTIMIZER_CPU_OFFLOAD:-True}
# activation recompute: selective (RECOMPUTE_MODULES only) | full (every layer, as upstream) | none
recompute=${RECOMPUTE:-selective}
recompute_modules=${RECOMPUTE_MODULES:-"[core_attn,moe_act]"}

rollout_tp=${ROLLOUT_TP:-2}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.4}
rollout_max_num_seqs=${ROLLOUT_MAX_NUM_SEQS:-256}
rollout_n=${ROLLOUT_N:-8}

total_epochs=${TOTAL_EPOCHS:-1}
total_training_steps=${TOTAL_TRAINING_STEPS:-null}
save_freq=${SAVE_FREQ:--1}
test_freq=${TEST_FREQ:--1}

project_name=${PROJECT_NAME:-verl_grpo_dapo_mxfp8}
experiment_name=${EXPERIMENT_NAME:-qwen3_30b_a3b_megatron_${INFER_BACKEND}_${PRECISION}_rocm}
########################### end user-adjustable ###########################

if [ "$(python3 -c 'import torch; print(int(torch.version.hip is not None))')" != 1 ]; then
    echo "This is the ROCm script; on NVIDIA GPUs use examples/mxfp8/run_qwen3_30b_a3b_megatron.sh" >&2
    exit 1
fi

########################### parameter arrays ###########################

TF=actor_rollout_ref.actor.megatron.override_transformer_config
REF_TF=actor_rollout_ref.ref.megatron.override_transformer_config
case "${PRECISION}" in
    mxfp8)
        export NVTE_ROCM_ENABLE_MXFP8=1
        export NVTE_USE_HIPKITTENS_GROUPED_GEMM=1
        PRECISION_ARGS=(
            # MXFP8 GEMMs in training (Transformer Engine); the weights stay in BF16
            ++${TF}.fp8=e4m3
            ++${TF}.fp8_recipe=mxfp8
            # pad each expert's tokens to the FP8 GEMM alignment (256 on ROCm, set by verl)
            ++${TF}.moe_router_padding_for_fp8=True
            # pad the packed sequence so the dense GEMMs' token dimension is a multiple of 256
            actor_rollout_ref.actor.megatron.pad_to_length=True
            actor_rollout_ref.actor.megatron.pad_to_length_bucket=${pad_to_length_bucket}
            # MXFP8 rollout: verl quantizes the synced weights; lm_head, embeddings and MoE routers stay in BF16
            actor_rollout_ref.rollout.quantization.linear=MXFP8
            actor_rollout_ref.rollout.quantization.moe=MXFP8
            # the reference model's forward runs in MXFP8 too, as upstream (where the ref inherits the actor's
            # overrides); the ref inherits pad_to_length from the actor
            ++${REF_TF}.fp8=e4m3
            ++${REF_TF}.fp8_recipe=mxfp8
            ++${REF_TF}.moe_router_padding_for_fp8=True
        )
        ;;
    bf16)
        # BF16 expert GEMMs on Transformer Engine's CK grouped GEMM (one launch for all local experts; needs TE with
        # ROCm/TransformerEngine#783, as in rocm/primus:v26.10). Without it the experts run one hipBLASLt GEMM per
        # expert: TE's Triton grouped GEMM (NVTE_USE_GROUPED_GEMM_TRITON=1) is skipped when
        # fuse_wgrad_accumulation is on, which Megatron enables.
        export NVTE_USE_CK_GROUPED_GEMM=1
        PRECISION_ARGS=()
        ;;
    *)
        echo "PRECISION must be mxfp8 or bf16, got '${PRECISION}'" >&2
        exit 1
        ;;
esac

export VLLM_ROCM_MXFP8_BLOCK32_SCALES=0

case "${recompute}" in
    full)
        RECOMPUTE_ARGS=(
            ++${TF}.recompute_granularity=full
            ++${TF}.recompute_method=uniform
            ++${TF}.recompute_num_layers=1
        )
        ;;
    selective)
        RECOMPUTE_ARGS=(
            ++${TF}.recompute_granularity=selective
            "++${TF}.recompute_modules=${recompute_modules}"
        )
        ;;
    none)
        RECOMPUTE_ARGS=()
        ;;
    *)
        echo "RECOMPUTE must be full, selective or none, got '${recompute}'" >&2
        exit 1
        ;;
esac

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="$TRAIN_FILE"
    data.val_files="$VAL_FILE"
    data.train_batch_size=${train_batch_size}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.filter_overlong_prompts=True
    data.truncation='error'
    data.seed=${seed}
)

MODEL=(
    actor_rollout_ref.model.path="$MODEL_PATH"
)

OPTIM=+actor_rollout_ref.actor.optim.override_optimizer_config
ACTOR=(
    actor_rollout_ref.actor.optim.lr=${actor_lr}
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${max_token_len_per_gpu}
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef}
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff}
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${actor_tp}
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${actor_pp}
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${actor_ep}
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${actor_etp}
    actor_rollout_ref.actor.megatron.param_offload=${offload}
    actor_rollout_ref.actor.megatron.optimizer_offload=${offload}
    ${OPTIM}.optimizer_cpu_offload=${optimizer_cpu_offload}
    ${OPTIM}.overlap_cpu_optimizer_d2h_h2d=${optimizer_cpu_offload}
    ${OPTIM}.use_precision_aware_optimizer=${optimizer_cpu_offload}
    ${OPTIM}.optimizer_offload_fraction=1.0
    "${RECOMPUTE_ARGS[@]}"
    ++${TF}.moe_grouped_gemm=True
    ++${TF}.moe_token_dispatcher_type=alltoall
    ++${TF}.moe_router_dtype=fp32
    # Megatron asserts attention_backend agrees with the image's NVTE_FLASH_ATTN=0 / NVTE_FUSED_ATTN=1
    ++${TF}.attention_backend=fused
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=${INFER_BACKEND}
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp}
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util}
    actor_rollout_ref.rollout.max_num_seqs=${rollout_max_num_seqs}
    actor_rollout_ref.rollout.n=${rollout_n}
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${logprob_max_token_len_per_gpu}
)

REF=(
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${logprob_max_token_len_per_gpu}
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${actor_tp}
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${actor_pp}
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=${actor_ep}
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=${actor_etp}
    actor_rollout_ref.ref.megatron.param_offload=${offload}
    # Setting any ref override_transformer_config key replaces the dict the ref would otherwise inherit from the
    # actor (verl's megatron_ref.yaml), so the ref gets the actor's forward settings explicitly here; its precision
    # is set in PRECISION_ARGS (MXFP8 with PRECISION=mxfp8).
    ++${REF_TF}.moe_grouped_gemm=True
    ++${REF_TF}.moe_token_dispatcher_type=alltoall
    ++${REF_TF}.moe_router_dtype=fp32
    # Megatron asserts attention_backend agrees with the image's NVTE_FLASH_ATTN=0 / NVTE_FUSED_ATTN=1
    ++${REF_TF}.attention_backend=fused
)

TRAINER=(
    trainer.critic_warmup=0
    trainer.logger='["console","wandb"]'
    trainer.project_name=${project_name}
    trainer.experiment_name=${experiment_name}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.save_freq=${save_freq}
    trainer.test_freq=${test_freq}
    trainer.val_before_train=False
    trainer.total_epochs=${total_epochs}
    trainer.total_training_steps=${total_training_steps}
)

EXTRA=(
    model_engine=megatron
)

########################### launch ###########################
# system python of the ROCm image for the driver and every Ray worker
python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${TRAINER[@]}" \
    "${PRECISION_ARGS[@]}" \
    "${EXTRA[@]}" \
    ray_kwargs.ray_init.runtime_env.py_executable=null \
    "$@"
