set -x
ENGINE=${1:-vllm}
DEVICE=${DEVICE:-$(python3 -c 'import torch_npu' 2>/dev/null && echo npu || echo gpu)}

MODEL_ID=${MODEL_ID:-Qwen/Qwen3-30B-A3B-Instruct-2507}
MODEL_PATH=${MODEL_PATH:-${HOME}/.cache/models/${MODEL_ID}}
TRAIN_FILE=$HOME/data/gsm8k/train.parquet
TEST_FILE=$HOME/data/gsm8k/test.parquet
max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 2))
rollout_max_num_seqs=$((128))
n_devices_per_node=$((8))
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 1))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 3))

SCRIPT_NAME="$(basename -- "${BASH_SOURCE[0]}" .sh)"
LOG_DIR=/root/.cache/nightly_log/$SCRIPT_NAME
mkdir -p $LOG_DIR

ROLLOUT_FILE=$LOG_DIR/$(date "+%Y%m%d_%H%M%S")
case "${DEVICE}" in
    gpu)
        ;;
    npu)
        export TASK_QUEUE_ENABLE=1
        export HCCL_OP_EXPANSION_MODE="AIV"
        export VLLM_USE_V1=1
        export VLLM_VERSION=0.18.0
        export VLLM_ASCEND_ENABLE_NZ=0
        export HCCL_BUFFSIZE=610
        export CKPT_DIR="./c0kpt30b"
        export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:1024
        export CUDA_DEVICE_MAX_CONNECTIONS=1
        n_devices_per_node=16
        ;;
    *)
        echo "Unsupported DEVICE=${DEVICE}. Expected 'gpu' or 'npu'." >&2
        exit 1
        ;;
esac

DATA=(
    algorithm.use_kl_in_reward=False
    algorithm.adv_estimator=grpo
    data.train_files=${TRAIN_FILE}
    data.val_files=${TEST_FILE}
    data.train_batch_size=32
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.filter_overlong_prompts=False
    data.truncation='error'
    trainer.rollout_data_dir="${ROLLOUT_FILE}"
)

MODEL=(
    actor_rollout_ref.model.path=${MODEL_PATH}
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.model.use_remove_padding=True
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.veomni.param_offload=True
    actor_rollout_ref.actor.veomni.optimizer_offload=True
    actor_rollout_ref.actor.ppo_mini_batch_size=16
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0.001
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.use_torch_compile=False
    actor_rollout_ref.actor.veomni.fsdp_size=-1
    actor_rollout_ref.actor.veomni.expert_parallel_size=8
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len}
)

ROLLOUT=(
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.data_parallel_size=1
    actor_rollout_ref.rollout.expert_parallel_size=1
    actor_rollout_ref.rollout.tensor_model_parallel_size=8
    actor_rollout_ref.rollout.name=$ENGINE
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len}
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.enable_prefix_caching=True
    actor_rollout_ref.rollout.max_model_len=$((max_prompt_length + max_response_length))
    actor_rollout_ref.rollout.max_num_batched_tokens=$((1024))
    actor_rollout_ref.rollout.max_num_seqs=${rollout_max_num_seqs}
    actor_rollout_ref.rollout.enforce_eager=False
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.n=4
    +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_capture_sizes="[1, 8, 16, 32, 40, 48, 64, 96, 128, 256]"
    +actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_cache_gb=0
)

REF=(
    actor_rollout_ref.ref.veomni.param_offload=True
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.ref.veomni.param_offload=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len}
)

TRAINER=(
    trainer.critic_warmup=0
    trainer.logger=console
    trainer.project_name='verl_qwen3_veomni'
    trainer.experiment_name='qwen3_30b_veomni'
    trainer.n_gpus_per_node=${n_devices_per_node}
    trainer.nnodes=1
    trainer.save_freq=-1
    trainer.default_local_dir=$CKPT_DIR
    trainer.test_freq=-1
    trainer.total_training_steps=15
    actor_rollout_ref.model.trust_remote_code=True
)

if [ "${DEVICE}" = "npu" ]; then
    ROLLOUT+=(
        +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode="FULL_DECODE_ONLY"
    )
fi

EXTRA=(
    model_engine=veomni
)

########################### launch ###########################
python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${TRAINER[@]}" \
    "${EXTRA[@]}" \
    "$@" 2>&1 | tee $LOG_DIR/$SCRIPT_NAME.log
