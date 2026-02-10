#!/bin/bash
#SBATCH --job-name=sapo-30B
#SBATCH --partition=main
#SBATCH --nodes=1                # Number of nodes
#SBATCH --ntasks-per-node=1      # One task per node
#SBATCH --cpus-per-task=128      # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:8
#SBATCH --gpus-per-node=8
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=500:00:00
#SBATCH --output=logs/sapo/30B/frugal_math/%x_%j.out
#SBATCH --error=logs/sapo/30B/frugal_math/%x_%j.err

# This script runs the training of RL on multi-nodes. It does resume automatically from latest checkpoint if the run crashes.
# Example run with Qwen3-30B SAPO with new model engine
set -x

export WANDB_API_KEY=YOUR_WANDB_API_KEY_HERE
ENV_NAME=verl_0_6_1

# Ensure Python can import the top-level verl package even when the script is relocated by Slurm
if [[ -n "$SLURM_SUBMIT_DIR" && -d "$SLURM_SUBMIT_DIR" ]]; then
    cd "$SLURM_SUBMIT_DIR"
    SCRIPT_SOURCE_DIR="$SLURM_SUBMIT_DIR"
else
    SCRIPT_SOURCE_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)
fi
REPO_ROOT=$(cd -- "$SCRIPT_SOURCE_DIR/../.." >/dev/null 2>&1 && pwd)
VERL_REPO_ROOT="$REPO_ROOT"

add_repo_to_pythonpath() {
    if [[ -z "$PYTHONPATH" ]]; then
        export PYTHONPATH="$VERL_REPO_ROOT"
    else
        case ":$PYTHONPATH:" in
            *":$VERL_REPO_ROOT:"*) ;;
            *) export PYTHONPATH="$VERL_REPO_ROOT:$PYTHONPATH" ;;
        esac
    fi
}

add_repo_to_pythonpath

# can make training faster depending on clusters
export NCCL_IBEXT_DISABLE=1
export NCCL_NVLS_ENABLE=1
export NCCL_IB_HCA=mlx5
export UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1

# Determine how many nodes were allocated. 
NNODES=${SLURM_JOB_NUM_NODES}
export NNODES

# Determine how many GPUs we actually have on the master node.
# Carefull! Assumes all nodes have same number of GPUs! 
# SLURM sets SLURM_GPUS_PER_NODE only when #SBATCH --gpus-per-node is used, not with --gres.
# uncomment below line to manually set number of gpus per node if not using --gpus-per-node
# export SLURM_GPUS_PER_NODE=8
# SLURM_GPUS_PER_NODE=${SLURM_GPUS_PER_NODE:-$(nvidia-smi -L | wc -l)} # 8
# export SLURM_GPUS_PER_NODE
echo "SLURM_GPUS_PER_NODE: $SLURM_GPUS_PER_NODE"

# Set DATA_ROOT to current working directory if not set
DATA_ROOT=${DATA_ROOT:-$PWD}
echo "DATA_ROOT: $DATA_ROOT"

# wandb logging
backend=fsdp # fsdp, fsdp2, megatron
project_name=RL4LLM
# experiment_name=qwen3-30B-base-sapo-$backend
experiment_name=qwen3-30B-base-vanilla-$backend
default_local_dir=$DATA_ROOT/checkpoint/$project_name/$experiment_name

# ===================================== Algorithm =====================================
adv_estimator=grpo
loss_mode=sapo # explicitly specify sapo! default is vanilla and is not compatible with SAPO. It uses clipping instead of smoothing.

# reference policy
use_kl_in_reward=False
kl_coef=0.001
use_kl_loss=False
kl_loss_coef=0.001

# Positive and negative tau for smoothing function in SAPO (https://arxiv.org/pdf/2511.20347)
# default values used in the paper with Qwen3-30B-A3B-Base
# clipping is not used in SAPO!
tau_pos=1.0
tau_neg=1.05

actor_lr=1e-6
critic_lr=2e-6
gae_gamma=1.0
gae_lam=0.95
critic_warmup=0

# ===================================== Data/Model =====================================

first_time_dataset_prep=true
HF_DATA_PATH="BytedTsinghua-SIA/DAPO-Math-17k"
STAGE="stage-1"

if [ "$first_time_dataset_prep" = true ]; then
    echo "Preparing training dataset..."
    python $VERL_REPO_ROOT/examples/data_preprocess/dapo_multiturn_w_tool.py \
        --local_save_dir $DATA_ROOT/dataset/dapo/ 
    echo "Training dataset prepared."

    echo "Preparing testing dataset..."
    python $VERL_REPO_ROOT/examples/data_preprocess/aime2024_multiturn_w_tool.py \
        --local_save_dir $DATA_ROOT/dataset/test/aime_24/
    echo "Testing dataset prepared."

    echo "Dataset preparation completed."
fi

train_files=$DATA_ROOT/dataset/dapo/train.parquet
test_files=$DATA_ROOT/dataset/test/aime_24/train.parquet

actor_model_path=Qwen/Qwen3-30B-A3B-Base
critic_model_path=$actor_model_path

max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 8))
enable_overlong_buffer=True
overlong_buffer_len=$((1024 * 4))
overlong_penalty_factor=1.0

train_batch_size=256
ppo_mini_batch_size=32
n_resp_per_prompt=16
n_resp_per_prompt_val=1

# ===================================== Training =====================================
actor_max_token_len_per_gpu=$(((max_prompt_length + max_response_length) * 3))
critic_max_token_len_per_gpu=$(((max_prompt_length + max_response_length) * 4))

enable_gradient_checkpointing=True
param_offload=False
optimizer_offload=False


VAL_BEFORE_TRAIN=False
SAVE_FREQ=-1 # we do not save!
TEST_FREQ=10
TOTAL_EPOCHS=10
TOTAL_TRAINING_STEPS=2000

# FSDP parallelism config
USP_SIZE=4
ACTOR_FSDP_CONFIG="
    actor_rollout_ref.actor.fsdp_config.strategy=$backend \
    actor_rollout_ref.actor.fsdp_config.param_offload=$param_offload \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$optimizer_offload \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$USP_SIZE"

# Megatron parallelism config
TP_SIZE=1
CP_SIZE=1
PP_SIZE=1
VPP_SIZE=null
EP_SIZE=8
ETP_SIZE=1
ACTOR_MEGATRON_CONFIG="
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=$TP_SIZE \
    actor_rollout_ref.actor.megatron.context_parallel_size=$CP_SIZE \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=$PP_SIZE \
    actor_rollout_ref.actor.megatron.virtual_pipeline_model_parallel_size=$VPP_SIZE \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=$EP_SIZE \
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=$ETP_SIZE \
    actor_rollout_ref.actor.megatron.param_offload=True \
    actor_rollout_ref.actor.megatron.grad_offload=True \
    actor_rollout_ref.actor.megatron.optimizer_offload=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype=fp32 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=True \
    actor_rollout_ref.actor.megatron.use_mbridge=True"

# Actor model config
ACTOR_CONFIG="
    actor_rollout_ref.actor.optim.lr=$actor_lr \
    actor_rollout_ref.model.path=$actor_model_path \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=${enable_gradient_checkpointing} \
    actor_rollout_ref.actor.use_kl_loss=$use_kl_loss \
    actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
    actor_rollout_ref.actor.tau_pos=$tau_pos \
    actor_rollout_ref.actor.tau_neg=$tau_neg \
    actor_rollout_ref.actor.policy_loss.loss_mode=${loss_mode}
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$actor_max_token_len_per_gpu"

# Critic model config
CIRITC_CONFIG="
    critic.optim.lr=$critic_lr \
    critic.model.path=$critic_model_path \
    critic.model.use_remove_padding=True \
    critic.ppo_max_token_len_per_gpu=$critic_max_token_len_per_gpu \
    critic.ulysses_sequence_parallel_size=$USP_SIZE"

CRITIC_FSDP_CONFIG="${ACTOR_FSDP_CONFIG//actor_rollout_ref.actor/critic.model}"
CRITIC_MEGATRON_CONFIG="${ACTOR_MEGATRON_CONFIG//actor_rollout_ref.actor/critic}"

if [[ $backend == "megatron" ]]; then
    CONFIG_NAME=ppo_megatron_trainer
    ACTOR_CONFIG="$ACTOR_CONFIG $ACTOR_MEGATRON_CONFIG"
    if [[ $adv_estimator == "gae" ]]; then
        CIRITC_CONFIG="$CIRITC_CONFIG $CRITIC_MEGATRON_CONFIG"
    else
        CIRITC_CONFIG=""
    fi
else # fsdp, fsdp2
    CONFIG_NAME=ppo_trainer
    ACTOR_CONFIG="$ACTOR_CONFIG $ACTOR_FSDP_CONFIG"
    if [[ $adv_estimator == "gae" ]]; then
        CIRITC_CONFIG="$CIRITC_CONFIG $CRITIC_FSDP_CONFIG"
    else
        CIRITC_CONFIG=""
    fi
fi

# ===================================== Inference =====================================
rollout_engine=vllm
infer_tp=4
infer_dp=1
infer_ep=1
gpu_memory_utilization=0.8

ROLLOUT_CONFIG="
    actor_rollout_ref.rollout.name=$rollout_engine \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$infer_tp \
    actor_rollout_ref.rollout.data_parallel_size=$infer_dp \
    actor_rollout_ref.rollout.expert_parallel_size=$infer_ep \
    actor_rollout_ref.rollout.gpu_memory_utilization=$gpu_memory_utilization \
    actor_rollout_ref.rollout.n=$n_resp_per_prompt \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n=$n_resp_per_prompt_val"

# ===================================== Reward =====================================
REWARD_CONFIG="
    reward_model.reward_manager.name=dapo \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward_model.reward_kwargs.max_resp_len=${max_response_length}"


# ============================= Prepare RAY on Slurm ===============================

# we should activate it before we start ray to avoid errors
echo "Activating $ENV_NAME environment..."
eval "$(conda shell.bash hook)"
conda deactivate
conda activate "$ENV_NAME"
add_repo_to_pythonpath

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export RAY_memory_monitor_refresh_ms=0
export RAY_LOGGING_LEVEL=DEBUG
export HYDRA_FULL_ERROR=1

# Let Ray know how many nodes to expect
export RAY_NUM_NODES=$NNODES

# Get head node and its IP
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# Convert to IPv4 if needed
if [[ "$head_node_ip" == *" "* ]]; then
  IFS=' ' read -ra ADDR <<<"$head_node_ip"
  if [[ ${#ADDR[0]} -gt 16 ]]; then
    head_node_ip=${ADDR[1]}
  else
    head_node_ip=${ADDR[0]}
  fi
  echo "IPV6 address detected. Using IPV4: $head_node_ip"
fi

port=6379
ip_head=$head_node_ip:$port
export MASTER_ADDR=$head_node_ip
export MASTER_PORT=$port
export ip_head

echo "Starting Ray HEAD at $head_node ($ip_head)"
until nvidia-smi > /dev/null 2>&1; do
  echo "Waiting for GPU visibility..."
  sleep 2
done
srun --nodes=1 --ntasks=1 -w "$head_node" \
  ray start --head --node-ip-address="$head_node_ip" --port=$port \
  --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${SLURM_GPUS_PER_NODE}" --block &

sleep 10

worker_num=$((SLURM_JOB_NUM_NODES - 1))
for ((i = 1; i <= worker_num; i++)); do
  node_i=${nodes_array[$i]}
  echo "Starting WORKER $i at $node_i"
  until nvidia-smi > /dev/null 2>&1; do
    echo "Waiting for GPU visibility..."
    sleep 2
  done
  srun --nodes=1 --ntasks=1 -w "$node_i" \
    ray start --address "$ip_head" --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${SLURM_GPUS_PER_NODE}" --block &
  sleep 5
done

# Final launch barrier
sleep 10

# ================================= Launch Training ================================

echo "Using $SLURM_NNODES nodes for training..."

echo "==== Confirming Ray sees all GPUs ===="
python -c "import ray; ray.init(address='auto'); print(ray.cluster_resources())"
echo "==== Done checking resources ===="

# we should activate it before we start ray to avoid errors
echo "Activating $ENV_NAME environment..."
eval "$(conda shell.bash hook)"
conda deactivate
conda activate "$ENV_NAME"
add_repo_to_pythonpath

srun --overlap --nodes=${NNODES} --ntasks=1 -w "$head_node"\
    python -m verl.trainer.main_ppo \
        --config-path=./config \
        --config-name=$CONFIG_NAME \
        algorithm.adv_estimator=$adv_estimator \
        algorithm.use_kl_in_reward=$use_kl_in_reward \
        algorithm.kl_ctrl.kl_coef=$kl_coef \
        algorithm.gamma=$gae_gamma \
        algorithm.lam=$gae_lam \
        data.train_files="$train_files" \
        data.val_files="$test_files" \
        data.return_raw_chat=True \
        data.train_batch_size=$train_batch_size \
        data.max_prompt_length=$max_prompt_length \
        data.max_response_length=$max_response_length \
        data.filter_overlong_prompts=True \
        data.filter_overlong_prompts_workers=64 \
        data.truncation='error' \
        trainer.use_legacy_worker_impl=disable \
        trainer.critic_warmup=$critic_warmup \
        trainer.logger=['console','wandb'] \
        trainer.project_name=$project_name \
        trainer.experiment_name=$experiment_name \
        trainer.default_local_dir=$default_local_dir \
        trainer.n_gpus_per_node=$SLURM_GPUS_PER_NODE \
        trainer.nnodes=$NNODES \
        trainer.val_before_train=$VAL_BEFORE_TRAIN \
        trainer.log_val_generations=100 \
        trainer.save_freq=$SAVE_FREQ \
        trainer.test_freq=$TEST_FREQ \
        trainer.total_epochs=$TOTAL_EPOCHS \
        trainer.total_training_steps=$TOTAL_TRAINING_STEPS \
        $ACTOR_CONFIG \
        $CIRITC_CONFIG \
        $ROLLOUT_CONFIG \
        $REWARD_CONFIG
