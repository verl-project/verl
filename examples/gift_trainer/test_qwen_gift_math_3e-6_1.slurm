#!/bin/bash
#SBATCH --job-name=rlvr_test
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
# SBATCH --mem=200G
#SBATCH --mem-per-cpu=0
#SBATCH --time=00:00:00
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --output=/mnt/vast/home/jimmy/logs/slurm-%j.out
#SBATCH --error=/mnt/vast/home/jimmy/logs/slurm-%j.err

beta=1
lr=3e-6
rollout_n=16
sp_size=4
ENV_DIR=/mnt/vast/home/jimmy/test_gift/verl_latest/verl/.venv
YOUR_RUN_NAME="GIFT_Beta_${beta}_LR_${lr}"
YOUR_PROJECT_NAME="Qwen3-chat"

math_train_path="$HOME/data/DAPO_data/dapo-math-17k.parquet"
aime_test_path="$HOME/data/DAPO_data/aime-2024.parquet"

train_files="[$math_train_path]"
test_files="[$aime_test_path]"

# Getting the node names
nodes_array=($(scontrol show hostnames "$SLURM_JOB_NODELIST" | tr '\n' ' '))

head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# if we detect a space character in the head node IP, we'll
# convert it to an ipv4 address. This step is optional.
if [[ "$head_node_ip" == *" "* ]]; then
IFS=' ' read -ra ADDR <<<"$head_node_ip"
if [[ ${#ADDR[0]} -gt 16 ]]; then
  head_node_ip=${ADDR[1]}
else
  head_node_ip=${ADDR[0]}
fi
echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
fi

port=6380
dashboard_port=8267
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"

# make sure we set environment variables before Ray initialization
# If you are using vllm<=0.6.3, you might need to set the following environment variable to avoid bugs:
# export VLLM_ATTENTION_BACKEND=XFORMERS

# Set separate cache directories for each job to avoid conflicts
export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}
export TORCHINDUCTOR_CACHE_DIR=/tmp/torch_cache_${SLURM_JOB_ID}

printenv


echo "Starting HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    bash -c "source $ENV_DIR/bin/activate && \
        ray start --head \
        --node-ip-address=\"$head_node_ip\" \
        --port=\"$port\" \
        --dashboard-port=\"$dashboard_port\" \
        --num-cpus \"${SLURM_CPUS_PER_TASK}\" \
        --num-gpus \"${SLURM_GPUS_PER_NODE}\" \
        --block" &
# optional, though may be useful in certain versions of Ray < 1.0.
sleep 10

# number of nodes other than the head node
worker_num=$((SLURM_JOB_NUM_NODES - 1))
echo "SLURM_JOB_NUM_NODES: $SLURM_JOB_NUM_NODES"
echo "worker_num: $worker_num"

for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        bash -c "source $ENV_DIR/bin/activate && \
            ray start --address \"$ip_head\" \
            --num-cpus \"${SLURM_CPUS_PER_TASK}\" \
            --num-gpus \"${SLURM_GPUS_PER_NODE}\" \
            --block" &
    sleep 5
done
sleep 10


# Ray initlization test (See whether any error in the above excution)
echo "Testing Ray initialization in the slurm nodes..."
bash -c "source $ENV_DIR/bin/activate && python3 -c '
import ray
try:
    ray.init(address=\"$ip_head\")
    print(\"\n=== Ray Cluster Status ===\")
    print(f\"Number of nodes: {len(ray.nodes())}\")
    for node in ray.nodes():
        print(\"Node: {}, Status: {}\".format(node[\"NodeManagerHostname\"], node[\"Alive\"]))
    ray.shutdown()
    print(\"Ray initialization successful!\")
except Exception as e:
    print(f\"Ray initialization failed: {str(e)}\")
'"
echo "=== Ray test completed ==="

echo "Initializing training..."

PYTHONUNBUFFERED=1 \
srun --overlap --nodes=1 --ntasks=1 \
bash -c "source '"$ENV_DIR"'/bin/activate && \
    export PYTHONPATH=/mnt/vast/home/jimmy/test_gift/verl_latest/verl:$PYTHONPATH && \
    python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gift \
    actor_rollout_ref.actor.policy_loss.loss_mode=gift \
    data.train_files=$train_files \
    data.val_files=$test_files \
    data.train_batch_size=1024 \
    data.max_prompt_length=2048 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=Qwen/Qwen3-30B-A3B-Base \
    actor_rollout_ref.actor.optim.lr=$lr \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=$beta \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=8 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.max_num_batched_tokens=6144 \
    actor_rollout_ref.rollout.n=$rollout_n \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$YOUR_PROJECT_NAME \
    trainer.experiment_name=$YOUR_RUN_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=$SLURM_NNODES \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=1 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size}"
