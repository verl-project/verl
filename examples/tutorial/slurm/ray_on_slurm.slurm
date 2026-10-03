#!/bin/bash
#SBATCH --job-name=verl-ray-on-slurm
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --mem=200G
#SBATCH --partition=your-partition
#SBATCH --time=01:00:00
#SBATCH --account=your-account
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# load necessary modules

# replace these information with your own
verl_workdir=/path/to/verl
train_files=/path/to/gsm8k/train.parquet
val_files=/path/to/gsm8k/test.parquet
apptainer_image_path=/path/to/verl-ngc.sif

# Optional: set this on multi-NIC clusters to bind Ray to a specific interface.
# For example: RAY_NETWORK_INTERFACE=ib0 sbatch ray_on_slurm.slurm
RAY_NETWORK_INTERFACE=${RAY_NETWORK_INTERFACE:-}
# replace these information with your own

get_node_ipv4() {
    local node_name=$1
    local node_ip

    node_ip=$(
        srun --overlap --nodes=1 --ntasks=1 -w "$node_name" \
            ip -4 -o addr show dev "$RAY_NETWORK_INTERFACE" scope global |
            awk 'NR == 1 {split($4, address, "/"); print address[1]}'
    )

    if [[ -z "$node_ip" ]]; then
        echo "Cannot find an IPv4 address for interface $RAY_NETWORK_INTERFACE on $node_name." >&2
        return 1
    fi

    printf '%s\n' "$node_ip"
}

# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=("$nodes")

head_node=${nodes_array[0]}
if [[ -n "$RAY_NETWORK_INTERFACE" ]]; then
    head_node_ip=$(get_node_ipv4 "$head_node") || exit 1
else
    head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
fi

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

port=6379
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"
if [[ -n "$RAY_NETWORK_INTERFACE" ]]; then
    echo "Ray network interface: $RAY_NETWORK_INTERFACE"
fi

# make sure we set environment variables before Ray initialization

printenv

echo "Starting HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    apptainer run --nv --bind $verl_workdir $apptainer_image_path \
        ray start --head --node-ip-address="$head_node_ip" --port=$port \
        --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${SLURM_GPUS_PER_NODE}" --block &
# optional, though may be useful in certain versions of Ray < 1.0.
sleep 10

# number of nodes other than the head node
worker_num=$((SLURM_JOB_NUM_NODES - 1))

for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    worker_node_ip_args=()
    if [[ -n "$RAY_NETWORK_INTERFACE" ]]; then
        node_i_ip=$(get_node_ipv4 "$node_i") || exit 1
        worker_node_ip_args=(--node-ip-address="$node_i_ip")
        echo "Starting WORKER $i at $node_i ($node_i_ip)"
    else
        echo "Starting WORKER $i at $node_i"
    fi
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        apptainer run --nv --bind $verl_workdir $apptainer_image_path \
            ray start --address "$ip_head" "${worker_node_ip_args[@]}" \
            --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${SLURM_GPUS_PER_NODE}" --block &
    sleep 5
done

PYTHONUNBUFFERED=1 srun --overlap --nodes=1 --ntasks=1 -w "$head_node" \
    apptainer run --nv --bind $verl_workdir $apptainer_image_path \
    python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    data.train_files=$train_files \
    data.val_files=$val_files \
    data.train_batch_size=256 \
    data.max_prompt_length=512 \
    data.max_response_length=256 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    critic.optim.lr=1e-5 \
    critic.model.path=Qwen/Qwen2.5-0.5B-Instruct \
    critic.ppo_micro_batch_size_per_gpu=4 \
    algorithm.use_kl_in_reward=False \
    trainer.logger=console \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node="${SLURM_GPUS_PER_NODE}" \
    trainer.nnodes="${SLURM_NNODES}" \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=15 2>&1 | tee verl_demo_slurm.log
