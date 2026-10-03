#!/bin/bash

#SBATCH --job-name=${JOB_NAME}
#SBATCH --nodes=${NUM_NODES}
#SBATCH --ntasks-per-node=2
#SBATCH --mem=200G
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --output=${LOGDIR}/train.log
#SBATCH --error=${LOGDIR}/train.log
#SBATCH --comment=${LOGDIR}/train.log
#SBATCH --exclusive
#SBATCH --chdir=${CODE_DIR}
#SBATCH --open-mode=append

# This should be very early to capture the difference between the delays of anything in this file and a scheduling delay.
echo "[SBATCH $(date +%Y-%m-%dT%H:%M:%S.%6N)] Starting"

# load necessary modules
### Run this setup
# [Cluster]: Use conda environment
# conda activate verl

### Project

# echo $PWD
verl_workdir="${HOME}/verl"
export TRANSFORMERS_CACHE="${HOME}/.cache/huggingface"
export HF_HOME=$TRANSFORMERS_CACHE
export DATA_PATH="${DATA_PATH}"
### Cluster Network Setting

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_USE_V1=0  # Disable V1: not compatible with multinode training
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# requeue on sigusr1
handle_sigusr1() {
  echo "[SBATCH $(date +%Y-%m-%dT%H:%M:%S.%N)] Received SIGUSR1 signal, preparing to requeue..."
  exit 7
}

handle_sigterm() {
  exit_code=$?
  export SLURM_TERMINATED_ME=1
  echo "[SBATCH $(date +%Y-%m-%dT%H:%M:%S.%6N)] received a SIGTERM."
  if [[ "${exit_code}" == "0" ]]; then
    echo "[SBATCH $(date +%Y-%m-%dT%H:%M:%S.%6N)] Overriding exit code to 143 to propagate termination"
    exit_code=143
  fi
  exit "${exit_code}"
}

# trap signals
trap 'handle_sigusr1' SIGUSR1
trap 'handle_sigterm' SIGTERM

### Ray launch the nodes before training

# Getting the node names
nodes_array=($(scontrol show hostnames "$SLURM_JOB_NODELIST" | tr '\n' ' '))

head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address 2>/dev/null | grep -oE '[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}' | head -1)

echo "Head node IP: $head_node_ip"

port=6379
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"

# Print out all env variables
printenv
echo "Starting HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    bash -c "eval \"\$(/home/sam/miniconda3/bin/conda shell.bash hook)\"; conda activate verl; cd ${verl_workdir}; \
        ray start --head --node-ip-address="$head_node_ip" --port=$port \
        --dashboard-port=8266 \
        --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${SLURM_GPUS_PER_NODE}" --block" &
# optional, though may be useful in certain versions of Ray < 1.0.
sleep 10

# number of nodes other than the head node
worker_num=$((SLURM_JOB_NUM_NODES - 1))

for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Debug: Starting worker on node_i = ${node_i}"
    if [ -z "$node_i" ]; then
        echo "Error: Empty node name for worker $i"
        continue
    fi
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        bash -c "eval \"\$(/home/sam/miniconda3/bin/conda shell.bash hook)\"; conda activate verl; cd ${verl_workdir}; \
            ray start --address "$ip_head" --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${SLURM_GPUS_PER_NODE}" --block" &
    sleep 5
done

# Ray initlization test (See whether any error in the above execution)
echo "Testing Ray initialization in the slurm nodes..."
bash -c 'eval "$(/home/sam/miniconda3/bin/conda shell.bash hook)"; conda activate verl; cd '"${verl_workdir}"'; python3 -c "
import ray
try:
    ray.init(address=\"auto\")
    print(\"\n=== Ray Cluster Status ===\")
    print(f\"Number of nodes: {len(ray.nodes())}\")
    for node in ray.nodes():
        print(\"Node: {}, Status: {}\".format(node[\"NodeManagerHostname\"], node[\"Alive\"]))
    ray.shutdown()
    print(\"Ray initialization successful!\")
except Exception as e:
    print(f\"Ray initialization failed: {str(e)}\")
"'
echo "=== Ray test completed ==="

echo "Start to train..."

# Fix flash_attn compatibility issue by removing it completely
echo "Removing flash_attn to avoid compatibility issues..."

MODEL_PATH="${MODEL_PATH}"

PYTHONUNBUFFERED=1 srun --overlap --nodes=${SLURM_NNODES} --ntasks=1 -w "$head_node" \
    --chdir="${verl_workdir}" \
    bash -c "eval \"\$(/home/sam/miniconda3/bin/conda shell.bash hook)\"; conda activate verl; nohup ${VERL_COMMAND}" 