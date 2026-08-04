## Using `ServerManager` with Slurm

The `ServerManager` class in `atroposlib` provides built-in support for discovering and managing inference servers distributed across nodes allocated by Slurm. Here's how to use it:

**Core Concept:**

The setup assumes you have a Slurm job allocation where:
1.  One or more nodes are designated for your main "training" or orchestrator process (the script that initializes `ServerManager`).
2.  The remaining nodes in the allocation are dedicated to running the LLM inference servers (e.g., SGLang, TGI, vLLM, etc., accessible via an OpenAI-compatible API).

**How `ServerManager` Detects Servers:**

When you initialize `ServerManager` with `slurm=True`:
1.  It reads the `SLURM_JOB_NODELIST` environment variable to get the hostnames of all allocated nodes. It uses the `scontrol show hostnames` command internally.
2.  It reads the `NUM_TRAINING_NODES` environment variable. This crucial variable tells the manager how many nodes *at the beginning* of the nodelist are *reserved for the training/orchestrator process* and should **not** be treated as inference server nodes.
3.  It iterates through the hostnames *after* the first `NUM_TRAINING_NODES`. These are assumed to be the inference nodes.
4.  For each inference node, it constructs potential server URLs. By default, it assumes:
    *   Servers run on ports starting from `9000` (`9000`, `9001`, `9002`, ...).
    *   The number of server instances per node is determined by `8 // INFER_TP` (where `INFER_TP` is another environment variable, defaulting to 1 if not set, implying 8 servers per node). You should set `INFER_TP` according to your inference server's tensor parallelism configuration if applicable.
    *   The URL format is `http://{node_hostname}:{port}/v1`.
5.  It uses the *first* configuration object you pass in the `configs` list as a template (for settings like `timeout`, `num_max_requests_at_once`, etc.) and creates specific `APIServerConfig` objects for each discovered URL.
6.  The `ServerManager` then load-balances requests across these automatically configured `OpenAIServer` instances.

**Setup Steps:**

1.  **Launch Inference Servers:** In your Slurm submission script (`sbatch`), launch your inference server instances on the designated inference nodes.
    *   Ensure they listen on the correct hostname and the expected ports (9000, 9001, ...).
    *   The number of instances per node should match the `8 // INFER_TP` logic. Adjust the port range or `INFER_TP` environment variable accordingly if your setup differs.
    *   You might use `srun` to launch these processes on specific nodes.
2.  **Set Environment Variables:** In the part of your Slurm script that launches your *main application* (the one using `ServerManager`):
    *   `export NUM_TRAINING_NODES=<number_of_non_inference_nodes>` (e.g., `export NUM_TRAINING_NODES=1` if only the first node runs the main script).
    *   `export INFER_TP=<your_tensor_parallel_size>` (Optional, defaults to 1. Set this if your inference servers use tensor parallelism and you run fewer than 8 instances per node).
3.  **Initialize `ServerManager`:** In your Python script:
    ```python
    from atroposlib.envs.server_handling.server_manager import ServerManager, ServerBaseline, APIServerConfig

    # Provide at least one config object. It will be used as a template
    # for Slurm-discovered servers if slurm=True.
    # If you pass ServerBaseline, ensure NUM_TRAINING_NODES and potentially INFER_TP are set.
    # If you pass a list of APIServerConfig, the first one is used as the template.
    base_config = ServerBaseline(
        timeout=1200,
        # other baseline settings...
    )
    # OR
    # base_config = APIServerConfig(
    #     base_url="http://dummy", # This URL is ignored when slurm=True finds nodes
    #     api_key="dummy",
    #     timeout=1200,
    #     # other config settings...
    # )

    server_manager = ServerManager(
        configs=base_config, # Or [base_config] if using APIServerConfig
        slurm=True
    )

    # Now use server_manager.chat_completion(...) or server_manager.completion(...)
    ```
4.  **Submit Slurm Job:** Submit your job ensuring the necessary nodes and resources (like GPUs for inference) are requested.

**Example Conceptual Slurm Script:**

```bash
#!/bin/bash
#SBATCH --nodes=5          # 1 trainer node + 4 inference nodes
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8  # Assuming 8 GPUs/node for inference
#SBATCH --job-name=atropos-rl

# Get allocated node hostnames
nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST)
nodes_array=($nodes)
training_node=${nodes_array[0]}
inference_nodes=${nodes_array[@]:1} # Nodes from index 1 onwards

echo "Training Node: $training_node"
echo "Inference Nodes: ${inference_nodes[@]}"

# --- Launch Inference Servers (Example using srun, adapt for your server type) ---
TP_SIZE=1 # Example: Tensor Parallelism = 1
INSTANCES_PER_NODE=$((8 / TP_SIZE))

echo "Launching $INSTANCES_PER_NODE inference servers per node..."

for node in ${inference_nodes[@]}; do
  for i in $(seq 0 $((INSTANCES_PER_NODE - 1))); do
    port=$((9000 + i))
    gpu_id=$i # Basic GPU assignment, might need refinement
    echo "Starting server on $node:$port (GPU $gpu_id)"
    srun --nodes=1 --ntasks=1 --gpus-per-task=1 --gpu-bind=map_gpu:$gpu_id --nodelist=$node \
      your_inference_server_launch_cmd --host 0.0.0.0 --port $port --tp $TP_SIZE [other_args] &
  done
done

echo "Waiting for servers to start..."
sleep 60 # Simple wait, consider a more robust check

# --- Launch W&B Watcher on each Inference Node ---
echo "Launching W&B watchers..."
# Assume the main API server runs on the training_node at default port 8000
TRAINER_API_ADDR="http://${training_node}:8000"

inference_node_index=0 # Start index for node_num
for node in ${inference_nodes[@]}; do
  echo "Starting watcher on $node (Node Index $inference_node_index)"
  srun --nodes=1 --ntasks=1 --nodelist=$node \
    python atroposlib/cli/inference_node_wandb_watcher.py \
      --api_addr $TRAINER_API_ADDR \
      --tp $TP_SIZE \
      --node_num $inference_node_index &
  inference_node_index=$((inference_node_index + 1))
done

# --- Launch Main Application on the Training Node ---
export NUM_TRAINING_NODES=1
export INFER_TP=$TP_SIZE

echo "Starting main application on $training_node..."
srun --nodes=1 --ntasks=1 --nodelist=$training_node \
  python your_main_atropos_script.py --some_arg=value

echo "Job finished."
wait # Wait for background server processes launched with '&'
```

**Important Notes:**

*   This setup relies on the `scontrol` command being available in the environment where `ServerManager` is initialized.
*   Ensure network connectivity and firewall rules allow the training node(s) to reach the inference nodes on ports 9000+.
*   The logic assumes a specific port assignment (9000+) and server count based on `INFER_TP`. If your inference server setup differs (e.g., different ports, different discovery mechanism), you would need to modify `server_manager.py` or manually provide the correct list of `APIServerConfig` objects instead of relying on `slurm=True`.

## Monitoring Inference Nodes with Weights & Biases

Atropos includes a utility script, `inference-node-wandb-watcher`, located in `atroposlib/cli/`, designed to run on each inference node alongside the inference servers.

**Purpose:**

*   **Health Monitoring:** Periodically checks the `/health_generate` endpoint of each local inference server instance (assuming ports 9000+).
*   **W&B Logging:** Logs the health status (1 for healthy, 0 for unhealthy) of each server instance to a shared Weights & Biases run group. This allows you to visualize server uptime and availability directly in your W&B dashboard alongside your training metrics.
*   **Step Synchronization:** It fetches the current training step from the main Atropos API server (`run-api`) to ensure W&B logs are correctly associated with training progress.

**Integration into Slurm Script:**

You can launch this watcher on each inference node using `srun` similarly to how the inference servers are launched. Add the following section to the example Slurm script, **after** launching the inference servers and **before** launching the main application:

```bash
# --- Launch W&B Watcher on each Inference Node ---
echo "Launching W&B watchers..."
# Assume the main API server runs on the training_node at default port 8000
TRAINER_API_ADDR="http://${training_node}:8000"

inference_node_index=0 # Start index for node_num
for node in ${inference_nodes[@]}; do
  echo "Starting watcher on $node (Node Index $inference_node_index)"
  srun --nodes=1 --ntasks=1 --nodelist=$node \
    python atroposlib/cli/inference_node_wandb_watcher.py \
      --api_addr $TRAINER_API_ADDR \
      --tp $TP_SIZE \
      --node_num $inference_node_index &
  inference_node_index=$((inference_node_index + 1))
done
```

**Explanation of Arguments:**

*   `--api_addr`: This is the address of the main Atropos API server (usually started with `run-api`). The script needs this to fetch W&B project/group info and the current training step. In the example, we construct it assuming the API runs on the `training_node` (first node in the allocation) at port `8000` (the default for `run-api`). **Ensure this port is correct and accessible from the inference nodes.**
*   `--tp`: This should be the same tensor parallelism size (`TP_SIZE`) used when launching the inference servers. It tells the watcher how many server instances (ports 9000 to 9000 + `8 // TP_SIZE` - 1) to monitor on the local node.
*   `--node_num`: A unique integer identifying this specific inference node within the Slurm job. This helps distinguish the metrics from different nodes in W&B (e.g., `server/server_heath_0_0`, `server/server_heath_1_0`). The example script assigns sequential indices starting from 0.

**Important Notes:**

*   Ensure the `run-api` server is running and accessible from the inference nodes.
*   The `inference-node-wandb-watcher` script should be executable and accessible from the inference nodes.
*   The script assumes the default port for the `run-api` server (8000). If your setup uses a different port, you may need to modify the script or the port in the `TRAINER_API_ADDR` construction.
