# Use Prometheus and Grafana to Monitor Rollout

**Author:** `https://github.com/meituan-search`

Last updated: 11/10/2026.

Monitor vLLM and SGLang rollout servers with Prometheus and Grafana in all V1 trainer modes: `sync`, `colocate_async`, and `separate_async`.

`LLMServerManager` registers rollout server addresses with Prometheus. In `separate_async` mode, the trainer registers the combined addresses of hybrid and standalone rollout servers.

## Overview

The historical monitoring examples below show the performance of Qwen235B on the AIME2024 dataset with a response length of 20k, where the emergence of a long-tail problem is clearly observable.

![fully_async_policy_structure](https://github.com/ArronHZG/verl-community/blob/main/docs/grafana_validate.png?raw=true)

The following figure presents the fully asynchronous training of the Qwen235B model. Here, resource idleness is distinctly noticeable, indicating that rollout resources can be reduced.

![fully_async_policy_structure](https://github.com/ArronHZG/verl-community/blob/main/docs/grafana_fully_async_train.png?raw=true)

Through the above two examples, we also illustrate the necessity of system observability.

## Architecture Overview

The overall workflow consists of the following steps:

1. **Multi-node Ray Cluster Setup**: Start Ray cluster across multiple nodes with Grafana and Prometheus information configured in environment variables on the master node
2. **Start Grafana Service**: Launch Grafana on the master node for visualization of monitoring dashboards
3. **Start Prometheus Service**: Launch Prometheus on the master node for metrics collection and storage
4. **V1 Training**: `verl.trainer.main_ppo` starts the selected V1 trainer and its rollout servers
5. **Automatic Prometheus Configuration**: verl automatically rewrites the Prometheus configuration to add monitoring for rollout servers and notifies Prometheus to reload the configuration
6. **Metrics Collection**: After program execution, metrics can be viewed in Prometheus
7. **Dashboard Visualization**: Upload and view monitoring metrics in Grafana dashboards

## Detailed Setup Steps

### Step 1: Environment Variables and Start Ray Cluster

First, set the necessary environment variables and start the Ray service.

> Reference: [configure-manage-dashboard](https://docs.ray.io/en/latest/cluster/configure-manage-dashboard.html)

```bash
# Master node environment variables
export GF_SERVER_HTTP_PORT=3000                     # Grafana service default port (customizable)
export PROMETHEUS_PORT=9090                         # Prometheus service default port (customizable)
export RAY_HEAD_PORT=6379                           # Ray master node port (customizable)
export RAY_DASHBOARD_PORT=8265                      # Ray dashboard default port (customizable)
export GRAFANA_PATHS_DATA=/tmp/grafana              # Grafana data storage directory (customizable)
export RAY_GRAFANA_HOST="http://${master_ip}:${GF_SERVER_HTTP_PORT}"        # Ray-associated Grafana address
export RAY_PROMETHEUS_HOST="http://${master_ip}:${PROMETHEUS_PORT}"         # Ray-associated Prometheus address

# Start Ray on master node
ray start --head --port=${RAY_HEAD_PORT} --dashboard-port=${RAY_DASHBOARD_PORT}

# Start Ray on worker nodes
ray start --address={master_addr}:${RAY_HEAD_PORT}
```

**Verification:** Visit `http://master_ip:8265` to confirm Ray has started successfully.

### Step 2: Start Grafana (Visualization Dashboard)

Grafana is used to display metrics collected by Prometheus (such as cache hit rate, throughput, etc.):

```bash
# Master node
nohup grafana-server \
  --config /tmp/ray/session_latest/metrics/grafana/grafana.ini \
  --homepath /usr/share/grafana \
  web > grafana.log 2>&1 &
```

**Verification:** Visit `http://master_ip:3000` to confirm Grafana has started successfully (default credentials: `admin/admin`).

If you need to change the port, modify the `GF_SERVER_HTTP_PORT` environment variable, and grafana-server will automatically recognize it.

### Step 3: Start Prometheus (Metrics Collection)

Prometheus is responsible for scraping metrics from vLLM services and storing them as time-series data:

```bash
# Master node
nohup prometheus \
  --config.file /tmp/ray/session_latest/metrics/prometheus/prometheus.yml \
  --web.enable-lifecycle \
  --web.listen-address=:${PROMETHEUS_PORT} \
  > prometheus.log 2>&1 &
```

**Verification:** Visit `http://master_ip:9090` to confirm Prometheus service has started successfully.

### Step 4 & 5: Start verl Training

Start verl training with the following parameters configured:

**Required Configuration:**

- `actor_rollout_ref.rollout.mode="async"`
- `actor_rollout_ref.rollout.disable_log_stats=False`
- `actor_rollout_ref.rollout.prometheus.enable=True`

If use default port, this parameter can be omitted.

- `actor_rollout_ref.rollout.prometheus.port=9090`

If use default path, this parameter can be omitted.

- `actor_rollout_ref.rollout.prometheus.file="/tmp/ray/session_latest/metrics/prometheus/prometheus.yml"`

`served_model_name` defaults to `actor_rollout_ref.model.path`. Set a shorter alias for dashboard labels:

- `actor_rollout_ref.rollout.prometheus.served_model_name="Qwen3-235B"`

**V1 Launch Example:**

Use this colocated async snippet in your training script and pass the usual model, dataset, and optimization overrides as script arguments. Training and rollout share the same GPU pool.

```bash
WORKING_DIR=${WORKING_DIR:-"${PWD}"}
RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/verl/trainer/runtime_env.yaml"}
ROLLOUT_NAME=${ROLLOUT_NAME:-vllm}  # vllm or sglang

ray job submit --no-wait --runtime-env="${RUNTIME_ENV}" \
    --working-dir "${WORKING_DIR}" \
    -- python3 -m verl.trainer.main_ppo \
    trainer.v1.trainer_mode=colocate_async \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=4 \
    data.return_raw_chat=True \
    actor_rollout_ref.rollout.name="${ROLLOUT_NAME}" \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.disable_log_stats=False \
    actor_rollout_ref.rollout.prometheus.enable=True \
    "$@"
```

This example uses four GPUs on one node, shared by training and rollout. Adjust `trainer.nnodes` and `trainer.n_gpus_per_node` for your cluster.

See [V1 async trainer](v1_async_trainer.md) for scheduling and weight synchronization settings.

### Step 6: View Metrics in Prometheus

After task execution, verify that Prometheus is correctly collecting metrics.

**Verification:** Visit the Prometheus interface at `http://master_ip:9090` and search for `vllm:` or `sglang:` to
confirm metrics are being reported correctly.

**Troubleshooting:**

If no metrics appear:

1. Check rollout server startup logs for the server addresses; in `separate_async`, check both hybrid and standalone servers
2. Visit `http://master_ip:server_port/metrics` to verify server metrics are available
3. Confirm that `actor_rollout_ref.rollout.disable_log_stats=False` is set

### Step 7: View Metrics in Grafana

After task execution, log in to Grafana to view and customize monitoring dashboards.

**Login:** Visit `http://master_ip:3000` (default credentials: `admin/admin`)

**Import Dashboard:**

1. Select `Dashboards` → `New` → `Import` → `Upload dashboard JSON file`
2. Upload a pre-built dashboard JSON file

**Available Dashboards:**

- [vLLM Grafana Dashboard style 1](https://github.com/ArronHZG/verl-community/blob/main/docs/grafana/vllm_grafana.json)
- [vLLM Grafana Dashboard style 2](https://github.com/vllm-project/vllm/blob/main/examples/observability/dashboards/grafana/performance_statistics.json)
- [vLLM Grafana Dashboard style 2](https://github.com/vllm-project/vllm/blob/main/examples/observability/dashboards/grafana/query_statistics.json)
- [SGLang Grafana Dashboard](https://github.com/sgl-project/sglang/blob/main/examples/monitoring/grafana/dashboards/json/sglang-dashboard.json)

## Additional Resources

- [Ray Monitoring Documentation](https://docs.ray.io/en/latest/cluster/configure-manage-dashboard.html)
- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [vLLM GitHub Repository](https://github.com/vllm-project/vllm)
- [SGLang GitHub Repository](https://github.com/sgl-project/sglang)
