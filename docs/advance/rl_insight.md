# Use RL-Insight to Monitor Training

Last updated: 07/15/2026.

[RL-Insight](https://github.com/verl-project/rl-insight) provides online observability for RL training. In verl, it can receive trainer scalar metrics, async rollout engine metrics, TransferQueue metrics, and rollout state traces, then show them in Grafana dashboards managed by the RL-Insight server.

## When to Use

Use RL-Insight when you want one monitor view for:

- trainer metrics such as rewards, losses, and throughput
- async vLLM or SGLang rollout server metrics
- TransferQueue metrics when TransferQueue is enabled
- RL state timelines around rollout generation
- CPU, memory, network, and Ascend NPU hardware metrics

## Step 1: Install and Start RL-Insight

Install RL-Insight in the environment where the monitor server runs. Prefer the latest source version:

```bash
pip install "git+https://github.com/verl-project/rl-insight.git"
```

Or install a released package:

```bash
pip install "rl-insight>=0.2.0"
```

### Install monitor services

`rl-insight server install` downloads Prometheus, Tempo, and Grafana into `~/.rl-insight/services`. The machine that runs this command needs network access to **GitHub release assets** and **`dl.grafana.com`**.

If that machine can reach those hosts:

```bash
rl-insight server install
rl-insight server start
```

If it cannot (common in air-gapped or restricted clusters), download the archives on a networked machine first, copy them to the RL-Insight host, then install from a local directory that contains all three archives. `/path/to/archives` below is only an example path — use any directory you choose, as long as the three packages are placed together in that directory.

Default download URLs for `linux-amd64` (installer versions):

| Service | Version | Download URL |
| --- | --- | --- |
| Prometheus | `2.54.1` | https://github.com/prometheus/prometheus/releases/download/v2.54.1/prometheus-2.54.1.linux-amd64.tar.gz |
| Tempo | `2.6.1` | https://github.com/grafana/tempo/releases/download/v2.6.1/tempo_2.6.1_linux_amd64.tar.gz |
| Grafana | `13.0.0` | https://dl.grafana.com/oss/release/grafana-13.0.0.linux-amd64.tar.gz |

For `linux-arm64`, replace `amd64` with `arm64` in the filenames and URLs (Tempo uses `linux_arm64` in the archive name). Filenames must match exactly.

```bash
rl-insight server install --local-archive /path/to/archives
rl-insight server start
```

`rl-insight server start` prints the detected server IP, Grafana URL, and related endpoints. Use that printed IP in the steps below. By default, RL-Insight uses:

| Service | Default port | Purpose |
| --- | --- | --- |
| RL-Insight server | `18080` | Receives metrics and trace registrations |
| Prometheus | `9090` | Stores and queries metrics |
| Tempo | `3200` | Stores traces |
| Grafana | `3000` | Shows dashboards |

## Step 2: Enable RL-Insight in verl

Set the RL-Insight server address before submitting the training job. `<server-ip>` must be the IP of the machine where you ran `rl-insight server start` (the address printed by that command), and it must be reachable from the training processes:

```bash
export RL_INSIGHT_SERVER_URL="http://<server-ip>:18080"
```

For a multi-node Ray cluster, add the variable to the runtime environment file submitted with the verl job, typically `verl/trainer/runtime_env.yaml`. This propagates the RL-Insight server address to workers on every node:

```yaml
env_vars:
  RL_INSIGHT_SERVER_URL: "http://<server-ip>:18080"
```

If your launch script passes another file through `ray job submit --runtime-env`, add the variable to that file instead.

Add `rl_insight` to `trainer.logger`. When `rl_insight` is enabled, verl sets `VERL_RL_INSIGHT_ENABLE=1` and initializes the RL-Insight client in each process that uses it.

```bash
python3 -m verl.trainer.main_ppo \
    trainer.logger='["console","rl_insight"]' \
    trainer.project_name=verl \
    trainer.experiment_name=ppo_rl_insight \
    ...
```

Trainer scalar metrics are reported to RL-Insight automatically through the logger backend.

## Step 3: Monitor Rollout and TransferQueue Metrics

For rollout engine metrics and TransferQueue metrics, keep rollout stats enabled and expose the TransferQueue metrics endpoint:

```bash
python3 -m verl.trainer.main_ppo \
    trainer.logger='["console","rl_insight"]' \
    actor_rollout_ref.rollout.disable_log_stats=False \
    transfer_queue.metrics.enabled=True \
    ...
```

When rollout replicas or TransferQueue metrics endpoints start, verl registers them with RL-Insight. The generation path is also wrapped with RL-Insight state traces for vLLM and SGLang rollout workers.

## Step 4: Add Hardware Metrics (Optional)

To monitor CPU, memory, network, or Ascend NPU metrics, follow the [RL-Insight Hardware Monitoring guide](https://github.com/verl-project/rl-insight/blob/main/docs/monitor/hardware/index.md). The guide explains how to install or reuse the exporters and register their monitoring endpoints with RL-Insight.

## View Dashboards

1. Check the terminal output of `rl-insight server start` and open the printed Grafana URL. By default it is `http://<server-ip>:3000`, where `<server-ip>` is the RL-Insight host.
2. Log in with the default credentials:
   - username: `admin`
   - password: `admin`
3. In the left navigation, open **Dashboards**, then open the **RL-Insight** folder.
4. Select the dashboard that matches your run, for example:
   - `verl_trainer_v1_with_vllm_engine` for vLLM rollout
   - `verl_trainer_v1_with_sglang_engine` for SGLang rollout
5. Set the time range to a recent window such as **Last 5 minutes** / **Last 15 minutes** while training is still running.

The dashboards should include training metrics, rollout metrics, TransferQueue metrics if enabled, and rollout state timelines. Example views:

**RL state timeline (sync mode)**

![sync timeline](https://github.com/mengchengTang/verl-data/raw/master/sync_timeline.png)

**RL state timeline (separate async mode)**

![separate async timeline](https://github.com/mengchengTang/verl-data/raw/master/separate_async_timeline.png)

**Inference engine metrics across replicas**

![infer engine metric of all replicas](https://github.com/mengchengTang/verl-data/raw/master/infer_engine_metric_of_all_replica.png)

**TransferQueue metrics**

![transfer queue metric](https://github.com/mengchengTang/verl-data/raw/master/transfer_queue_metric.png)

**CPU hardware metrics**

![CPU hardware metrics](https://github.com/mengchengTang/verl-data/blob/master/cpu%E6%8C%87%E6%A0%87.png?raw=1)

## Troubleshooting

- If trainer metrics do not appear, check that `trainer.logger` contains `rl_insight` and `RL_INSIGHT_SERVER_URL` points to the machine that runs `rl-insight server start`.
- If rollout metrics do not appear, check that `actor_rollout_ref.rollout.disable_log_stats=False` is set.
- If TransferQueue metrics do not appear, check that `transfer_queue.metrics.enabled=True` is set.
- If `server install` fails to download packages, use the offline `--local-archive` path above.

For more RL-Insight server installation details, see the [RL-Insight server installation guide](https://github.com/verl-project/rl-insight/blob/main/docs/monitor/server_installation.md) and [quick start](https://github.com/verl-project/rl-insight/blob/main/docs/monitor/quick_start.md).
