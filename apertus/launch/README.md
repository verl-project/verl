# Apertus Launch Scripts

The `launch/` directory contains scripts for starting Apertus training jobs with
different Slurm configurations.

## Multi-node Async Training With Sandbox Rewards

The [`multinode_async_sandbox/`](./multinode_async_sandbox/) launcher submits a
multi-node async VERL training job that uses the Kubernetes sandbox service for
code reward evaluation.

The launcher is split into two files:

- [`launch.sh`](./multinode_async_sandbox/launch.sh) configures the experiment,
  sets the Kubernetes sandbox endpoint, and submits the training job.
- [`_verl_training.sbatch`](./multinode_async_sandbox/_verl_training.sbatch)
  starts Ray across the training and rollout nodes, then runs
  `verl.experimental.fully_async_policy.fully_async_main`.

Training uses the
[`async.yaml`](../../verl/experimental/fully_async_policy/config/async.yaml)
configuration by default.

## Configure Paths

Before launching, check the path variables near the top of
[`launch.sh`](./multinode_async_sandbox/launch.sh). Update them if your VERL
checkout, cache directories, r-gym checkout, or training data live elsewhere.

```bash
WORKING_DIR=/iopsstor/scratch/cscs/$USER/projects/verl
HOME=/iopsstor/scratch/cscs/$USER
HF_HOME=/iopsstor/scratch/cscs/$USER/huggingface
REASONING_GYM_DIR=/iopsstor/scratch/cscs/$USER/projects/r-gym
TRAINING_DATA_DIR=/capstor/store/cscs/swissai/infra01/reasoning/data/RL-prod/apertus_demo_rl
KUBERNETES_SANDBOX_URL=https://sandbox-dev.swissai.svc.cscs.ch/harness-test
```

The default `TRAINING_DATA_DIR` points to preprocessed parquet files. You can
also create the training data locally with
[`data_preprocess.py`](../data_preprocess.py).

The Kubernetes sandbox endpoint is expected to expose `/evaluate` with the
scheduler-compatible request and response schema. The default endpoint is only
reachable from the allowed ETH/EPFL/CSCS network or VPN.

## Configure The Experiment

The main experiment settings are also defined in `launch.sh`. The most common
values to review are:

- `PROJECT_NAME`: output grouping under `outputs/`.
- `MODEL_NAME_OR_PATH`: model checkpoint or Hugging Face model ID.
- `TOKENIZER_NAME_OR_PATH`: tokenizer path, if it differs from the model path.
- `CONFIG_NAME`: VERL config name, defaults to `async`.
- `TRAIN_NNODES` and `ROLLOUT_NNODES`: number of training and rollout nodes.
- `SLURM_TIME`: scheduler and training job time limit.
- `ENABLE_THINKING`, `FORCE_THINKING`, and `THINK_PREFIX_TOKEN`: thinking-token
  controls for the data/chat template.
- `ROLLOUT_N`, `SEED`, and `USE_GROUP_FILTERING`: rollout and optimization
  settings.

The launcher writes training Slurm logs into the generated `RUN_DIR` under:

```text
${WORKING_DIR}/outputs/${PROJECT_NAME}/${RUN_NAME}
```

## Prepare gyms
### Prepare r-gym
Clone the `r-gym` repository at the path configured by `REASONING_GYM_DIR` and checkout the `translate` branch. This repository contains more tasks than the one supported by `reasoning_gym` package.

```bash
git clone https://github.com/EduardDurech/r-gym.git /iopsstor/scratch/cscs/$USER/projects/r-gym
git checkout translate
```

If you choose a different location, update `REASONING_GYM_DIR` in `launch.sh`.

## Launch

Run the launcher from the `multinode_async_sandbox/` directory:

```bash
cd apertus/launch/multinode_async_sandbox
bash launch.sh
```

`launch.sh` submits the training job with `KUBERNETES_SANDBOX_URL` injected.
The training script also exports the same value as `SCHEDULER_URL` for backward
compatibility with existing reward config plumbing.

To use a different Kubernetes sandbox endpoint, set `KUBERNETES_SANDBOX_URL`
before launching:

```bash
KUBERNETES_SANDBOX_URL=https://<host>/<prefix> bash launch.sh
```
