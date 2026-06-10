# Apertus Launch Scripts

The `launch/` directory contains scripts for starting Apertus training jobs with
different Slurm configurations.

## Multi-node Async Training With Sandbox Rewards

The [`multinode_async_sandbox/`](./multinode_async_sandbox/) launcher starts a
code-gym sandbox scheduler and then submits a multi-node async VERL training job
that uses the scheduler for reward evaluation.

The launcher is split into three files:

- [`launch.sh`](./multinode_async_sandbox/launch.sh) configures the experiment,
  submits the sandbox scheduler, waits for it to become reachable, and then
  submits the training job.
- [`_sandbox_scheduler.sbatch`](./multinode_async_sandbox/_sandbox_scheduler.sbatch)
  runs the code-gym scheduler and native sandbox workers.
- [`_verl_training.sbatch`](./multinode_async_sandbox/_verl_training.sbatch)
  starts Ray across the training and rollout nodes, then runs
  `verl.experimental.fully_async_policy.fully_async_main`.

Training uses the
[`async.yaml`](../../verl/experimental/fully_async_policy/config/async.yaml)
configuration by default.

## Configure Paths

Before launching, check the path variables near the top of
[`launch.sh`](./multinode_async_sandbox/launch.sh). Update them if your VERL
checkout, cache directories, code-gym checkout, or training data live elsewhere.

```bash
WORKING_DIR=/iopsstor/scratch/cscs/$USER/projects/verl
HOME=/iopsstor/scratch/cscs/$USER
HF_HOME=/iopsstor/scratch/cscs/$USER/huggingface
CODE_GYM_DIR=/iopsstor/scratch/cscs/$USER/projects/code-gym
TRAINING_DATA_DIR=/capstor/store/cscs/swissai/infra01/reasoning/data/RL-prod/apertus_demo_rl
```

The default `TRAINING_DATA_DIR` points to preprocessed parquet files. You can
also create the training data locally with
[`data_preprocess.py`](../data_preprocess.py).

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

The launcher writes scheduler and training Slurm logs into the generated
`RUN_DIR` under:

```text
${WORKING_DIR}/outputs/${PROJECT_NAME}/${RUN_NAME}
```

## Prepare gyms
### Prepare code-gym
> ⚠️ *soon to be deprecated in favor of sandbox env running on Kubernetes*

Clone the `code-gym` repository at the path configured by `CODE_GYM_DIR`:

```bash
git clone https://github.com/swiss-ai/code-gym.git /iopsstor/scratch/cscs/$USER/projects/code-gym
```

If you choose a different location, update `CODE_GYM_DIR` in `launch.sh`.

### Prepare r-gym
Clone the `r-gym` repository at the path configured by `REASONING_GYM_DIR` and checkout the `translate` branch. This repository contains more tasks than the one supported by `reasoning_gym` package.

```bash
git clone https://github.com/EduardDurech/r-gym.git /iopsstor/scratch/cscs/$USER/projects/r-gym
git checkout translate
```

If you choose a different location, update `REASONING_GYM_DIR` in `launch.sh`.
If you do not have access to this repository, set `REASONING_GYM_DIR=""` to install `reasoning-gym` from PyPI instead.

## Launch

Run the launcher from the `multinode_async_sandbox/` directory:

```bash
cd apertus/launch/multinode_async_sandbox
bash launch.sh
```

`launch.sh` submits the scheduler first, waits until it is running, builds the
scheduler URL, and then submits the training job with `SCHEDULER_URL` injected.

To reuse an already running scheduler, set `SCHEDULER_URL` before launching:

```bash
SCHEDULER_URL=http://<scheduler-node>:8000 bash launch.sh
```
