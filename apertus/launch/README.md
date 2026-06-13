# Apertus Launch Scripts

The `launch/` directory contains scripts for starting Apertus training jobs with
different Slurm configurations.

## Multi-node Async Training With Sandbox Rewards

The [`multinode_async_sandbox/`](./multinode_async_sandbox/) launcher starts a
multi-node async VERL training job with sandbox-backed code rewards. It now uses
the Kubernetes sandbox service by default; code-gym is still available as an
explicit backend.

The launcher is split into three files:

- [`launch.sh`](./multinode_async_sandbox/launch.sh) configures the experiment,
  checks or starts the configured sandbox backend, and then submits the training
  job.
- [`_sandbox_scheduler.sbatch`](./multinode_async_sandbox/_sandbox_scheduler.sbatch)
  runs the code-gym scheduler and native sandbox workers, **only when
  `SANDBOX_BACKEND=codegym`** (by default the backend is `kubernetes`).
- [`_verl_training.sbatch`](./multinode_async_sandbox/_verl_training.sbatch)
  starts Ray across the training and rollout nodes, then runs
  `verl.experimental.fully_async_policy.fully_async_main`.

Training uses the
[`async.yaml`](../../verl/experimental/fully_async_policy/config/async.yaml)
configuration by default.

## Configure Paths

Before launching, check the path variables near the top of
[`launch.sh`](./multinode_async_sandbox/launch.sh). Update them if your VERL
checkout, cache directories, sandbox backend, or training data live elsewhere.

```bash
WORKING_DIR=/iopsstor/scratch/cscs/$USER/projects/verl
HOME=/iopsstor/scratch/cscs/$USER
HF_HOME=/iopsstor/scratch/cscs/$USER/huggingface
SANDBOX_BACKEND=kubernetes
KUBERNETES_SANDBOX_URL=https://sandbox-dev.swissai.svc.cscs.ch/harness-test
REASONING_GYM_DIR=/iopsstor/scratch/cscs/$USER/projects/r-gym
QA_GYM_RERANKER_URL=https://api.swissai.svc.cscs.ch/v1/score
CSCS_SERVING_API=<token>
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
- `SANDBOX_BACKEND`: `kubernetes`, `codegym`, or `none`. `none` excludes code
  tasks.
- `KUBERNETES_SANDBOX_URL`: Kubernetes sandbox service URL when using
  `SANDBOX_BACKEND=kubernetes`.
- `CODE_GYM_DIR` or `SCHEDULER_URL`: code-gym checkout or existing scheduler URL
  when using `SANDBOX_BACKEND=codegym`.
- `SANDBOX_REWARD_CONTINUOUS`: whether code rewards are fractional instead of
  binary.
- `QA_GYM_RERANKER_URL`: Qwen3 reranker score endpoint used by QA Gym rewards.
- `CSCS_SERVING_API`: bearer token used for the QA Gym reranker service.
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
### Kubernetes sandbox

The default backend is the Kubernetes sandbox. `launch.sh` checks
`KUBERNETES_SANDBOX_URL` with `curl` before submitting the training job.
The code evaluation endpoint on kubernetes is only reachable with EPFL/ETH VPNs or CSCS network.

### QA Gym reranker

QA Gym answer verification uses the Qwen3 reranker score endpoint. Set
`CSCS_SERVING_API` before launching. The launcher sends a minimal scoring
request to `QA_GYM_RERANKER_URL` before submitting the Slurm job.

```bash
curl -fsS -X POST "${QA_GYM_RERANKER_URL:-https://api.swissai.svc.cscs.ch/v1/score}" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${CSCS_SERVING_API}" \
  -d '{"model":"tomaarsen/Qwen3-Reranker-8B-seq-cls","text_1":"<|im_start|>system\n\nShould <Response_B> truthful answering <Question> base on <Response_A> truthful answer <Question>\n\n\"yes\"or\"no\".\n\n<|im_end|>\n<|im_start|>user\n\n<Question>: What is the capital of France?\n<Response_A>: Paris\n<Question>: What is the capital of France?\n<Response_B>: ","text_2":["The capital is Paris.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think><answer>\""]}'
```

### Prepare code-gym

Use this only when running with `SANDBOX_BACKEND=codegym`.

Clone the `code-gym` repository at the path configured by `CODE_GYM_DIR`:

```bash
git clone https://github.com/swiss-ai/code-gym.git /iopsstor/scratch/cscs/$USER/projects/code-gym
```

If you choose a different location, update `CODE_GYM_DIR` in `launch.sh`, or set
`SCHEDULER_URL` to reuse an already running code-gym scheduler.

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

With the default Kubernetes backend, `launch.sh` checks the configured
`KUBERNETES_SANDBOX_URL` and then submits the training job. With
`SANDBOX_BACKEND=codegym`, it submits or reuses a code-gym scheduler and injects
`SCHEDULER_URL` into training.

To reuse an already running code-gym scheduler, set `SANDBOX_BACKEND=codegym`
and `SCHEDULER_URL` before launching:

```bash
SANDBOX_BACKEND=codegym SCHEDULER_URL=http://<scheduler-node>:8000 bash launch.sh
```
