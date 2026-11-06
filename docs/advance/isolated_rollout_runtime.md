# Separate Python environments for training and rollout

Last updated: 10/28/2026.

`actor_rollout_ref.rollout.runtime.py_executable` selects a pre-provisioned Python
environment for standalone vLLM rollout processes. Training, reference, reward,
teacher, and agent-loop processes keep their existing environment. Leaving the
option unset preserves the shared-environment behavior.

The initial supported topology is Linux/CUDA, the V1 `separate_async` trainer,
FSDP2 training, vLLM rollout, and the NCCL checkpoint engine, with hybrid switching
disabled. This separates Python dependencies while retaining the existing
generate/train/weight-sync pipeline.

## Provision the environments

From the same checkout on every assigned node, use the project's locked extras:

```bash
UV_PROJECT_ENVIRONMENT=/opt/verl/train uv sync --locked --extra fsdp
UV_PROJECT_ENVIRONMENT=/opt/verl/rollout uv sync --locked --extra vllm --extra cupy-cu130
```

The rollout NCCL checkpoint worker needs CuPy as well as the inference engine.
Both environments must use identical Python, Ray and PyTorch versions and the
same verl Python sources. Preflight checks compare those identities before
creating the rollout checkpoint workers or loading model weights. This feature
does not install packages at actor startup or support arbitrary mixed
PyTorch/CUDA versions. Both environments still share the host GPU driver.

## Configuration

Launch the driver with the training interpreter and add these overrides to a
working separate-async training configuration:

```yaml
ray_kwargs:
  ray_init:
    runtime_env:
      py_executable: /opt/verl/train/bin/python

actor_rollout_ref:
  hybrid_engine: false
  actor:
    strategy: fsdp2
  rollout:
    name: vllm
    nnodes: 1
    n_gpus_per_node: 1
    tensor_model_parallel_size: 1
    checkpoint_engine:
      backend: nccl
    runtime:
      py_executable: /opt/verl/rollout/bin/python
      env_vars: {}
      startup_timeout: 300.0

trainer:
  use_v1: true
  v1:
    trainer_mode: separate_async
    separate_async:
      hybrid_rollout:
        enable_switch: false
```

The absolute executable path must exist on all selected rollout nodes. Two GPUs
can validate one training rank and one TP=1 rollout replica with a small model;
larger models and TP launches require additional resources.

The override covers the HTTP actors, their engine/worker subprocesses, and the
rollout-side checkpoint workers. The replica controller does not import vLLM;
the native server actor is created by a long-lived launcher actor inside the
selected runtime and its handle is returned to the controller. The controller
retains the launcher, so idle task-worker reclamation cannot terminate the
server. Both launcher and server are explicitly non-detached: controller/job
teardown terminates them through Ray ownership. Existing async methods,
placement and weight-update semantics are retained.

## Precedence and failures

Ray's inherited working directory, code distribution, tracing and profiler
options are preserved. The role's interpreter replaces the inherited
`py_executable`; its `env_vars` overlay inherited variables. Internally assigned
rank and device variables retain priority and cannot be configured through the
role overlay. Provision environments and library paths so that training
site-packages are not injected into the rollout environment.

An inherited `pip`, `conda` or `uv` package specification conflicts with the
pre-provisioned interpreter and is rejected. Same-process hybrid workers,
colocated rollout, PD disaggregation, other inference backends, and reward or
teacher runtime overrides are unsupported in this initial version.

Each selected node is probed before model loading. Logs report the node,
executable, environment prefix, PID and core/backend versions without dumping
environment variables. Core version/source mismatches and missing backend or
transport dependencies fail startup. `startup_timeout` bounds asynchronous
preflight/server initialization. Cleanup kills known server/checkpoint actor
handles and launchers, and removes placement groups owned by the replica. Killing
a launcher also terminates a server whose return handle was lost, without
looking up or killing a different actor that already owns the requested name.
Existing synchronous placement-group allocation and worker registration retain
their normal Ray timeouts.
