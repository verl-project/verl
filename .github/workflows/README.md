### Adding a New Workflow

When adding a new workflow for continuous integration (CI), you have two runner options: a fixed runner or a machine from the vemlp.

- **Fixed Runner**: To use a fixed runner, specify it in your workflow using the `runs-on` keyword, like `runs-on: [L20x8]`. 
- **Vemlp Runner**: Opting for a Vemlp machine allows you to launch tasks elastically. 

Here is a template to assist you. This template is designed for using Vemlp machines. Currently, for each workflow, you need to create a `setup` and a `cleanup` job. When using this template, the main parts you need to modify are the `IMAGE` environment variable and the specific `job steps`.

```yaml
name: Your Default Workflow

on:
  push:
    branches:
      - main
      - v0.*
  pull_request:
    branches:
      - main
      - v0.*
    paths:
      - "**/*.py"
      - ".github/workflows/template.yml"

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: ${{ github.ref != 'refs/heads/main' }}

permissions:
  contents: read

env:
  IMAGE: "your vemlp image" # e.g. "verl-ci-cn-beijing.cr.volces.com/verlai/verl:uv.cu130"
  DYNAMIC_RUNNER_URL: "https://sd10g3clalm04ug7alq90.apigateway-cn-beijing.volceapi.com/runner" # public veFaas api

jobs:
  setup:
    if: github.repository_owner == 'verl-project'
    runs-on: ubuntu-latest
    outputs:
      runner-label: ${{ steps.create-runner.outputs.runner-label }}
      task-id: ${{ steps.create-runner.outputs.task-id }}
    steps:
      - uses: actions/checkout@v4
      - id: create-runner
        uses: volcengine/vemlp-github-runner@v1 
        with:
          mode: "create"
          faas-url: "${{ env.DYNAMIC_RUNNER_URL }}"
          image: "${{ env.DEFAULT_IMAGE }}"

  your_job:
    needs: setup
    runs-on: ["${{ needs.setup.outputs.runner-label || 'default-runner' }}"]
    env:
      # Volcengine runners reach the public internet only through this proxy.
      # Without it `uv run` cannot fetch the wheelhouse wheels (hosted as GitHub
      # release assets) on a cache miss, and hf downloads fail too. Hard-coded
      # rather than read from a secret because GitHub withholds secrets from fork
      # PRs, which left the proxy empty exactly on the runs that need it. Upper
      # case only: GitHub compares `env:` keys case-insensitively, so adding
      # `http_proxy` here is rejected as a redefinition of `HTTP_PROXY`.
      HF_ENDPOINT: "https://hf-mirror.com"
      # With the uv image (verl:uv.cu130) there is no install step: `uv run` syncs
      # .venv from the committed uv.lock on first use, offline from the baked uv
      # cache. Pick one inference engine + one training backend, e.g. `vllm
      # megatron`, `sglang fsdp`, or `cpu` for CPU-only, plus any conflict-free
      # add-ons the job needs: `math` (math-verify), `ci` (hf_transfer,
      # sglang-router), `veomni-sft`. `manage_envs.py list` shows them all.
      #
      # Keep these extras identical in every step of the job — and matching what
      # the e2e script picks for the same backend combo — or uv re-syncs torch
      # between steps.
      UV_RUN: "uv run --frozen --all-packages --extra vllm --extra megatron"
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      # ${HOME} is the runner's persistent storage, the container filesystem is
      # not: keep uv's cache there so a wheel the image did not bake is fetched
      # once per machine instead of once per job. It has to be a `run:` step —
      # a job `env:` entry would keep `$HOME` literal, GitHub does not expand it.
      - name: Keep the uv cache on the runner's persistent ${HOME}
        run: |
          echo "HOME=${HOME} (uv cache baked into the image: ${UV_CACHE_DIR})"
          echo "UV_CACHE_DIR=${HOME}/.cache/uv" | tee -a "${GITHUB_ENV}"
      # Records the exact resolved environment in the job log, so a failure can be
      # attributed to a version change without re-running the job.
      - name: Check final pip list
        run: |
          $UV_RUN uv pip list
      - name: Run your tests
        run: |
          $UV_RUN xxxx # your jobs

  cleanup:
    runs-on: ubuntu-latest
    needs: [setup, your_job]
    if: always()
    steps:
      - id: destroy-runner
        uses: volcengine/vemlp-github-runner@v1
        with:
          mode: "destroy"
          faas-url: "${{ env.DYNAMIC_RUNNER_URL }}"
          task-id: "${{ needs.setup.outputs.task-id }}"
```

### Getting an environment

No workflow on the uv image has an install step. Every command runs through
`uv run`, which syncs `.venv` from the committed `uv.lock` on first use — offline,
from the cache baked into `verl:uv.cu130`. Jobs whose work happens in the workflow
itself (unit tests, SFT, `model.yml`) declare a `UV_RUN` prefix in the job `env:`
and use it on every command.

The e2e jobs instead let their launch scripts build the prefix, so the same script
works when run by hand, and drive the Ray workers through it via
`ray_kwargs.ray_init.runtime_env.py_executable`:

| script | extras it picks |
| --- | --- |
| `run_ppo_trainer_megatron.sh` | `$ENGINE` + `megatron` + `math` |
| `ppo_trainer/run_function_reward.sh` | `$ENGINE` + `fsdp`\|`megatron` (from `STRATEGY`) |
| `run_one_step_off_policy.sh` | `vllm` + `fsdp`\|`megatron` (from `ACTOR_STRATEGY`) |
| `run_fully_async_policy_opd.sh` | `vllm` + `megatron` |
| `run_v1_colocate_async_disrm.sh` | `$ROLLOUT_NAME` + `fsdp` |

`fsdp2` rides the `fsdp` extra — only the trainer strategy differs. Set
`VERL_USE_UV=0` (as the ascend / rocm / trtllm workflows do at the top level) to
fall back to ambient python on images that ship their own torch.

`uv run` syncs **inexactly** by default (`--exact` opts into pruning). It corrects
the version of anything the lock names, but leaves packages the lock does not
mention alone. Two consequences worth knowing:

- Something layered on with `uv pip install` and *absent* from the lock survives
  later `uv run` calls — that is how `mlflow` reaches `gpu_unit_tests.yml`.
- Something layered on that the lock *does* name is reset to the locked version on
  the next `uv run`. A pinned git commit of `megatron-core` cannot be held that
  way, which is why the e2e jobs take megatron from the lock. For a one-step
  version matrix use `--with`, as `model.yml` does for transformers 4.54.1.

`manage_envs.py` is still the entry point for locking, for warming the image cache
(`prefetch`, used by `Dockerfile.uv`) and for creating a venv on a dev box
(`sync`); CI just no longer needs it. `manage_envs.py list` shows every extra.

### Adding a dependency a CI job needs

Prefer the lock over `uv pip install`: add the package to an extra in
`pyproject.toml`, run `python manage_envs.py lock`, and name that extra in the
job's `UV_RUN`. Add-on extras (`math`, `ci`, `veomni-sft`) are conflict-free, so
they compose with any backend combo and are pre-warmed into the image's uv cache
by `manage_envs.py prefetch`.

Two cases the lock cannot express, and how they are handled:

- **A one-step version matrix.** Use `uv run --with pkg==x.y`, which layers the
  version over the synced env for that step only. `model.yml` does this for
  transformers 4.54.1.
- **A package that would poison the resolution.** `mlflow` caps `pandas<3` and
  `cryptography<49`, and `ci` shares one resolution fork with every backend, so
  locking it would drag the whole project back to pandas 2.x. `gpu_unit_tests.yml`
  layers it in with `uv pip install` and relies on the inexact sync to keep it.

### Model and Dataset
To avoid CI relies on network, we pre-download dataset on a NFS on the CI machine. The path for models are \${HOME}/models and the path for dataset is \${HOME}/models/hf_data.

Being persistent, that same \${HOME} is where every uv job points `UV_CACHE_DIR`.