# Ascend documentation doctests

These scripts extract existing commands from
`docs/ascend_tutorial/zh/get_start/` and exercise verl's installation and GRPO
Quick Start. Test-side selectors identify the installation code blocks and
Quick Start table entries. The existing installation and Quick Start pages keep
their layout; the container and Docker guides use standard RST code-block names
for image selection, prerequisites and builds. Missing or ambiguous commands fail the check.

Read-only checks require Bash and Python 3.10 or newer, without an NPU:

```bash
bash tests/e2e/doctests/scripts/run_doctests.sh installation check
DRY_RUN=1 bash tests/e2e/doctests/scripts/run_doctests.sh quickstart fsdp2_vllm
```

`installation check` validates all registered commands and the syntax
of installation blocks. `DRY_RUN=1` checks the selected Quick Start blocks
and training script without sourcing CANN, reading models, preprocessing
data, or starting training. Set `DOCTEST_PYTHON` to select the interpreter
used by the extraction tool when `python3` is unavailable locally.

Real Quick Start runs require a compatible Ascend container, CANN/ATB,
installed verl and backend packages, a local Qwen3-0.6B model, and GSM8K input:

```bash
MODEL_PATH=/path/to/Qwen3-0.6B GSM8K_DATASET_PATH=/path/to/gsm8k \
  bash tests/e2e/doctests/scripts/run_doctests.sh quickstart fsdp2_vllm
```

Supported cases are `fsdp2_vllm`, `megatron_vllm`, `fsdp2_sglang`, and
`megatron_sglang`. Each run defaults to one GRPO step and eight visible NPU
devices; override `TOTAL_TRAINING_STEPS` and `NDEVICES_PER_NODE` as needed.
The worker runs the documented preprocessing command and uses its output
directory (`GSM8K_OUTPUT_DIR`, default `$HOME/data/gsm8k`). To reuse existing
parquet files, set **both** `TRAIN_FILE` and `TEST_FILE`; this skips preprocessing.

Source installation requires a disposable Ubuntu/openEuler container with
compatible drivers, CANN/ATB and Conda already installed. It installs system
build tools, initializes Conda, creates a new environment, runs the documented
installer, logs Python/package versions, checks imports, and executes the chosen
Quick Start case. Version logging helps diagnose mismatches; it is not a strict
comparison against the documentation's compatibility table.

For the A3 Ubuntu environment used by CI, follow the public
[CANN container source-install guide](../../../docs/ascend_tutorial/zh/get_start/container_source_install.rst).
Its image assignments are the source of truth for CI, and the Ubuntu worker
executes its system-dependency block. The guide explains how to start the
container, prepare Conda, and continue with the existing installation commands.

```bash
ALLOW_INSTALL=1 INSTALL_BACKEND=sglang INSTALL_CASE=fsdp2_sglang \
MODEL_PATH=/path/to/Qwen3-0.6B GSM8K_DATASET_PATH=/path/to/gsm8k \
  bash tests/e2e/doctests/scripts/run_doctests.sh installation source
```

`INSTALL_BACKEND` defaults to `vllm`; `INSTALL_CASE` defaults to
`fsdp2_<backend>` and must match that backend. It selects `USE_MEGATRON`
for the installer. The checkout defaults to the current repository's **HEAD
commit**, so uncommitted changes are not included in the cloned installation.
Set `VERL_REPOSITORY`/`VERL_REVISION` to test another repository or commit.
Quick Start itself uses the working checkout, including uncommitted changes.

The harness isolates the documented named Conda environment under
`INSTALL_WORKDIR/conda-envs` and redirects the documented verl clone to the
repository/revision under test. `CANN_ENV_SCRIPT`, `ATB_ENV_SCRIPT`,
`GSM8K_DATASET_PATH` and `GSM8K_OUTPUT_DIR` override local paths only during
execution; plain `doctest_helper.py extract` returns the original commands.

The automatically created temporary installation directory is deleted on exit.
Set `KEEP_INSTALL_WORKDIR=1` to retain it, or provide an empty
`INSTALL_WORKDIR` (which is always retained). Source installation also supports
`ALLOW_INSTALL=1 DRY_RUN=1` to validate the plan without installing packages.

The GitHub workflow runs CPU tool tests on relevant pull requests and covers
three installation routes in the upstream repository:

| Route | Runner | Validation |
| --- | --- | --- |
| Build the documented A3 vLLM Dockerfile | GitHub ARM64 CPU runner | Build the image with the PR merge commit, start it without NPU devices, check its verl commit and installed package metadata |
| Use the published A3 vLLM image | A3 eight-device runner | Keep image dependencies, install the PR's verl source with `--no-deps --no-build-isolation`, then run one GRPO step for FSDP2 and Megatron |
| Install from source in a CANN container | A3 eight-device runner | Create the documented Conda environment, install dependencies and run one GRPO step for each of the four combinations |

The first two routes read commands/images from the public
[Docker guide](../../../docs/ascend_tutorial/zh/get_start/dockerfile_build_guidance.rst).
Like vLLM-Ascend's documentation tests, published-image jobs use the workflow's
`container.image`; they do not invoke Docker inside the NPU container. The build
job does not push to a registry or require registry credentials. Its image stays
on the build runner: **published-image Quick Start does not validate that newly
built image on NPU**. Such validation needs a supported image transfer or publishing
path between the build runner and the NPU runner.

Build locally on a disposable ARM64 Docker host:

```bash
ALLOW_INSTALL=1 PIP_INDEX_URL=https://mirrors.huaweicloud.com/repository/pypi/simple \
  bash tests/e2e/doctests/scripts/run_doctests.sh installation docker
```

This uses the current checkout's HEAD commit, runs the documented build command
from the repository root, and retains the image locally. `ALLOW_INSTALL=1 DRY_RUN=1`
validates the command without requiring Docker. CI retains the build log as an
artifact and allows up to six hours for the full image build.

Published-image tests run before source installation, with one NPU job at a time.
One failed case does not cancel the remaining cases. Fork pull requests follow the repository's workflow
approval policy; a maintainer must approve runs when GitHub requires it.
New runs on the same ref cancel older runs. Workflows running in forks execute
only the CPU checks.

Manual runs can select one combination or `all`; selecting a vLLM case also runs
the Dockerfile build and that case's published-image test. SGLang selections only
run source installation. Configured coverage does not
mean that every combination has already passed; inspect each job's result.
Preview the exact image/case matrix without installing anything:

```bash
python3 tests/e2e/doctests/scripts/doctest_helper.py matrix
python3 tests/e2e/doctests/scripts/doctest_helper.py matrix --case fsdp2_sglang
python3 tests/e2e/doctests/scripts/doctest_helper.py matrix --method prebuilt
```

GitHub requires
the workflow to exist on the default branch before `workflow_dispatch` is
available; the pull-request trigger provides installation coverage before merge.
The CANN base image comes from the selected backend's command in the public
container guide; the workflow bootstraps Miniforge, while the original installation
documentation creates the Python environment.
The runner must expose its NPU devices and cached inputs to the container:
Qwen3-0.6B under `/root/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B` and GSM8K
under `/root/.cache/modelscope/dataset/openai/gsm8k`, matching the existing
Ascend E2E dataset cache. These shared inputs can be mounted read-only.
The workflow sets `HF_DATASETS_CACHE=/tmp/verl-doctest/huggingface/datasets`
inside the disposable container for Datasets lock files and Arrow processing
cache. It checks the model config, dataset directory, and cache write access
before installing packages. A2 and openEuler are not included in this workflow.

The installation jobs configure download sources for the A3 runner:

- Conda uses the [TUNA conda-forge mirror](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/)
  through setup-miniconda's `channel-alias` and `channels` inputs, with `defaults`
  removed. This applies to subsequent documented `conda create` commands.
- pip uses the [Huawei Cloud PyPI mirror](https://support.huaweicloud.com/mirrors_faq/mirrors_faq_0010.html)
  through `PIP_INDEX_URL`, including isolated build dependency installation.
  The installers retain their explicit Triton-Ascend extra index.
- CANN container images already use Huawei Cloud SWR in China, as specified in
  the public container guide. A Docker Hub mirror does not replace this registry.

The Docker build also receives `PIP_INDEX_URL` as a build argument; it does not use
Conda. Published-image Quick Start reuses dependencies and does not install them.
Miniforge remains pinned to `26.9.4-1` and downloads from GitHub Releases; Git
clones still use the URLs in the documented installers. Package mirrors do not
resolve failures in those GitHub downloads or native operator builds.

```bash
python3 -m pytest -q tests/special_sanity/test_doctest_helper.py
```

On Windows, the Bash integration tests can use Git Bash via `DOCTEST_TEST_BASH`.
These CPU checks validate test orchestration; an actual installation/training
run is still required on Ascend hardware.
