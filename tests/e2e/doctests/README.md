# Ascend documentation doctests

These scripts extract existing commands from
`docs/ascend_tutorial/zh/get_start/` and exercise verl's installation and GRPO
Quick Start. Test-side selectors identify the installation code blocks and
Quick Start table entries; the user documentation needs no markers or layout
changes. Missing or ambiguous commands fail the check.

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

The GitHub workflow runs CPU tool tests on relevant pull requests. Manual runs
add source installation and one GRPO step on the existing A3 eight-device runner.
The CANN base image comes from the selected backend's Dockerfile; the workflow
bootstraps Miniforge, while the documentation creates the Python environment.
The runner must expose its NPU devices and cached Qwen3-0.6B/GSM8K data to the
container. A2 and openEuler are not included in this workflow.

```bash
python3 -m pytest -q tests/special_sanity/test_doctest_helper.py
```

On Windows, the Bash integration tests can use Git Bash via `DOCTEST_TEST_BASH`.
These CPU checks validate test orchestration; an actual installation/training
run is still required on Ascend hardware.
