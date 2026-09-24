# Ascend documentation doctests

These scripts extract marked RST code blocks from
`docs/ascend_tutorial/get_start/` and exercise verl's installation and GRPO
Quick Start. The dispatcher structure is reused from vllm-ascend; the workers
use verl's training/rollout combinations.

Read-only checks require Bash and Python 3.10 or newer, without an NPU:

```bash
bash tests/e2e/doctests/scripts/run_doctests.sh installation check
DRY_RUN=1 bash tests/e2e/doctests/scripts/run_doctests.sh quickstart fsdp2_vllm
```

`installation check` validates all registered markers/config and the syntax
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
installer, checks imports, and executes the chosen Quick Start case:

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

The automatically created temporary installation directory is deleted on exit.
Set `KEEP_INSTALL_WORKDIR=1` to retain it, or provide an empty
`INSTALL_WORKDIR` (which is always retained). Source installation also supports
`ALLOW_INSTALL=1 DRY_RUN=1` to validate the plan without installing packages.

The manual GitHub workflow runs CPU tool tests, uses the helper/config to select
the matching A3 Ubuntu image, checks installation documentation, and executes
Quick Start. It does **not** run a fresh source installation. `config.json`
also describes A2 vLLM profiles for local planning; it currently contains no
openEuler or A2 SGLang profile.

```bash
python3 -m pytest -q tests/special_sanity/test_doctest_helper.py
python3 tests/e2e/doctests/scripts/doctest_helper.py plan --quickstart all --device a3 --os ubuntu
```

On Windows, the Bash integration tests can use Git Bash via `DOCTEST_TEST_BASH`.
These CPU checks validate test orchestration; an actual installation/training
run is still required on Ascend hardware.
